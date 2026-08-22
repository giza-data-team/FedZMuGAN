"""
Reviewer-requested robustness study, run independently of the production
pipeline (Machine_Unlearning/Unlearning/train/train.py is left untouched):

1. f_ref seed sensitivity
   For each dataset, N_CLIENTS=10, FORGET_CLASS=3, using the already-trained
   checkpoints on disk (original model + zMuGAN generators), rerun impair+repair
   5 times with different random seeds and report mean +/- std of Acc_f / Acc_r / U.

   What "seed" actually varies here: f_ref in this codebase is not a separately
   initialized network - it is a frozen deepcopy of the already-trained model
   (see UnlearningProcessor.__init__ in train.py: `self.ref_model = deepcopy(model)`).
   So there is no random "reference model init" to reseed. The randomness the
   impair step actually depends on is x_rand ~ U(0,1) (out_r = f_ref(x_rand),
   see _impair_step below) plus the GAN proxy-sampling noise/labels used to build
   the forget/retain sets. Reseeding random/numpy/torch/cuda before each run
   varies exactly that. If "reference model init" was meant literally (a freshly
   re-initialized network standing in for f_ref), that is a different experiment
   and would need a design change to the impair step, not just a reseed - flag
   this back to the reviewers if that is what they intended.

2. Design-choice comparison
   One extra run per dataset that replaces out_r = f_ref(x_rand) with a uniform
   distribution over classes as the impair target (skips the f_ref forward pass
   entirely).

Each dataset uses its own architecture (see DATASET_CONFIG below): vgg13 for
cifar10, allcnn for svhn, resnet18 for mnist (mnist is grayscale, IMG_CHANNELS=1
for that run only - cifar10/svhn stay at 3). MODEL_NAME/IMG_CHANNELS are switched
in os.environ per dataset before that dataset's model/generators are loaded, then
every downstream read (ModelFactory, base_model.py's self.dataset_config, etc.)
picks it up since ConfigurationManager reads os.environ fresh on every call.

Requires existing checkpoints on disk for every dataset passed via --datasets:
  - {MODELS_PATH_ORIGINAL}/model_{MODEL_NAME.upper()}_{n_clients}_{dataset}.pth
  - {OUTPUT_PATH_ZMUGAN}/model_weights/generator_{MODEL_NAME}_{dataset}_{1..N_GENERATORS}.pth

Usage:
  python zmugan_f_ref_seed_sensitivity.py --datasets cifar10 svhn mnist --n_clients 10 --forget_class 3
"""

import argparse
import os

# Must be set before config_manager/dotenv-backed modules are imported.
STATIC_ENV_OVERRIDES = {
    "N_CLIENTS": "10",
    "FORGET_CLASS": "3",
}
for _key, _value in STATIC_ENV_OVERRIDES.items():
    os.environ.setdefault(_key, _value)

import random
import time
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from client_utils.load_client_data import LoadClientData
from client_utils.weights_controller import WeightsController
from config_manager import ConfigurationManager
from data_split import DataSplitter
from jit_grid_search import aggregate_results
from Machine_Unlearning.Unlearning.metrics.all_metrics import get_metrics
from Machine_Unlearning.Unlearning.train.dataset import zMuGANDataGenerator
from Machine_Unlearning.Unlearning.train.loss import DistillKL
from Machine_Unlearning.Unlearning.train.utils import predict_classes
from Machine_Unlearning.zMuGAN.generator import ZMuGANGenerator

# Per-dataset architecture + input channels. mnist is grayscale (1 channel);
# cifar10/svhn stay RGB (3 channels).
DATASET_CONFIG = {
    "cifar10": {"model_name": "vgg13", "img_channels": 3},
    "svhn": {"model_name": "allcnn", "img_channels": 3},
    "mnist": {"model_name": "resnet18", "img_channels": 1},
}
DEFAULT_DATASETS = list(DATASET_CONFIG.keys())
DEFAULT_SEEDS = [42, 123, 2024, 7, 31415]


def _seed_everything(seed):
    """Reseed every RNG source the impair/repair pipeline touches. Self-contained
    (does not call client_utils.general_utils.set_seed / mutate SEED env), so this
    script has no side effect on the rest of the pipeline's seeding behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SeedSensitivityUnlearningProcessor:
    """Standalone copy of UnlearningProcessor.our_method (impair + repair), with
    an added impair_target switch. Deliberately not importing/reusing the shared
    UnlearningProcessor so the production code path is left untouched."""

    def __init__(self, model, device, lr_unlearn, temperature, num_classes):
        self.model = model.to(device)
        self.device = device
        self.lr_unlearn = lr_unlearn
        self.temperature = temperature
        self.num_classes = num_classes

    def _impair_step(self, forget_dataloader, ref_model, criterion, impair_target):
        print(f"forget step (impair_target={impair_target})...")
        for batch in forget_dataloader:
            images, labels = batch
            out = self.model(images)

            if impair_target == "uniform":
                out_r = torch.full(
                    (images.shape[0], self.num_classes),
                    1.0 / self.num_classes,
                    device=self.device,
                )
            else:  # "f_ref"
                random_input = torch.rand(images.shape, device=self.device)
                out_r = ref_model(random_input)

            loss = criterion(out, out_r)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _repair_step(self, retain_dataloader, ref_model, criterion):
        print("retain step...")
        for batch in retain_dataloader:
            images, labels = batch
            out_r = ref_model(images)
            out = self.model(images)
            loss = criterion(out, out_r)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def run(self, forget_dataloader, retain_dataloader, impair_target="f_ref", train_with_retain=True):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr_unlearn)
        criterion = DistillKL(self.temperature)

        ref_model = deepcopy(self.model)
        ref_model.eval()

        self.model.train()
        torch.cuda.empty_cache()

        self._impair_step(forget_dataloader, ref_model, criterion, impair_target)
        if train_with_retain:
            self._repair_step(retain_dataloader, ref_model, criterion)

        return self.model


class FRefSeedSensitivityExperiment:
    """Runs impair+repair for one dataset across multiple seeds/design choices,
    reusing the already-trained original model + zMuGAN generators on disk."""

    def __init__(self, config_manager, dataset_name):
        self.config_manager = config_manager
        self.dataset_name = dataset_name
        self.model_name = config_manager.get_model_name()
        self.batch_size = config_manager.get_batch_size_unlearn()
        self.forget_classes = config_manager.get_forget_classes()
        self.forget_tag = config_manager.forget_class_tag(self.forget_classes)
        self.n_clients = config_manager.get_n_clients()
        self.num_classes = config_manager.get_num_classes()
        self.img_channels = config_manager.get_img_channels()
        self.image_size = config_manager.get_image_size()
        self.samples_per_class = config_manager.get_samples_per_class()
        self.lr_unlearn = config_manager.get_lr_unlearn()
        self.temperature = 4.0  # matches UnlearningProcessor.our_method's DistillKL(4.0)
        self.zmu_noise_dim = config_manager.get_noise_dim_zmugan()
        self.zmu_n_generators = config_manager.get_n_generators_zmugan()
        self.zmu_model_dir = config_manager.get_output_path_zmugan()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        DataSplitter(
            num_clients=self.n_clients,
            min_instances_per_client=1000,
            dataset_name=self.dataset_name,
        ).generate_and_split_data()

        model_path = os.path.join(
            self.config_manager.get_models_path_original(),
            f"model_{self.model_name.upper()}_{self.n_clients}_{self.dataset_name}.pth",
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.original_model = WeightsController().load_model(model_path, self.device)
        self.scratch_model_path = os.path.join(
            self.config_manager.get_models_path_original(),
            f"scratch_model_{self.model_name.upper()}_{self.n_clients}_{self.dataset_name}.pth",
        )

        self._generators = None
        self.load_client_data(client_id=0)

    def load_client_data(self, client_id):
        client_loader = LoadClientData(client_id=client_id)
        self.train_loader, _, self.test_loaders = client_loader.get_normal_loaders(self.batch_size)
        self.forget_train_loader, self.retain_train_loader = client_loader.get_forget_retain_loaders(
            forget_class=self.forget_classes, split="train", batch_size=self.batch_size
        )
        self.forget_test_loader, self.retain_test_loader = client_loader.get_forget_retain_loaders(
            forget_class=self.forget_classes, split="test", batch_size=self.batch_size
        )

    def _load_generators(self):
        if self._generators is not None:
            return self._generators

        generators = []
        for i in range(self.zmu_n_generators):
            generator_path = os.path.join(
                self.zmu_model_dir,
                "model_weights",
                f"generator_{self.model_name}_{self.dataset_name}_{i + 1}.pth",
            )
            if not os.path.exists(generator_path):
                raise FileNotFoundError(f"zMuGAN generator not found at {generator_path}")

            z_generator = ZMuGANGenerator(
                self.num_classes, self.img_channels, self.zmu_noise_dim, image_size=self.image_size
            ).to(self.device)
            z_generator.load_state_dict(
                torch.load(generator_path, map_location=self.device, weights_only=True)
            )
            z_generator.eval()
            generators.append(z_generator)

        self._generators = generators
        return self._generators

    def _build_forget_retain_loaders(self):
        generators = self._load_generators()
        samples_per_generator = self.samples_per_class // self.zmu_n_generators
        data_gen = zMuGANDataGenerator(
            generators,
            self.forget_classes,
            self.num_classes,
            samples_per_generator,
            self.batch_size,
            self.zmu_noise_dim,
            self.device,
            self.original_model,
        )
        generated_images = data_gen.sample_kegnet_data(self.samples_per_class, generators)
        predicted_classes = predict_classes(self.original_model, generated_images, self.device, self.batch_size)
        forget_mask = torch.isin(
            predicted_classes, torch.tensor(self.forget_classes, device=predicted_classes.device)
        )

        forget_images = generated_images[forget_mask][: self.samples_per_class]
        forget_labels = predicted_classes[forget_mask][: self.samples_per_class]
        retain_images = generated_images[~forget_mask][: self.samples_per_class]
        retain_labels = predicted_classes[~forget_mask][: self.samples_per_class]

        forget_loader = DataLoader(TensorDataset(forget_images, forget_labels), batch_size=self.batch_size)
        retain_loader = DataLoader(TensorDataset(retain_images, retain_labels), batch_size=self.batch_size)
        return forget_loader, retain_loader

    def _evaluate_model(self, unlearned_model_path):
        metrics = get_metrics(
            original_model_path=os.path.join(
                self.config_manager.get_models_path_original(),
                f"model_{self.model_name.upper()}_{self.n_clients}_{self.dataset_name}.pth",
            ),
            unlearned_model_path=unlearned_model_path,
            scratch_model_path=self.scratch_model_path,
            forget_test_loader=self.forget_test_loader,
            retain_test_loader=self.retain_test_loader,
            device=self.device,
            forget_train_loader=self.forget_train_loader,
            retain_train_loader=self.retain_train_loader,
            test_loader=self.test_loaders,
            alpha=self.config_manager.get_alpha_unlearn_eval(),
        )
        return {"forget_acc": metrics["forget_acc"], "retain_acc": metrics["retain_acc"], "mia": metrics["mia"]}

    def run_once(self, run_tag, seed, impair_target, models_dir):
        _seed_everything(seed)
        start = time.time()

        forget_loader, retain_loader = self._build_forget_retain_loaders()

        model = deepcopy(self.original_model)
        processor = SeedSensitivityUnlearningProcessor(
            model, self.device, self.lr_unlearn, self.temperature, self.num_classes
        )
        unlearned_model = processor.run(forget_loader, retain_loader, impair_target=impair_target)

        os.makedirs(models_dir, exist_ok=True)
        unlearned_model_path = os.path.join(
            models_dir,
            f"unlearned_{self.model_name}_{self.dataset_name}_forget_{self.forget_tag}_{run_tag}.pth",
        )
        torch.save(unlearned_model.state_dict(), unlearned_model_path)

        per_client_metrics = []
        for client_id in range(self.n_clients):
            self.load_client_data(client_id=client_id)
            per_client_metrics.append(self._evaluate_model(unlearned_model_path))
        avg = aggregate_results(per_client_metrics, self.n_clients)

        acc_f, acc_r = avg["forget_acc"], avg["retain_acc"]
        row = {
            "dataset": self.dataset_name,
            "model_name": self.model_name,
            "img_channels": self.img_channels,
            "run_tag": run_tag,
            "impair_target": impair_target,
            "seed": seed,
            "acc_f": acc_f,
            "acc_r": acc_r,
            "u": acc_r + (100 - acc_f),
            "mia": avg["mia"],
            "training_time_s": time.time() - start,
        }
        print(f"[{self.dataset_name}/{run_tag}] Acc_f={acc_f:.2f} Acc_r={acc_r:.2f} "
              f"U={row['u']:.2f} MIA={row['mia']:.4f}")
        return row


def summarize_seed_runs(rows):
    """Mean +/- std of Acc_f / Acc_r / U across the f_ref seed runs of one dataset."""
    df = pd.DataFrame(rows)
    summary = {"dataset": df["dataset"].iloc[0], "impair_target": "f_ref", "n_seeds": len(df)}
    for metric in ["acc_f", "acc_r", "u"]:
        summary[f"{metric}_mean"] = df[metric].mean()
        summary[f"{metric}_std"] = df[metric].std(ddof=0)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                         help=f"Datasets to run (must have existing checkpoints). Default: {DEFAULT_DATASETS}. "
                              f"Architecture/channels are looked up per dataset in DATASET_CONFIG; a dataset not "
                              f"listed there falls back to whatever MODEL_NAME/IMG_CHANNELS is in .env.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                         help=f"Seeds for the f_ref sensitivity sweep. Default: {DEFAULT_SEEDS}")
    parser.add_argument("--uniform_seed", type=int, default=None,
                         help="Seed for the extra uniform-target run. Defaults to the first --seeds value.")
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--forget_class", type=str, default="3")
    args = parser.parse_args()

    os.environ["N_CLIENTS"] = str(args.n_clients)
    os.environ["FORGET_CLASS"] = args.forget_class
    uniform_seed = args.uniform_seed if args.uniform_seed is not None else args.seeds[0]

    config_manager = ConfigurationManager()
    models_dir = config_manager.get_models_path_unlearn()

    experiment_id = f"zmugan_f_ref_seed_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = os.path.join("experiment_results", experiment_id)
    os.makedirs(results_dir, exist_ok=True)

    all_rows = []
    summary_rows = []

    for dataset_name in args.datasets:
        os.environ["DATASET_NAME"] = dataset_name
        dataset_cfg = DATASET_CONFIG.get(dataset_name)
        if dataset_cfg is not None:
            os.environ["MODEL_NAME"] = dataset_cfg["model_name"]
            os.environ["IMG_CHANNELS"] = str(dataset_cfg["img_channels"])
        else:
            print(f"[WARN] No DATASET_CONFIG entry for {dataset_name!r}; "
                  f"using MODEL_NAME/IMG_CHANNELS from .env as-is.")
        print(f"\n{'=' * 20} Dataset: {dataset_name} "
              f"(model={os.environ.get('MODEL_NAME')}, img_channels={os.environ.get('IMG_CHANNELS')}) "
              f"{'=' * 20}")

        experiment = FRefSeedSensitivityExperiment(config_manager, dataset_name)

        seed_rows = []
        for seed in args.seeds:
            row = experiment.run_once(run_tag=f"fref_seed{seed}", seed=seed,
                                       impair_target="f_ref", models_dir=models_dir)
            seed_rows.append(row)
            all_rows.append(row)
            pd.DataFrame(all_rows).to_csv(os.path.join(results_dir, "per_run_results.csv"), index=False)

        summary_rows.append(summarize_seed_runs(seed_rows))
        pd.DataFrame(summary_rows).to_csv(os.path.join(results_dir, "seed_sensitivity_summary.csv"), index=False)

        uniform_row = experiment.run_once(run_tag=f"uniform_seed{uniform_seed}", seed=uniform_seed,
                                           impair_target="uniform", models_dir=models_dir)
        all_rows.append(uniform_row)
        pd.DataFrame(all_rows).to_csv(os.path.join(results_dir, "per_run_results.csv"), index=False)

    print(f"\nDone. Per-run results: {os.path.join(results_dir, 'per_run_results.csv')}")
    print(f"Seed-sensitivity summary (mean +/- std): {os.path.join(results_dir, 'seed_sensitivity_summary.csv')}")


if __name__ == "__main__":
    main()
