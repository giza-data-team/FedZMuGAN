import os

# Static run parameters for this ablation run, set here instead of editing .env.
# Only the listed keys are overridden; every other .env value (MODEL_NAME,
# N_GENERATORS, MODELS_PATH_*, etc.) still comes from .env as usual. Must be set
# before config_manager/dotenv loads, so this block stays above the other imports.
STATIC_ENV_OVERRIDES = {
    "DATASET_NAME": "cifar10",
    "N_CLIENTS": "10",
    "FORGET_CLASS": "3",
    "LR_UNLEARN": "0.0001",
    "BATCH_SIZE_UNLEARN": "64",
    "SAMPLES_PER_CLASS": "500",
}
for _key, _value in STATIC_ENV_OVERRIDES.items():
    os.environ[_key] = _value

import argparse
import time
from copy import deepcopy
from datetime import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from client_utils.load_client_data import LoadClientData
from client_utils.weights_controller import WeightsController
from config_manager import ConfigurationManager
from data_split import DataSplitter
from jit_grid_search import aggregate_results
from Machine_Unlearning.Unlearning.metrics.all_metrics import get_metrics
from Machine_Unlearning.Unlearning.train.dataset import zMuGANDataGenerator
from Machine_Unlearning.Unlearning.train.train import UnlearningProcessor
from Machine_Unlearning.Unlearning.train.utils import predict_classes
from Machine_Unlearning.zMuGAN.generator import ZMuGANGenerator

MODES = ["full", "impair_only", "repair_only", "noise_proxy"]
DEFAULT_MODES = ["impair_only", "repair_only", "noise_proxy"]  # "full" already run separately


class ZMuGANAblationExperiment:
    """
    Runs the FedzMuGAN unlearning pipeline under 4 ablation configurations
    (Full, Impair-only, Repair-only, Noise-proxy). Loads the original model and
    zMuGAN generators directly (same style as jit_grid_search.py) so every step
    is visible/steppable in this one script. Each mode's unlearned model is
    evaluated against every client's local data (get_metrics) and
    weighted-averaged across N_CLIENTS. Logs Acc_f, Acc_r, U, MIA, ForgetScore
    per row, tagged with the mode that produced it.
    """

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.dataset_name = config_manager.get_dataset_name()
        self.model_name = config_manager.get_model_name()
        self.batch_size = config_manager.get_batch_size_unlearn()
        self.forget_classes = config_manager.get_forget_classes()
        self.forget_tag = config_manager.forget_class_tag(self.forget_classes)
        self.n_clients = config_manager.get_n_clients()
        self.num_classes = config_manager.get_num_classes()
        self.img_channels = config_manager.get_img_channels()
        self.image_size = config_manager.get_image_size()
        self.samples_per_class = config_manager.get_samples_per_class()
        self.zmu_noise_dim = config_manager.get_noise_dim_zmugan()
        self.zmu_n_generators = config_manager.get_n_generators_zmugan()
        self.zmu_model_dir = config_manager.get_output_path_zmugan()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.experiment_id = (
            f"zmugan_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.results_dir = os.path.join("experiment_results", self.experiment_id)
        os.makedirs(self.results_dir, exist_ok=True)
        self.results = []

        self._generators = None  # lazily loaded once, reused across full/impair_only/repair_only

        # (Re)generate client_data/ for the current N_CLIENTS/DATASET_NAME before
        # loading anything below. client_data/ is shared across datasets, so this
        # also re-splits (overwrites) it if you last ran a different dataset.
        data_splitter = DataSplitter(
            num_clients=self.n_clients,
            # homogeneous=config_manager.get_homogeneous(),
            min_instances_per_client=1000,
            dataset_name=self.dataset_name,
        )
        data_splitter.generate_and_split_data()

        self.load_model_and_data()

    def load_model_and_data(self, client_id=0):
        """Load the original model and prepare data loaders (same convention as jit_grid_search.py)."""
        model_path = os.path.join(
            self.config_manager.get_models_path_original(),
            f"model_{self.model_name.upper()}_{self.n_clients}_{self.dataset_name}.pth",
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        weight_controller = WeightsController()
        self.original_model = weight_controller.load_model(model_path, self.device)

        client_loader = LoadClientData(client_id=client_id)
        self.train_loader, _, self.test_loaders = client_loader.get_normal_loaders(
            self.batch_size
        )
        (
            self.forget_train_loader,
            self.retain_train_loader,
        ) = client_loader.get_forget_retain_loaders(
            forget_class=self.forget_classes, split="train", batch_size=self.batch_size
        )
        (
            self.forget_test_loader,
            self.retain_test_loader,
        ) = client_loader.get_forget_retain_loaders(
            forget_class=self.forget_classes, split="test", batch_size=self.batch_size
        )

    def _load_generators(self):
        """Load zMuGAN generator checkpoints directly. Not needed for noise_proxy mode."""
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
                self.num_classes,
                self.img_channels,
                self.zmu_noise_dim,
                image_size=self.image_size,
            ).to(self.device)
            z_generator.load_state_dict(
                torch.load(generator_path, map_location=self.device, weights_only=True)
            )
            z_generator.eval()  # BatchNorm must use learned running stats, not per-batch stats
            generators.append(z_generator)
            print(f"[DEBUG] Loaded generator {i + 1}")

        self._generators = generators
        return self._generators

    def _generate_proxy_images(self, mode):
        """
        GAN-sampled proxy images for full/impair_only/repair_only; random-noise
        images (same total count) for noise_proxy. Total is SAMPLES_PER_CLASS *
        N_GENERATORS either way, matching zMuGANDataGenerator.sample_kegnet_data.
        """
        if mode == "noise_proxy":
            total_samples = self.samples_per_class * self.zmu_n_generators
            return torch.rand(
                (total_samples, self.img_channels, self.image_size, self.image_size),
                device=self.device,
            )

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
        return data_gen.sample_kegnet_data(self.samples_per_class, generators)

    def _build_forget_retain_loaders(self, mode):
        """Generate proxy images, label them with the frozen original model, and partition."""
        generated_images = self._generate_proxy_images(mode)
        predicted_classes = predict_classes(
            self.original_model, generated_images, self.device, self.batch_size
        )
        forget_mask = torch.isin(
            predicted_classes,
            torch.tensor(self.forget_classes, device=predicted_classes.device),
        )

        forget_images = generated_images[forget_mask][: self.samples_per_class]
        forget_labels = predicted_classes[forget_mask][: self.samples_per_class]
        retain_images = generated_images[~forget_mask][: self.samples_per_class]
        retain_labels = predicted_classes[~forget_mask][: self.samples_per_class]
        print(
            f"[{mode}] proxy pool forget={forget_images.shape[0]} retain={retain_images.shape[0]}"
        )

        forget_loader = DataLoader(
            TensorDataset(forget_images, forget_labels), batch_size=self.batch_size
        )
        retain_loader = DataLoader(
            TensorDataset(retain_images, retain_labels), batch_size=self.batch_size
        )
        return forget_loader, retain_loader

    def _evaluate_model(self, unlearned_model_path):
        """Evaluate one unlearned model against the currently-loaded client's data."""
        final_metrics = get_metrics(
            original_model_path=os.path.join(
                self.config_manager.get_models_path_original(),
                f"model_{self.model_name.upper()}_{self.n_clients}_{self.dataset_name}.pth",
            ),
            unlearned_model_path=unlearned_model_path,
            scratch_model_path=os.path.join(
                self.config_manager.get_models_path_original(),
                f"scratch_model_{self.model_name.upper()}_{self.n_clients}_{self.dataset_name}.pth",
            ),
            forget_test_loader=self.forget_test_loader,
            retain_test_loader=self.retain_test_loader,
            device=self.device,
            forget_train_loader=self.forget_train_loader,
            retain_train_loader=self.retain_train_loader,
            test_loader=self.test_loaders,
            alpha=self.config_manager.get_alpha_unlearn_eval(),
        )
        return {
            "forget_acc": final_metrics["forget_acc"],
            "retain_acc": final_metrics["retain_acc"],
            "mia": final_metrics["mia"],
            "forget_score": abs(final_metrics["mia"] - 0.5),
            "anamnesis_index": final_metrics["anamnesis_index"],
        }

    def run_mode(self, mode):
        print(f"\n=== Running zMuGAN ablation mode: {mode} ===")
        start_time = time.time()

        forget_loader, retain_loader = self._build_forget_retain_loaders(mode)

        model = deepcopy(self.original_model)
        processor = UnlearningProcessor(
            model,
            self.model_name,
            self.dataset_name,
            self.image_size,
            self.img_channels,
            self.device,
            self.config_manager.get_unlearning_configs(),
        )
        unlearned_model = processor.our_method(
            forget_loader,
            retain_loader,
            train_with_retain=(mode != "impair_only"),
            run_impair=(mode != "repair_only"),
        )

        models_path_unlearn = self.config_manager.get_models_path_unlearn()
        os.makedirs(models_path_unlearn, exist_ok=True)
        unlearned_model_path = os.path.join(
            models_path_unlearn,
            f"unlearned_{self.model_name}_{self.dataset_name}_forget_class_{self.forget_tag}_{mode}.pth",
        )
        torch.save(unlearned_model.state_dict(), unlearned_model_path)

        training_time = time.time() - start_time

        # Evaluate this unlearned model against every client's local data, like jit_grid_search.py
        per_client_metrics = []
        for client_id in range(self.n_clients):
            self.load_model_and_data(client_id=client_id)
            per_client_metrics.append(self._evaluate_model(unlearned_model_path))

        avg_metrics = aggregate_results(per_client_metrics, self.n_clients)
        acc_f = avg_metrics["forget_acc"]
        acc_r = avg_metrics["retain_acc"]
        u = acc_r + (100 - acc_f)
        mia = avg_metrics["mia"]

        row = {
            "mode": mode,
            "forget_classes": self.forget_tag,
            "acc_f": acc_f,
            "acc_r": acc_r,
            "u": u,
            "mia": mia,
            "forget_score": avg_metrics["forget_score"],
            "anamnesis_index": avg_metrics["anamnesis_index"],
            "training_time": training_time,
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "n_clients": self.n_clients,
            "n_generators": self.zmu_n_generators,
            "samples_per_class": self.samples_per_class,
        }
        self.results.append(row)
        self._save_results()
        print(
            f"[{mode}] Acc_f={acc_f:.2f} Acc_r={acc_r:.2f} U={u:.2f} "
            f"MIA={mia:.4f} ForgetScore={avg_metrics['forget_score']:.4f}"
        )
        return row

    def _save_results(self):
        df = pd.DataFrame(self.results)
        df.to_csv(os.path.join(self.results_dir, "results.csv"), index=False)

    def run(self, modes):
        for mode in modes:
            self.run_mode(mode)
        results_path = os.path.join(self.results_dir, "results.csv")
        print(f"\nAblation study complete. Results saved to {results_path}")
        return self.results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "zMuGAN ablation study: Full / Impair-only / Repair-only / Noise-proxy. "
            "With no --mode flag, runs Impair-only, Repair-only, and Noise-proxy "
            "sequentially (skips Full, since it's already been run) and logs each "
            "mode's results as its own row in one CSV for later comparison."
        )
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        choices=MODES,
        default=None,
        help=(
            "Run only these ablation modes (space-separated), overriding the default "
            "of impair_only/repair_only/noise_proxy. Pass '--mode full' to (re)run Full too."
        ),
    )
    parser.add_argument(
        "--forget_class",
        type=str,
        default=None,
        help=(
            "Override FORGET_CLASS for this run, e.g. '3' or '0,1,2'. "
            "Defaults to STATIC_ENV_OVERRIDES['FORGET_CLASS'] above if not passed."
        ),
    )
    args = parser.parse_args()

    if args.forget_class is not None:
        os.environ["FORGET_CLASS"] = args.forget_class

    config_manager = ConfigurationManager()
    experiment = ZMuGANAblationExperiment(config_manager)

    modes_to_run = args.mode if args.mode else DEFAULT_MODES
    experiment.run(modes_to_run)


if __name__ == "__main__":
    main()
