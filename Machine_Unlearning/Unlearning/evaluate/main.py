from dotenv import load_dotenv
from client_utils.general_utils import set_seed

load_dotenv()
set_seed()

import os
import torch
from dotenv import load_dotenv
from Machine_Unlearning.Unlearning.evaluate.dataset import (
    prepare_forget_retain_dataloaders,
)
from Machine_Unlearning.Unlearning.metrics.membership_inference_attack import (
    get_membership_attack_prob,
)
from Machine_Unlearning.Unlearning.metrics.anamnesis_index import anamnesis_index
from original_model.train import ModelTrainer
from original_model.test import test
from original_model.dataset import Dataset
from copy import deepcopy
from client_utils.models.model_factory import ModelFactory
from config_manager import ConfigurationManager
from client_utils.general_utils import load_model, log_results_to_csv


class UnlearningEvaluator:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_configurations()

        self.original_model_path = os.path.join(
            self.model_configs["MODELS_PATH_ORIGINAL"],
            f"{self.model_configs['MODEL_NAME']}_{self.dataset_configs['DATASET_NAME']}.pt",
        )
        self.unlearned_model_path = os.path.join(
            self.unlearning_configs["MODELS_PATH_UNLEARN"],
            f"unlearned_{self.model_configs['MODEL_NAME']}_{self.dataset_configs['DATASET_NAME']}_{self.unlearning_configs['GENERATOR_TYPE']}_forget_class_{self.unlearning_configs['FORGET_CLASS']}.pt",
        )

        os.makedirs(
            self.unlearning_configs["SCRATCH_MODELS_PATH_UNLEARN"], exist_ok=True
        )
        self.scratch_model_path = os.path.join(
            self.unlearning_configs["SCRATCH_MODELS_PATH_UNLEARN"],
            f"scratch_{self.model_configs['MODEL_NAME']}_{self.dataset_configs['DATASET_NAME']}_forget_class_{self.unlearning_configs['FORGET_CLASS']}.pt",
        )

        self.dataset = Dataset(
            self.dataset_configs["VALIDATION_SPLIT"],
            self.specific_data_path,
            self.dataset_configs["DATASET_NAME"],
            self.dataset_configs["IMAGE_SIZE"],
            self.seed,
            "classifier",
        )

        self.num_classes = self.dataset_configs["NUM_CLASSES"]
        self.img_channels = self.dataset_configs["IMG_CHANNELS"]

        self.original_model = load_model(
            self.original_model_path,
            self.model_configs["MODEL_NAME"],
            self.num_classes,
            self.img_channels,
            self.device,
        )
        self.unlearned_model = load_model(self.unlearned_model_path, self.device)

        self._prepare_dataloaders()

    def _load_configurations(self):
        self.seed = self.config_manager.get_seed()
        self.dataset_configs = self.config_manager.get_dataset_configs()
        self.model_configs = self.config_manager.get_original_model_configs()
        self.learning_rate = self.config_manager.get_lrs_original()[
            self.dataset_configs["DATASET_NAME"]
        ][self.model_configs["MODEL_NAME"]]
        self.unlearning_configs = self.config_manager.get_unlearning_configs()
        self.specific_data_path = os.path.join(
            self.dataset_configs["DATASET_PATH"], self.dataset_configs["DATASET_NAME"]
        )

    def _prepare_dataloaders(self):
        self.train_loader = self.dataset.get_train_dataset(
            loader=True, batch_size=self.model_configs["BATCH_SIZE_ORIGINAL"]
        )
        self.val_loader = self.dataset.get_validate_dataset(
            loader=True, batch_size=self.model_configs["BATCH_SIZE_ORIGINAL"]
        )
        self.test_loader = self.dataset.get_test_dataset(
            loader=True, batch_size=self.model_configs["BATCH_SIZE_ORIGINAL"]
        )

        self.ain_train_loader = self.dataset.get_train_dataset(
            loader=True, batch_size=self.unlearning_configs["BATCH_SIZE_AIN"]
        )
        (
            self.ain_forget_validate_loader,
            _,
        ) = prepare_forget_retain_dataloaders(
            self.dataset,
            self.unlearning_configs["BATCH_SIZE_AIN"],
            self.unlearning_configs["FORGET_CLASS"],
            dataset_type="validate",
        )

        (
            self.forget_train_loader,
            self.retain_train_loader,
        ) = prepare_forget_retain_dataloaders(
            self.dataset,
            self.model_configs["BATCH_SIZE_ORIGINAL"],
            self.unlearning_configs["FORGET_CLASS"],
            dataset_type="train",
        )
        (
            self.forget_validate_loader,
            self.retain_validate_loader,
        ) = prepare_forget_retain_dataloaders(
            self.dataset,
            self.model_configs["BATCH_SIZE_ORIGINAL"],
            self.unlearning_configs["FORGET_CLASS"],
            dataset_type="validate",
        )
        (
            self.forget_test_loader,
            self.retain_test_loader,
        ) = prepare_forget_retain_dataloaders(
            self.dataset,
            self.model_configs["BATCH_SIZE_ORIGINAL"],
            self.unlearning_configs["FORGET_CLASS"],
            dataset_type="test",
        )

    def evaluate_original_model(self):
        _, original_forget_accuracy = test(
            self.original_model, self.forget_test_loader, self.device
        )
        _, original_retain_accuracy = test(
            self.original_model, self.retain_test_loader, self.device
        )
        _, original_forget_accuracy_val = test(
            self.original_model, self.forget_validate_loader, self.device
        )

        original_mia = get_membership_attack_prob(
            self.retain_train_loader,
            self.forget_train_loader,
            self.test_loader,
            self.original_model,
            self.device,
        )

        return (
            original_forget_accuracy_val,
            original_forget_accuracy,
            original_retain_accuracy,
            original_mia,
        )

    def train_scratch_model(self):
        print("Training scratch model on retain training data...")
        model_factory = ModelFactory()
        self.scratch_model = model_factory.create_model(
            model_name=self.model_configs["MODEL_NAME"],
            num_classes=self.num_classes,
            img_channels=self.img_channels,
        ).to(self.device)

        trainer = ModelTrainer(
            self.scratch_model,
            self.model_configs,
            self.retain_train_loader,
            self.retain_validate_loader,
            self.learning_rate,
            self.device,
            self.scratch_model_path,
            self.unlearning_configs["RESULTS_PATH_UNLEARN_EVAL"],
            self.dataset_configs["DATASET_NAME"],
            self.unlearning_configs,
            True,
        )
        (
            train_loss,
            val_loss,
            train_accuracy,
            val_accuracy,
        ) = trainer.train()

        print("Scratch model training complete.\n")
        return (
            train_loss,
            val_loss,
            train_accuracy,
            val_accuracy,
        )

    def evaluate_scratch_model(self):
        scratch_model = load_model(
            self.scratch_model_path,
            self.model_configs["MODEL_NAME"],
            self.num_classes,
            self.img_channels,
            self.device,
        )

        _, scratch_forget_accuracy = test(
            scratch_model, self.forget_test_loader, self.device
        )
        _, scratch_retain_accuracy = test(
            scratch_model, self.retain_test_loader, self.device
        )

        scratch_mia = get_membership_attack_prob(
            self.retain_train_loader,
            self.forget_train_loader,
            self.test_loader,
            scratch_model,
            self.device,
        )

        return scratch_forget_accuracy, scratch_retain_accuracy, scratch_mia

    def evaluate_unlearned_model(self, original_accuracy):
        _, unlearned_forget_accuracy = test(
            self.unlearned_model, self.forget_test_loader, self.device
        )
        _, unlearned_retain_accuracy = test(
            self.unlearned_model, self.retain_test_loader, self.device
        )

        unlearned_mia = get_membership_attack_prob(
            self.retain_train_loader,
            self.forget_train_loader,
            self.test_loader,
            self.unlearned_model,
            self.device,
        )

        scratch_model = load_model(
            self.scratch_model_path,
            self.model_configs["MODEL_NAME"],
            self.num_classes,
            self.img_channels,
            self.device,
        )

        unlearned_model_ain = deepcopy(self.unlearned_model).to(self.device)
        ain = anamnesis_index(
            model_u=unlearned_model_ain,
            model_s=scratch_model,
            train_loader=self.ain_train_loader,
            forget_val_loader=self.ain_forget_validate_loader,
            learning_rate=self.unlearning_configs["LR_AIN"],
            original_accuracy=original_accuracy,
            alpha=self.unlearning_configs["ALPHA_UNLEARN_EVAL"],
            device=self.device,
            max_steps=self.unlearning_configs["MAX_EPOCHS_UNLEARN_EVAL"],
        )

        return unlearned_forget_accuracy, unlearned_retain_accuracy, unlearned_mia, ain


if __name__ == "__main__":
    evaluator = UnlearningEvaluator()

    cfg_u = evaluator.unlearning_configs
    cfg_m = evaluator.model_configs
    cfg_d = evaluator.dataset_configs

    save_path = os.path.join(
        cfg_u["RESULTS_PATH_UNLEARN_EVAL"],
        f"{cfg_m['MODEL_NAME']}_{cfg_d['DATASET_NAME']}_forget_class_{cfg_u['FORGET_CLASS']}_{cfg_u['GENERATOR_TYPE']}_test_results.csv",
    )

    os.makedirs(cfg_u["RESULTS_PATH_UNLEARN_EVAL"], exist_ok=True)

    (
        original_forget_accuracy_val,
        original_forget_accuracy,
        original_retain_accuracy,
        original_mia,
    ) = evaluator.evaluate_original_model()
    original_mia *= 100

    log_results_to_csv(
        save_path,
        "Original",
        original_forget_accuracy,
        original_retain_accuracy,
        original_mia,
    )

    print(f"Original Membership Inference Attack: {original_mia:.2f}")
    print(f"Original Forget Accuracy: {original_forget_accuracy:.2f}%\n")
    print(f"Original Retain Accuracy: {original_retain_accuracy:.2f}%\n")

    if os.path.exists(evaluator.scratch_model_path):
        print("Scratch model already exists, skipping training")
    else:
        evaluator.train_scratch_model()

    (
        scratch_forget_accuracy,
        scratch_retain_accuracy,
        scratch_mia,
    ) = evaluator.evaluate_scratch_model()
    scratch_mia *= 100

    log_results_to_csv(
        save_path,
        "Scratch",
        scratch_forget_accuracy,
        scratch_retain_accuracy,
        scratch_mia,
    )

    print(f"Scratch Membership Inference Attack: {scratch_mia:.2f}")
    print(f"Scratch Forget Accuracy: {scratch_forget_accuracy:.2f}%\n")
    print(f"Scratch Retain Accuracy: {scratch_retain_accuracy:.2f}%\n")

    (
        unlearned_forget_accuracy,
        unlearned_retain_accuracy,
        unlearned_mia,
        ain,
    ) = evaluator.evaluate_unlearned_model(original_forget_accuracy_val)
    unlearned_mia *= 100

    log_results_to_csv(
        save_path,
        "Unlearned",
        unlearned_forget_accuracy,
        unlearned_retain_accuracy,
        unlearned_mia,
        ain,
    )

    print(f"Unlearned Model Forget Accuracy: {unlearned_forget_accuracy:.2f}%\n")
    print(f"Unlearned Model Retain Accuracy: {unlearned_retain_accuracy:.2f}%\n")
    print(f"Membership inference attack: {unlearned_mia:.2f}")
    print(f"Anamnesis Index (AIN): {ain:.4f}")
