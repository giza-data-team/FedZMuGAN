from dotenv import load_dotenv
from client_utils.general_utils import set_seed

load_dotenv()
set_seed()

import os
import torch
from torch import nn
from Machine_Unlearning.Unlearning.train.dataset import (
    zMuGANDataGenerator,
)
from Machine_Unlearning.Unlearning.train.train import UnlearningProcessor
from Machine_Unlearning.zMuGAN.generator import ZMuGANGenerator
from client_utils.models.model_factory import ModelFactory
from config_manager import ConfigurationManager
from torch.utils.data import DataLoader, TensorDataset


class UnlearningProcess:
    def __init__(self):
        print("[DEBUG] Initializing UnlearningProcess...")
        self.config_manager = ConfigurationManager()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_configurations()
        self.original_model = None
        self.unlearned_model = None

    def _load_configurations(self):
        self.seed = self.config_manager.get_seed()
        self.num_classes = self.config_manager.get_num_classes()
        self.img_channels = self.config_manager.get_img_channels()
        self.original_model_dir = self.config_manager.get_models_path_original()
        self.dataset_configs = self.config_manager.get_dataset_configs()
        self.model_name = self.config_manager.get_model_name()
        self.num_gpus = self.config_manager.get_num_gpus()
        self.zmu_model_dir = self.config_manager.get_output_path_zmugan()
        self.zmu_noise_dim = self.config_manager.get_noise_dim_zmugan()
        self.zmu_n_generators = self.config_manager.get_n_generators_zmugan()
        self.unlearning_configs = self.config_manager.get_unlearning_configs()
        self.specific_data_path = os.path.join(
            self.dataset_configs["DATASET_PATH"], self.dataset_configs["DATASET_NAME"]
        )

    def load_pretrained_model(self):
        model_factory = ModelFactory()
        self.original_model = model_factory.create_model().to(self.device)
        original_model_path_full = os.path.join(
            self.original_model_dir,
            f"{self.model_name}_{self.dataset_configs['DATASET_NAME']}.pth",
        )
        if not os.path.exists(original_model_path_full):
            raise FileNotFoundError(
                f"Pretrained model not found at {original_model_path_full}"
            )

        self.original_model.load_state_dict(
            torch.load(
                original_model_path_full, map_location=self.device, weights_only=True
            )
        )
        print("[DEBUG] Loaded pre-trained original model successfully.\n")

    def perform_unlearning(self):
        print("[DEBUG] Starting perform_unlearning...")
        unlearning_processor = UnlearningProcessor(
            self.original_model,
            self.model_name,
            self.dataset_configs["DATASET_NAME"],
            self.dataset_configs["IMAGE_SIZE"],
            self.img_channels,
            self.device,
            self.unlearning_configs,
        )

        unlearned_model_path = os.path.join(
            self.unlearning_configs["MODELS_PATH_UNLEARN"],
            f"unlearned_{self.model_name}_{self.dataset_configs['DATASET_NAME']}_forget_class_{self.unlearning_configs['FORGET_CLASS']}.pth",
        )

        self._perform_zmugan_unlearning(unlearning_processor)

        os.makedirs(self.unlearning_configs["MODELS_PATH_UNLEARN"], exist_ok=True)
        torch.save(self.unlearned_model.state_dict(), unlearned_model_path)
        print(
            f"[DEBUG] zMuGAN Unlearning process completed and model saved to {unlearned_model_path}."
        )

    def _perform_zmugan_unlearning(self, unlearning_processor):
        print("[DEBUG] Loading pre-trained zMuGAN model...")
        # calculate samples per generator
        samples_per_generator = (
            self.unlearning_configs["SAMPLES_PER_CLASS"] // self.zmu_n_generators
        )

        generators = []
        for i in range(self.zmu_n_generators):
            generator_path = os.path.join(
                self.zmu_model_dir,
                "model_weights",
                f"generator_{self.model_name}_{self.dataset_configs['DATASET_NAME']}_{i+1}.pth",
            )
            if not os.path.exists(generator_path):
                raise FileNotFoundError(
                    f"zMuGAN generator not found at {generator_path}"
                )

            z_generator = ZMuGANGenerator(
                self.num_classes, self.img_channels, self.zmu_noise_dim
            ).to(self.device)

            z_generator.load_state_dict(
                torch.load(generator_path, map_location=self.device, weights_only=True)
            )
            generators.append(z_generator)
            print(f"[DEBUG] Loaded generator {i+1}")

        # First generate a small batch to verify all classes are represented
        print("[DEBUG] Generating test batch to verify class distribution...")
        data_gen = zMuGANDataGenerator(
            generators,
            self.unlearning_configs["FORGET_CLASS"],
            self.num_classes,
            samples_per_generator,
            self.unlearning_configs["BATCH_SIZE_UNLEARN"],
            self.zmu_noise_dim,
            self.device,
            self.original_model,
        )

        generated_images = data_gen.sample_kegnet_data(
            self.unlearning_configs["SAMPLES_PER_CLASS"],
            generators,
        )
        labels = self.predict_labels(
            self.original_model,
            generated_images,
            self.unlearning_configs["BATCH_SIZE_UNLEARN"],
        )
        images_labels_pairs = zip(generated_images, labels)

        forget_dataset_images = []
        forget_dataset_labels = []

        retain_dataset_images = []
        retain_dataset_labels = []

        class_count = {}
        for index, pair in enumerate(images_labels_pairs):
            image, label_prop = pair
            predicted_label = label_prop.argmax(dim=0)

            if predicted_label.item() in class_count:
                class_count[predicted_label.item()] += 1
            else:
                class_count[predicted_label.item()] = 1

            if (predicted_label.item() == self.unlearning_configs["FORGET_CLASS"]) and (
                len(forget_dataset_images)
                < self.unlearning_configs["SAMPLES_PER_CLASS"]
            ):
                forget_dataset_images.append(image.unsqueeze(0))
                forget_dataset_labels.append(label_prop)
            elif (
                predicted_label.item() != self.unlearning_configs["FORGET_CLASS"]
            ) and (
                len(retain_dataset_images)
                < self.unlearning_configs["SAMPLES_PER_CLASS"]
            ):
                retain_dataset_images.append(image.unsqueeze(0))
                retain_dataset_labels.append(label_prop)
        print(f"[DEBUG] class_count: {class_count}")
        forget_dataset_images, forget_dataset_labels = torch.vstack(
            forget_dataset_images
        ), torch.vstack(forget_dataset_labels)
        retain_dataset_images, retain_dataset_labels = torch.vstack(
            retain_dataset_images
        ), torch.vstack(retain_dataset_labels)
        forget_data_loader = DataLoader(
            TensorDataset(forget_dataset_images, forget_dataset_labels),
            self.unlearning_configs["BATCH_SIZE_UNLEARN"],
        )
        retain_data_loader = DataLoader(
            TensorDataset(retain_dataset_images, retain_dataset_labels),
            self.unlearning_configs["BATCH_SIZE_UNLEARN"],
        )
        print("[DEBUG] DataLoaders created. Starting unlearning processor...")
        self.unlearned_model = unlearning_processor.our_method(
            forget_data_loader, retain_data_loader
        )
        print("[DEBUG] zMuGAN unlearning process completed.")

    def predict_labels(self, model, sampled_data, bs):
        """
        Predict the labels of sampled data.
        """

        print("[DEBUG] Predicting labels...")
        model.eval()
        softmax = nn.Softmax(dim=1)
        loader = DataLoader(TensorDataset(sampled_data), batch_size=bs)
        labels = []
        for (x,) in loader:
            labels.append(model(x).detach())
        return softmax(torch.cat(tuple(labels), dim=0))


if __name__ == "__main__":
    unlearning_process = UnlearningProcess()
    unlearning_process.load_pretrained_model()
    unlearning_process.perform_unlearning()
