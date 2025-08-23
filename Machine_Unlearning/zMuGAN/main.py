
from dotenv import load_dotenv
from client_utils.general_utils import set_seed

load_dotenv()
set_seed()

import os
import torch
from Machine_Unlearning.zMuGAN.generator import ZMuGANGenerator
from Machine_Unlearning.zMuGAN.decoder import Decoder
from Machine_Unlearning.zMuGAN.train import train
from config_manager import ConfigurationManager
from client_utils.weights_controller import WeightsController
from client_utils.general_utils import set_seed

set_seed()


class ZMuGANTrainer:
    def __init__(self, generator_index=None):
        self.seed = int(os.getenv("SEED"))
        self.config_manager = ConfigurationManager()
        self.weight_controller = WeightsController()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_configurations()
        self.cls_network = self._load_pretrained_model()
        self.gen_network, self.dec_network = self._initialize_networks()

        weights_path = os.path.join(self.zmugan_configs["OUTPUT_PATH_ZMUGAN"], "model_weights")
        images_path = os.path.join(self.zmugan_configs["OUTPUT_PATH_ZMUGAN"], "samples")

        os.makedirs(weights_path, exist_ok=True)
        os.makedirs(images_path, exist_ok=True)
        
        # If a generator_index is provided, append it to the file names to avoid overwriting.
        idx_suffix = f"_{generator_index}" if generator_index is not None else ""
        self.final_model_path = os.path.join(
            weights_path, 
            f"generator_{self.model_name}_{self.dataset_configs['DATASET_NAME']}{idx_suffix}.pth"
        )

    def _load_configurations(self):
        self.original_model_dir = self.config_manager.get_models_path_original()
        self.model_name = self.config_manager.get_model_name()
        self.dataset_configs = self.config_manager.get_dataset_configs()
        self.zmugan_configs = self.config_manager.get_zmugan_configs()
        self.num_classes = self.dataset_configs["NUM_CLASSES"]
        self.img_channels = self.dataset_configs["IMG_CHANNELS"]

    def _load_pretrained_model(self):
        print("\n=== Loading pre-trained model ===")
        original_model_path = os.path.join(
            self.original_model_dir, 
            f"{self.model_name}_{self.dataset_configs['DATASET_NAME']}.pth"
        )
        
        cls_network = self.weight_controller.load_model(original_model_path, self.device)
        return cls_network

    def _initialize_networks(self):
        print("\n=== Initializing networks ===")
        gen_network = ZMuGANGenerator(
            self.num_classes, self.img_channels, self.zmugan_configs["NOISE_DIM_ZMUGAN"]
        ).to(self.device)
        in_features = (
            self.dataset_configs["IMAGE_SIZE"] * self.dataset_configs["IMAGE_SIZE"] * self.img_channels
        )  # Flattened image size
        dec_network = Decoder(
            in_features, 
            self.zmugan_configs["NOISE_DIM_ZMUGAN"], 
            self.zmugan_configs["DECODER_LAYERS_ZMUGAN"]
        ).to(self.device)
        return gen_network, dec_network

    def train_model(self, generator_index=None):
        print("\n=== Starting Training ===")
        train(
            (self.gen_network, self.cls_network, self.dec_network),
            self.num_classes,
            self.zmugan_configs["NOISE_DIM_ZMUGAN"],
            self.zmugan_configs["BATCH_SIZE_ZMUGAN"],
            self.zmugan_configs["NUM_BATCHES_ZMUGAN"],
            self.zmugan_configs["LR_ZMUGAN"],
            self.zmugan_configs["ALPHA_ZMUGAN"],
            self.zmugan_configs["BETA_ZMUGAN"],
            self.zmugan_configs["EPOCHS_ZMUGAN"],
            self.zmugan_configs["EARLY_STOPPING_PATIENCE_ZMUGAN"],
            self.zmugan_configs["EARLY_STOPPING_MIN_DELTA_ZMUGAN"],
            self.device,
            self.zmugan_configs["OUTPUT_PATH_ZMUGAN"],
            self.model_name, 
            self.dataset_configs["DATASET_NAME"],
            self.final_model_path,
            generator_index=generator_index
        )

    def run(self, generator_index=None):
        self.train_model(generator_index)
        print("\n=== Finished training the generator ===")


if __name__ == "__main__":
    # Train 5 generators in a loop
    config = ConfigurationManager()
    n = config.get_n_generators_zmugan()
    for i in range(n):
        print(f"\n********** Training Generator {i+1} of {n} **********")
        trainer = ZMuGANTrainer(generator_index=i+1)
        trainer.run(generator_index=i+1)
