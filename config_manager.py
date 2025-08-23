import os
from dotenv import load_dotenv
from threading import Lock


class ConfigurationManager:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConfigurationManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, env_path=".env"):
        if not self._initialized:
            self._initialized = True
            load_dotenv(env_path)  # Load environment variables from .env file

    @staticmethod
    def _str_to_bool(value):
        """
        Converts a string representation of a boolean to a Python bool.
        Strictly accepts only "True" or "False". Otherwise, raises a ValueError.
        """
        if value not in ("True", "False"):
            raise ValueError(
                f"Expected boolean to be either 'True' or 'False', got {value}"
            )
        return value == "True"

    # =============================
    # General Configurations
    # =============================
    def get_seed(self):
        return int(os.getenv("SEED", 42))
    
    def get_num_gpus(self):
        return int(os.getenv("NUM_GPUS"))

    def get_n_clients(self):
        return int(os.getenv("N_CLIENTS", 5))

    def get_n_rounds(self):
        return int(os.getenv("N_ROUNDS", 100))

    def get_port(self):
        return int(os.getenv("PORT", 1234))

    def get_homogeneous(self):
        # Note: The default is set to "True" (as a string) to comply with our strict check.
        return self._str_to_bool(os.getenv("HOMOGENEOUS", "True"))

    # =============================
    # Dataset Configurations
    # =============================
    def get_image_size(self):
        return int(os.getenv("IMAGE_SIZE", 32))

    def get_img_channels(self):
        return int(os.getenv("IMG_CHANNELS", 3))

    def get_train_split(self):
        return float(os.getenv("TRAIN_SPLIT", 0.7))

    def get_val_split(self):
        return float(os.getenv("VAL_SPLIT", 0.15))

    def get_num_classes(self):
        return int(os.getenv("NUM_CLASSES", 10))

    def get_dataset_name(self):
        return os.getenv("DATASET_NAME")

    def get_dataset_path(self):
        return os.getenv("DATASET_PATH", "./data")

    def get_patience(self):
        return int(os.getenv("PATIENCE", 10))


    # =============================
    # Original Model Configurations
    # =============================
    def get_model_name(self):
        return os.getenv("MODEL_NAME", "net")

    def get_batch_size_original(self):
        return int(os.getenv("BATCH_SIZE_ORIGINAL", 128))

    def get_epochs_original(self):
        return int(os.getenv("EPOCHS_ORIGINAL", 1))

    def get_optim_weight_decay_original(self):
        return float(os.getenv("OPTIM_WEIGHT_DECAY_ORIGINAL", 5e-4))

    def get_optim_momentum(self):
        return float(os.getenv("OPTIM_MOMENTUM_ORIGINAL", 0.9))

    def get_scheduler_t_max(self):
        return int(os.getenv("SCHEDULER_T_MAX_ORIGINAL", 200))

    def get_dropout_rate(self):
        return float(os.getenv("DROPOUT_RATE_ORIGINAL", 0.5))
    
    def get_models_path_original(self):
        return os.getenv("MODELS_PATH_ORIGINAL", "./original_model/models_weights")
    
    def get_retrain(self):
        return self._str_to_bool(os.getenv("RE_TRAIN"))
    
    def get_early_stopping_patience_original(self):
        return int(os.getenv("EARLY_STOPPING_PATIENCE_ORIGINAL"))
    
    def get_early_stopping_min_delta_original(self):
        return float(os.getenv("EARLY_STOPPING_MIN_DELTA_ORIGINAL"))
    
    def get_results_path_original(self):
        return os.getenv("RESULTS_PATH_ORIGINAL")

    def get_lr_original(self):
        return float(os.getenv("LR_ORIGINAL"))
        


    # =============================
    # zMuGAN Model Configurations
    # =============================
    def get_noise_dim_zmugan(self):
        return int(os.getenv("NOISE_DIM_ZMUGAN", 100))

    def get_batch_size_zmugan(self):
        return int(os.getenv("BATCH_SIZE_ZMUAGN", 256))

    def get_num_batches_zmugan(self):
        return int(os.getenv("NUM_BATCHES_ZMUAGN", 100))

    def get_decoder_layers_zmugan(self):
        return int(os.getenv("DECODER_LAYERS_ZMUGAN", 2))

    def get_lr_zmugan(self):
        return float(os.getenv("LR_ZMUGAN", 1e-4))

    def get_alpha_zmugan(self):
        return float(os.getenv("ALPHA_ZMUGAN", 1.0))

    def get_beta_zmugan(self):
        return float(os.getenv("BETA_ZMUGAN", 1.0))

    def get_epochs_zmugan(self):
        return int(os.getenv("EPOCHS_ZMUGAN", 4))


    def get_output_path_zmugan(self):
        return os.getenv("OUTPUT_PATH_ZMUGAN", "./zMuGAN/output")
    
    def get_n_generators_zmugan(self):
        return int(os.getenv("N_GENERATORS"))
    
    def get_early_stopping_patience_zmugan(self):
        return int(os.getenv("EARLY_STOPPING_PATIENCE_ZMUGAN"))
    
    def get_early_stopping_min_delta_zmugan(self):
        return float(os.getenv("EARLY_STOPPING_MIN_DELTA_ZMUGAN"))


    # =============================
    # Unlearning Configurations
    # =============================
    def get_forget_class(self):
        return int(os.getenv("FORGET_CLASS", 5))

    def get_samples_per_class(self):
        return int(os.getenv("SAMPLES_PER_CLASS", 2000))

    def get_lr_unlearn(self):
        return float(os.getenv("LR_UNLEARN", 0.001))

    def get_epochs_unlearn(self):
        return int(os.getenv("EPOCHS_UNLEARN", 10))

    def get_batch_size_unlearn(self):
        return int(os.getenv("BATCH_SIZE_UNLEARN", 64))

    def get_alpha_unlearn_eval(self):
        return float(os.getenv("ALPHA_UNLEARN_EVAL", 5))

    def get_generator_type(self):
        return os.getenv("GENERATOR_TYPE", "mugan_local")

    def get_models_path_unlearn(self):
        return os.getenv("MODELS_PATH_UNLEARN", "./Unlearning/unlearned_models_weights")
    
    def get_lr_ain(self):
        return float(os.getenv("LR_AIN"))
    
    def get_batch_size_ain(self):
        return int(os.getenv("BATCH_SIZE_AIN"))
    
    
    def get_scratch_models_path_unlearn(self):
        return os.getenv("SCRATCH_MODELS_PATH_UNLEARN")
    
    def get_max_epochs_unlearn_eval(self):
        return int(os.getenv("MAX_EPOCHS_UNLEARN_EVAL"))
    
    def get_results_path_unlearn(self):
        return os.getenv("RESULTS_PATH_UNLEARN")
    
    def get_results_path_unlearn_eval(self):
        return os.getenv("RESULTS_PATH_UNLEARN_EVAL")
    
    def get_unlearning_method(self):
        return os.getenv("UNLEARNING_METHOD", "zmugan")
    
    def get_lr_lipschitz(self):
        return float(os.getenv("LR_LIPSCHITZ", 0.001))
    
    def get_noise_std_lipschitz(self):
        return float(os.getenv("NOISE_STD_LIPSCHITZ", 0.1))

    # =============================
    # Grouped Configurations (Optional)
    # =============================
    
    def get_re_train(self):
        return int(os.getenv("RE_TRAIN", 0))


    def get_all_general_configs(self):
        return {
            "seed": self.get_seed(),
            "n_clients": self.get_n_clients(),
            "n_rounds": self.get_n_rounds(),
            "port": self.get_port(),
            "homogeneous": self.get_homogeneous(),
        }

    def get_all_dataset_configs(self):
        return {
            "image_size": self.get_image_size(),
            "img_channels": self.get_img_channels(),
            "train_split": self.get_train_split(),
            "val_split": self.get_val_split(),
            "num_classes": self.get_num_classes(),
            "dataset_name": self.get_dataset_name(),
            "dataset_path": self.get_dataset_path(),
        }

    def get_all_original_model_configs(self):
        return {
            "model_name": self.get_model_name(),
            "batch_size_original": self.get_batch_size_original(),
            "lr_original": self.get_lr_original(),
            "epochs_original": self.get_epochs_original(),
            "optim_weight_decay_original": self.get_optim_weight_decay_original(),
            "optim_momentum_original": self.get_optim_momentum(),
            "scheduler_t_max_original": self.get_scheduler_t_max(),
            "dropout_rate_original": self.get_dropout_rate(),
            "lr_finder_epochs_original": self.get_lr_finder_epochs_original(),
            "lr_finder_start_lr_original": self.get_lr_finder_start_lr_original(),
            "lr_finder_end_lr_original": self.get_lr_finder_end_lr_original(),
            "models_path_original": self.get_models_path_original(),
        }

    ## zMuGAN Model Parameters
    def get_zmugan_configs(self):
        return {
            "NOISE_DIM_ZMUGAN": self.get_noise_dim_zmugan(),
            "BATCH_SIZE_ZMUGAN": self.get_batch_size_zmugan(),
            "NUM_BATCHES_ZMUGAN": self.get_num_batches_zmugan(),
            "DECODER_LAYERS_ZMUGAN": self.get_decoder_layers_zmugan(),
            "LR_ZMUGAN": self.get_lr_zmugan(),
            "ALPHA_ZMUGAN": self.get_alpha_zmugan(),
            "BETA_ZMUGAN": self.get_beta_zmugan(),
            "EPOCHS_ZMUGAN": self.get_epochs_zmugan(),
            "EARLY_STOPPING_PATIENCE_ZMUGAN": self.get_early_stopping_patience_zmugan(),
            "EARLY_STOPPING_MIN_DELTA_ZMUGAN": self.get_early_stopping_min_delta_zmugan(),
            "OUTPUT_PATH_ZMUGAN": self.get_output_path_zmugan(),
            "N_GENERATORS": self.get_n_generators_zmugan()
        }

    def get_all_unlearning_configs(self):
        return {
            "forget_class": self.get_forget_class(),
            "samples_per_class": self.get_samples_per_class(),
            "lr_unlearn": self.get_lr_unlearn(),
            "epochs_unlearn": self.get_epochs_unlearn(),
            "batch_size_unlearn": self.get_batch_size_unlearn(),
            "alpha_unlearn_eval": self.get_alpha_unlearn_eval(),
            "generator_type": self.get_generator_type(),
            "models_path_unlearn": self.get_models_path_unlearn(),
        }

    def get_all_configs(self):
        """
        Return all configurations as a dictionary.
        """
        return {
            "general": self.get_all_general_configs(),
            "dataset": self.get_all_dataset_configs(),
            "original_model": self.get_all_original_model_configs(),
            "zmugan": self.get_all_zmugan_configs(),
            "unlearning": self.get_all_unlearning_configs(),
        }

    def get_all_model_configs(self):
        """
        Return all model-specific configurations as a dictionary.
        """
        return {
            "model_name": self.get_model_name(),
            "input_channels": self.get_img_channels(),
            "num_classes": self.get_num_classes(),
            "dropout_rate": self.get_dropout_rate(),
        }
    
    def get_dataset_configs(self):
        return {
            "IMAGE_SIZE": self.get_image_size(),
            "NUM_CLASSES": self.get_num_classes(),
            "IMG_CHANNELS": self.get_img_channels(),
            "VALIDATION_SPLIT": self.get_val_split(),
            "DATASET_NAME": self.get_dataset_name(),
            "DATASET_PATH": self.get_dataset_path(),
        }
    
        ## Unlearning Parameters
    def get_unlearning_configs(self):
        return {
            "FORGET_CLASS": self.get_forget_class(),
            "SAMPLES_PER_CLASS": self.get_samples_per_class(),
            "LR_UNLEARN": self.get_lr_unlearn(),
            "BATCH_SIZE_UNLEARN": self.get_batch_size_unlearn(),
            "ALPHA_UNLEARN_EVAL": self.get_alpha_unlearn_eval(),
            "MODELS_PATH_UNLEARN": self.get_models_path_unlearn(),
            "SCRATCH_MODELS_PATH_UNLEARN": self.get_scratch_models_path_unlearn(),
            "LR_AIN": self.get_lr_ain(),
            "BATCH_SIZE_AIN": self.get_batch_size_ain(),
            "RESULTS_PATH_UNLEARN": self.get_results_path_unlearn(),
            "RESULTS_PATH_UNLEARN_EVAL": self.get_results_path_unlearn_eval()
        }