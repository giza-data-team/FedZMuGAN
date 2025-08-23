from collections import OrderedDict
import torch
import os
import numpy
from client_utils.models.model_factory import ModelFactory
from config_manager import ConfigurationManager


class WeightsController:
    """
    A class to manage PyTorch model weights, including converting to/from bytes
    and storing metadata for serialization and deserialization.
    """

    def __init__(self):
        self.best_performance = float("inf")
        self.config_manager = ConfigurationManager()
        self.save_path = self.config_manager.get_models_path_original()
        os.makedirs(self.save_path, exist_ok=True)

    def get_weights(self, net):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_weights(self, net, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def save_weights(self, model_pram, model_id, dataset_name, file_path=None):
        """Save weights with model ID identifier"""
        if file_path is None:
            filename = f"{dataset_name}_{model_id}.pth"
            path = os.path.join(self.save_path, filename)

        else:
            path = file_path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Convert weights to model state dict and save
        model = ModelFactory.create_model()
        self.set_weights(model, model_pram)
        torch.save(model.state_dict(), path)

    @staticmethod
    def load_model(path, device):
        # Check if path is a directory
        if os.path.isdir(path):
            raise ValueError(f"Expected a file path, got directory: {path}")

        # Check if file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")

        try:
            model = ModelFactory.create_model().to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")
