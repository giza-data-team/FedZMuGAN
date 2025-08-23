from torch import nn
from abc import ABC, abstractmethod
from config_manager import ConfigurationManager


class BaseModel(nn.Module, ABC):
    """
    BaseModel is an abstract base class that inherits from PyTorch's nn.Module and ABC.
    It provides a common interface for all models, including configuration management.
    """

    def __init__(self):
        """
        Initializes the BaseModel instance.
        Sets up the configuration manager and retrieves all model configurations.
        """
        super().__init__()
        self.config = ConfigurationManager()
        self.model_config = self.config.get_all_model_configs()
        self.dataset_config = self.config.get_all_dataset_configs()

    @abstractmethod
    def forward(self, x):
        """
        Abstract method to be implemented by subclasses.
        Defines the forward pass of the model.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        pass

    def get_config(self):
        """
        Retrieves the model configuration.

        Returns:
            dict: The model configuration.
        """
        return self.model_config
