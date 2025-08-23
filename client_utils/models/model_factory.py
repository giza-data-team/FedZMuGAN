from config_manager import ConfigurationManager
from .vgg16 import VGG16,VGG13
from .resnet import ResNet101, ResNet152, ResNet18, ResNet50
from .mobilenet import MobileNet
from .net import Net
from .all_cnn import AllCNN

class ModelFactory:
    @staticmethod
    def create_model():
        config = ConfigurationManager()
        models = {
            "vgg16": VGG16,
            "vgg13": VGG13,
            "resnet18": ResNet18,
            "resnet50": ResNet50,
            "resnet101": ResNet101,
            "resnet152": ResNet152,
            "mobilenet": MobileNet,
            "net": Net,
            "allcnn": AllCNN,
        }

        model_name = config.get_model_name().lower()

        if model_name not in models:
            raise ValueError(
                f"Model {model_name} not supported. Available models: {list(models.keys())}"
            )

        return models[model_name]()
