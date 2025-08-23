from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from config_manager import ConfigurationManager

config = ConfigurationManager()
img_size = config.get_image_size()


class DatasetLoader(ABC):
    def __init__(self, image_size = img_size, root="./data", transform_for="classifier"):
        self.root = root
        self.image_size = image_size
        self.transform_for = transform_for
        self.num_classes = None
        self.img_channels = None

    def gan_transform(self):
        return transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    @abstractmethod
    def get_train_dataset(self) -> Dataset:
        pass

    @abstractmethod
    def get_test_dataset(self) -> Dataset:
        pass

class MNISTLoader(DatasetLoader):
    def __init__(self, root="./data", image_size=img_size, transform_for="classifier"):
        super().__init__(image_size=image_size, root=root, transform_for=transform_for)
        self.transform = transforms.Compose(
            [   
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # Mean and std for MNIST
            ]
        )

    def get_train_dataset(self) -> Dataset:
        return torchvision.datasets.MNIST(
            root=self.root, train=True, download=True, transform=self.transform
        )

    def get_test_dataset(self) -> Dataset:
        return torchvision.datasets.MNIST(
            root=self.root, train=False, download=True, transform=self.transform
        )




class CIFAR10Loader(DatasetLoader):
    def get_train_dataset(self) -> Dataset:
        transform_train = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.Pad(padding=2),
            transforms.RandomCrop(size=(self.image_size,self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63. / 255., saturation=[0.5,1.5], contrast=[0.2,1.8]),
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091),(0.24703223, 0.24348513, 0.26158784))
        ])

        return torchvision.datasets.CIFAR10(
            root=self.root, 
            train=True, 
            download=True, 
            transform=transform_train if self.transform_for == "classifier" else self.gan_transform()
        )

    def get_test_dataset(self) -> Dataset:
        transform_test = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
        ])

        return torchvision.datasets.CIFAR10(
            root=self.root, 
            train=False, 
            download=True, 
            transform=transform_test if self.transform_for == "classifier" else self.gan_transform()
        )
    

class SVHNLoader(DatasetLoader):
    def get_train_dataset(self) -> Dataset:
        transform_train = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.Pad(padding=2),
            transforms.RandomCrop(size=(self.image_size, self.image_size)),
            transforms.ColorJitter(brightness=63. / 255., saturation=[0.5, 1.5], contrast=[0.2, 1.8]),
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])

        return torchvision.datasets.SVHN(
            root=self.root, 
            split="train", 
            download=True, 
            transform=transform_train if self.transform_for == "classifier" else self.gan_transform()
        )

    def get_test_dataset(self) -> Dataset:
        transform_test = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])

        return torchvision.datasets.SVHN(
            root=self.root, 
            split="test", 
            download=True, 
            transform=transform_test if self.transform_for == "classifier" else self.gan_transform()
        )




class DatasetFactory:
    @staticmethod
    def create_dataset(dataset_name: str, **kwargs) -> DatasetLoader:
        providers = {
            "cifar10": CIFAR10Loader,
            "svhn": SVHNLoader,
            "mnist": MNISTLoader,
        }
        if dataset_name.lower() not in providers:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return providers[dataset_name.lower()](**kwargs)