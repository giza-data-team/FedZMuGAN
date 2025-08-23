import torch
from client_utils.general_utils import set_seed

set_seed()
from Machine_Unlearning.Unlearning.train.utils import (
    accumulate_batches,
    predict_classes,
)
from Machine_Unlearning.zMuGAN.utils import sample_noises, sample_labels


class zMuGANDataGenerator:
    """
    Generate synthetic data for zMuGAN by sampling noises and one-hot labels, and then split the data based on model predictions.
    """

    def __init__(
        self,
        generator,
        forget_class,
        num_classes,
        class_samples_size,
        batch_size,
        noise_dim,
        device,
        original_model,
    ):
        """
        Initialize with a GAN generator, target forget class, number of classes, samples per class, batch size, noise dimension, device, and a filtering model.
        """

        self.generator = generator
        self.forget_class = forget_class
        self.class_samples_size = class_samples_size
        self.device = device
        self.original_model = original_model
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.dataset_size = self.class_samples_size * num_classes
        self.original_model.eval()

    def generate_batch(self, bs):
        """
        Generate one batch of synthetic data using sampled noises and one-hot labels.
        """

        noises = sample_noises((bs, self.noise_dim)).to(self.device)
        labels = sample_labels(bs, self.num_classes, dist="onehot").to(self.device)
        with torch.no_grad():
            batch_data = self.generator(labels, noises)
        return batch_data

    def generate_synthetic_data(self):
        """
        Accumulate synthetic data batches until reaching the total dataset_size.
        """

        return accumulate_batches(
            lambda bs: self.generate_batch(bs), self.dataset_size, self.batch_size
        )

    def get_datasets(self):
        """
        Generate synthetic data, predict classes with the original model, and split the data into forget and retain datasets.
        """

        synthetic_data = self.generate_synthetic_data()
        predicted_classes = predict_classes(
            self.original_model, synthetic_data, self.device
        )

        forget_mask = predicted_classes == self.forget_class
        retain_mask = ~forget_mask

        Df_data = synthetic_data[forget_mask]
        Df_labels = predicted_classes[forget_mask]

        Dr_data = synthetic_data[retain_mask]
        Dr_labels = predicted_classes[retain_mask]

        print(f"Generated Forget Dataset: {Df_data.shape}, {Df_labels.shape}")
        print(f"Generated Retain Dataset: {Dr_data.shape}, {Dr_labels.shape}")

        return (Df_data, Df_labels), (Dr_data, Dr_labels)

    def sample_kegnet_data(self, num_data, generators):
        """
        Sample artificial data using generator networks.
        """
        from torch.utils.data import DataLoader, TensorDataset

        ny = generators[0].num_classes
        nz = generators[0].num_noises
        noises = sample_noises(size=(num_data, nz))
        labels_in = sample_labels(num_data, ny, dist="onehot")
        loader = DataLoader(
            TensorDataset(noises, labels_in), batch_size=self.batch_size
        )

        images_list = []
        for idx, generator in enumerate(generators):
            l1 = []
            for z, y in loader:
                z = z.to(self.device)
                y = y.to(self.device)
                l1.append(generator(y, z).detach())
            images_list.append(torch.cat(tuple(l1), dim=0))
        return torch.cat(tuple(images_list), dim=0)
