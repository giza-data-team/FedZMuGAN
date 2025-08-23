import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from config_manager import ConfigurationManager
from data_loader import DatasetFactory
import numpy as np
import os

import logging

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataSplitter:
    def __init__(
        self,
        num_clients=5,
        homogeneous=True,
        data_path="client_data",
        min_instances_per_client=1000,
        dataset_name="MNIST",
    ):
        self.num_clients = num_clients
        self.homogeneous = homogeneous
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.min_instances_per_client = min_instances_per_client
        self.dataset_loader = DatasetFactory.create_dataset(self.dataset_name)
        self.config = ConfigurationManager()
        self.random_state = self.config.get_seed()

    def _split_data_homogeneous(self, data, targets, length_check=True):
        """
        Split data homogeneously with guaranteed equal splits
        """

        skf = StratifiedKFold(
            n_splits=self.num_clients, shuffle=True, random_state=self.random_state
        )

        client_datasets = []
        for _, client_indices in skf.split(data, targets):
            client_data = data[client_indices]
            client_targets = targets[client_indices]

            # Verify minimum instances requirement
            if len(client_data) < self.min_instances_per_client and length_check:
                raise Exception(
                    f"Client has fewer than {self.min_instances_per_client} instances."
                )

            client_datasets.append((client_data, client_targets))

        return client_datasets

    def _split_data_heterogeneous(self, data, targets, length_check=True):
        """
        Split the dataset heterogeneously, ensuring that all clients receive instances
        from all target classes, the total data size matches the original dataset size,
        and each client has at least min_instances_per_client data points.
        """
        num_classes = len(torch.unique(targets))
        idx_per_class = [
            torch.where(targets == i)[0] for i in range(num_classes)
        ]  # Indices for each class

        # Initialize storage for client data
        client_splits = {client: [] for client in range(self.num_clients)}
        client_classes = {
            client: set() for client in range(self.num_clients)
        }  # Track classes per client

        for c in range(num_classes):
            # Shuffle indices for the current class
            indices = idx_per_class[c]
            np.random.shuffle(indices.numpy())

            # Distribute class data heterogeneously across clients
            proportions = np.random.dirichlet(
                np.ones(self.num_clients) * 0.5
            )  # Dirichlet distribution
            proportions = (proportions * len(indices)).astype(int)
            proportions[-1] += len(indices) - sum(
                proportions
            )  # Ensure all indices are assigned

            start_idx = 0
            for client_id, size in enumerate(proportions):
                if size > 0:
                    class_indices = indices[start_idx : start_idx + size]
                    client_splits[client_id].append(class_indices)
                    client_classes[client_id].add(c)
                    start_idx += size

        # Ensure all clients have instances from all classes
        for c in range(num_classes):
            for client_id in range(self.num_clients):
                if c not in client_classes[client_id]:
                    # Add a small number of samples from the missing class to this client
                    missing_indices = idx_per_class[c][:1]  # Take at least one sample
                    client_splits[client_id].append(missing_indices)
                    client_classes[client_id].add(c)

        # Finalize client data
        client_data = []
        for client_id in range(self.num_clients):
            # Concatenate indices assigned to this client
            client_indices = (
                torch.cat(client_splits[client_id])
                if client_splits[client_id]
                else torch.tensor([], dtype=torch.long)
            )

            # Ensure minimum number of instances per client
            if len(client_indices) < self.min_instances_per_client and length_check:
                additional_needed = self.min_instances_per_client - len(client_indices)
                extra_indices = torch.randperm(len(data))[
                    :additional_needed
                ]  # Random extra indices
                client_indices = torch.cat((client_indices, extra_indices))

            client_data.append((data[client_indices], targets[client_indices]))

        return client_data

    def _load_combined_mnist_data(self):
        # Load train and test data from MNIST
        train_dataset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.1307,), (0.3081,)
                    ),  # Mean and std for MNIST
                ]
            ),
        )
        test_dataset = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.1307,), (0.3081,)
                    ),  # Mean and std for MNIST
                ]
            ),
        )

        # Combine train and test data
        train_data = (
            torch.tensor(train_dataset.data).unsqueeze(1).float() / 255
        )  # Normalize
        train_targets = torch.tensor(train_dataset.targets)

        test_data = (
            torch.tensor(test_dataset.data).unsqueeze(1).float() / 255
        )  # Normalize
        test_targets = torch.tensor(test_dataset.targets)

        # Concatenate train and test data
        data = torch.cat((train_data, test_data), dim=0)
        targets = torch.cat((train_targets, test_targets), dim=0)

        return data, targets

    def _load_data(self):
        # Get train and test datasets from DatasetLoader
        train_dataset = self.dataset_loader.get_train_dataset()
        test_dataset = self.dataset_loader.get_test_dataset()
        # Convert train and test datasets to tensors
        train_data = torch.stack(
            [train_dataset[i][0] for i in range(len(train_dataset))]
        )
        train_targets = torch.tensor(
            [train_dataset[i][1] for i in range(len(train_dataset))]
        )

        test_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
        test_targets = torch.tensor(
            [test_dataset[i][1] for i in range(len(test_dataset))]
        )

        # Concatenate train and test data
        # data = torch.cat((train_data, test_data), dim=0)
        # targets = torch.cat((train_targets, test_targets), dim=0)

        # return data, targets
        return train_data, train_targets, test_data, test_targets

    def generate_and_split_data(self):
        # Load train and test datasets separately
        train_data, train_targets, test_data, test_targets = self._load_data()
        print("data loaded")
        print("start split train data")
        # Split the train data among clients
        if self.homogeneous:
            train_splits = self._split_data_homogeneous(train_data, train_targets)
            print("start split test data")
            test_splits = self._split_data_homogeneous(
                test_data, test_targets, length_check=False
            )
        else:
            train_splits = self._split_data_heterogeneous(train_data, train_targets)
            test_splits = self._split_data_heterogeneous(
                test_data, test_targets, length_check=False
            )

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        # Track target distribution for validation
        all_train_targets = {i: set() for i in range(self.num_clients)}
        all_test_targets = {i: set() for i in range(self.num_clients)}

        # Save the train and test data separately for each client
        for i, (client_train_data, client_train_targets) in enumerate(train_splits):
            logger.info(f"client_{i}_train_data_length_is_{len(client_train_data)}")
            torch.save(
                {"data": client_train_data, "targets": client_train_targets},
                f"{self.data_path}/client_{i}_train.pt",
            )
            all_train_targets[i].update(client_train_targets.unique().tolist())

        for i, (client_test_data, client_test_targets) in enumerate(test_splits):
            logger.info(f"client_{i}_test_data_length_is_{len(client_test_data)}")
            torch.save(
                {"data": client_test_data, "targets": client_test_targets},
                f"{self.data_path}/client_{i}_test.pt",
            )
            all_test_targets[i].update(client_test_targets.unique().tolist())

        # Validate class distribution across clients
        all_train_classes = set.union(*all_train_targets.values())
        all_test_classes = set.union(*all_test_targets.values())

        missing_train_classes = (
            set(range(len(torch.unique(train_targets)))) - all_train_classes
        )
        missing_test_classes = (
            set(range(len(torch.unique(test_targets)))) - all_test_classes
        )

        if missing_train_classes:
            raise ValueError(
                f"Missing train classes in the split: {sorted(missing_train_classes)}"
            )
        else:
            logger.info("All train classes are properly distributed across clients.")

        if missing_test_classes:
            raise ValueError(
                f"Missing test classes in the split: {sorted(missing_test_classes)}"
            )
        else:
            logger.info("All test classes are properly distributed across clients.")

        return True
