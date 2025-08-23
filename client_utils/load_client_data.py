import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import logging
from config_manager import ConfigurationManager
from sklearn.model_selection import train_test_split
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LoadClientData:
    def __init__(
        self, client_id, device="cpu", dataset_name="MNIST", data_path="client_data"
    ):
        self.client_id = client_id
        self.dataset_name = dataset_name.lower()
        self.data_path = data_path
        config = ConfigurationManager()
        self.val_split = config.get_val_split()
        self.device = device

    def load_client_subsets(self):
        try:
            train_data_path = f"{self.data_path}/client_{self.client_id}_train.pt"
            test_data_path = f"{self.data_path}/client_{self.client_id}_test.pt"

            if not os.path.exists(train_data_path) or not os.path.exists(
                test_data_path
            ):
                logger.error(
                    f"Error: Missing train or test data for client {self.client_id}."
                )
                return None, None, None

            train_data_dict = torch.load(train_data_path)
            test_data_dict = torch.load(test_data_path)

            train_data, train_targets = (
                train_data_dict["data"],
                train_data_dict["targets"],
            )
            test_data, test_targets = test_data_dict["data"], test_data_dict["targets"]

            # Ensure data has a channel dimension
            if len(train_data.shape) == 3:
                train_data = train_data.unsqueeze(1)
            if len(test_data.shape) == 3:
                test_data = test_data.unsqueeze(1)

            full_train_dataset = TensorDataset(train_data, train_targets)
            full_test_dataset = TensorDataset(test_data, test_targets)

            total_train_size = len(full_train_dataset)
            val_size = int(self.val_split * total_train_size)

            # Convert tensors to numpy for stratification
            train_targets_np = train_targets.numpy()

            # Stratified split (train -> train/val)
            train_indices, val_indices = train_test_split(
                range(len(train_targets_np)),
                test_size=self.val_split,
                stratify=train_targets_np,
                random_state=42,
            )

            train_subset = Subset(full_train_dataset, train_indices)
            val_subset = Subset(full_train_dataset, val_indices)
            test_subset = Subset(full_test_dataset, list(range(len(full_test_dataset))))

            return train_subset, val_subset, test_subset

        except Exception as e:
            logger.error(f"Unexpected error loading client {self.client_id} data: {e}")
            return None, None, None

    def get_normal_loaders(self, batch_size):
        """
        Returns efficient DataLoaders for training, validation, and test.
        """
        train_subset, val_subset, test_subset = self.load_client_subsets()
        if train_subset is None:
            return None, None, None

        # DataLoader params
        loader_args = {
            "batch_size": batch_size,
            "num_workers": 4,  # Use multiple workers for speed (adjust to your CPU)
            "pin_memory": True,  # Faster transfer to CUDA
        }

        train_loader = DataLoader(train_subset, shuffle=True, **loader_args)
        val_loader = DataLoader(val_subset, shuffle=False, **loader_args)
        test_loader = DataLoader(test_subset, shuffle=False, **loader_args)

        logger.info(
            f"Efficient DataLoaders loaded for client {self.client_id} with batch size {batch_size}"
        )
        return train_loader, val_loader, test_loader

    def _split_subset_by_class(self, subset):
        """
        Splits a given Subset (either training, validation, or test) into class-specific Subsets.
        """
        original_dataset = subset.dataset  # This is our TensorDataset.
        indices = subset.indices

        class_to_indices = {}
        for idx in indices:
            _, label = original_dataset[idx]
            label = int(label.item())  # Ensure label is an integer.
            class_to_indices.setdefault(label, []).append(idx)

        # Create a Subset for each class.
        class_subsets = {
            label: Subset(original_dataset, idx_list)
            for label, idx_list in class_to_indices.items()
        }
        return class_subsets

    def get_class_specific_loaders(self, split, batch_size):
        """
        Returns a dictionary of DataLoaders, one per class, for the specified data split.
        The 'split' argument can be 'train', 'val', or 'test'.
        """
        train_subset, val_subset, test_subset = self.load_client_subsets()
        if train_subset is None:
            return None

        shuffle = True if split == "train" else False

        if split == "train":
            subset = train_subset
        elif split == "val":
            subset = val_subset
        elif split == "test":
            subset = test_subset
        else:
            raise ValueError("split must be either 'train', 'val', or 'test'")

        # Split the chosen subset by class.
        class_subsets = self._split_subset_by_class(subset)

        # Create DataLoaders for each class.
        class_loaders = {
            class_label: DataLoader(
                class_subset, batch_size=batch_size, shuffle=shuffle
            )
            for class_label, class_subset in class_subsets.items()
        }
        return class_loaders

    def get_forget_retain_loaders(self, forget_class, split, batch_size):
        """
        Returns DataLoaders that split the data into 'forget' and 'retain' subsets for the specified split.
        The 'split' argument can be 'train', 'val', or 'test'.
        """
        train_subset, val_subset, test_subset = self.load_client_subsets()
        if train_subset is None:
            return None, None

        shuffle = True if split == "train" else False

        # Choose the appropriate subset.
        if split == "train":
            subset = train_subset
        elif split == "val":
            subset = val_subset
        elif split == "test":
            subset = test_subset
        else:
            raise ValueError("split must be either 'train', 'val', or 'test'")

        # Build indices for forget (target_class) and retain (non-target_class)
        forget_indices = [
            i for i, datapoint in enumerate(subset) if datapoint[1] == forget_class
        ]
        retain_indices = [
            i for i, datapoint in enumerate(subset) if datapoint[1] != forget_class
        ]

        # Create Subsets for each.
        forget_subset = Subset(subset, forget_indices)
        retain_subset = Subset(subset, retain_indices)

        # Create DataLoaders.
        forget_loader = DataLoader(
            forget_subset, batch_size=batch_size, shuffle=shuffle
        )
        retain_loader = DataLoader(
            retain_subset, batch_size=batch_size, shuffle=shuffle
        )

        return forget_loader, retain_loader
