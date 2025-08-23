from torch.utils.data import DataLoader, random_split
import torch
import os
from data_loader import DatasetFactory
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(
        self,
        val_split,
        dataset_path,
        dataset_name,
        image_size,
        regenerate=False,
        seed=42,
    ):
        """
        Load and preprocess a dataset, split it into train/validate/test sets (saving splits to disk), and organize each split by class.
        """

        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.splits_dir = f"{self.dataset_path}/splits"
        self.regenerate = regenerate
        self.image_size = image_size
        self.num_classes = None
        self.img_channels = None

        os.makedirs(self.splits_dir, exist_ok=True)
        self.train_path = os.path.join(self.splits_dir, "train_split.pt")
        self.validate_path = os.path.join(self.splits_dir, "validate_split.pt")
        self.test_path = os.path.join(self.splits_dir, "test_split.pt")

        if (
            not self.regenerate
            and os.path.exists(self.train_path)
            and os.path.exists(self.validate_path)
            and os.path.exists(self.test_path)
        ):
            print("Loading dataset splits from disk...")
            self.train_dataset = torch.load(self.train_path)
            self.validate_dataset = torch.load(self.validate_path)
            self.test_dataset = torch.load(self.test_path)

            self.num_classes = torch.load(
                os.path.join(self.splits_dir, "num_classes.pt")
            )
            self.img_channels = torch.load(
                os.path.join(self.splits_dir, "img_channels.pt")
            )

        else:
            dataset_loader = DatasetFactory.create_dataset(
                dataset_name=self.dataset_name,
                root=self.dataset_path,
                image_size=self.image_size,
            )

            full_train_dataset = dataset_loader.get_train_dataset()
            self.test_dataset = dataset_loader.get_test_dataset()

            self.num_classes = dataset_loader.get_num_classes()
            self.img_channels = dataset_loader.get_img_channels()

            val_size = len(full_train_dataset) // val_split
            train_size = len(full_train_dataset) - val_size

            train_indices, val_indices = train_test_split(
                list(range(len(full_train_dataset))),
                test_size=val_size,
                stratify=[
                    full_train_dataset[i][1] for i in range(len(full_train_dataset))
                ],
                random_state=seed,
            )
            self.train_dataset = torch.utils.data.Subset(
                full_train_dataset, train_indices
            )
            self.validate_dataset = torch.utils.data.Subset(
                full_train_dataset, val_indices
            )
            print(f"Size of train_dataset = {len(self.train_dataset)}")
            print(f"Size of validate_dataset = {len(self.validate_dataset)}")
            print(f"Size of test_dataset = {len(self.test_dataset)}")

            torch.save(self.train_dataset, self.train_path)
            torch.save(self.validate_dataset, self.validate_path)
            torch.save(self.test_dataset, self.test_path)
            torch.save(
                self.num_classes, os.path.join(self.splits_dir, "num_classes.pt")
            )
            torch.save(
                self.img_channels, os.path.join(self.splits_dir, "img_channels.pt")
            )

            print(f"Saved train, validate, and test splits to {self.splits_dir}.")

        print(
            f"{self.dataset_name} dataset initialized with {self.num_classes} classes and {self.img_channels} image channels!"
        )

        self.train_dataset_split_by_class = self._split_dataset_by_class(
            self.train_dataset
        )
        print("Splitted the train dataset by class successfully!")

        self.validate_dataset_split_by_class = self._split_dataset_by_class(
            self.validate_dataset
        )
        print("Splitted the validate dataset by class successfully!")

        self.test_dataset_split_by_class = self._split_dataset_by_class(
            self.test_dataset
        )
        print("Splitted the test dataset by class successfully!")

    def get_train_dataset(self, loader=False, batch_size=None):
        """Return the training dataset, optionally as a DataLoader if loader=True and batch_size is specified."""
        if loader and batch_size:
            return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        return self.train_dataset

    def get_validate_dataset(self, loader=False, batch_size=None):
        """Return the validation dataset, optionally as a DataLoader if loader=True and batch_size is specified."""
        if loader and batch_size:
            return DataLoader(
                self.validate_dataset, batch_size=batch_size, shuffle=False
            )
        return self.validate_dataset

    def get_test_dataset(self, loader=False, batch_size=None):
        """Return the testing dataset, optionally as a DataLoader if loader=True and batch_size is specified."""
        if loader and batch_size:
            return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        return self.test_dataset

    @staticmethod
    def _split_dataset_by_class(dataset):
        """Organizes dataset samples into a dictionary grouped by class labels."""
        dataset_split_by_class = {}
        for datapoint in dataset:
            if datapoint[1] in dataset_split_by_class:
                dataset_split_by_class[datapoint[1]].append(
                    [datapoint[0], datapoint[1]]
                )
            else:
                dataset_split_by_class[datapoint[1]] = [[datapoint[0], datapoint[1]]]
        return dataset_split_by_class

    def get_iterations_per_epoch(self, batch_size):
        """Returns the number of iterations (batches) per epoch."""
        dataset_size = len(self.train_dataset)  # Total training samples
        iterations = (dataset_size + batch_size - 1) // batch_size  # Ceiling division
        return iterations
