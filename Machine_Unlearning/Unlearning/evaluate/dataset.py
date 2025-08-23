from torch.utils.data import DataLoader, Subset


def prepare_forget_retain_dataloaders(dataset, batch_size, forget_class, dataset_type):
    """
    Return DataLoaders splitting the specified dataset into forget (target_class) and retain (non-target_class) subsets.
    """

    if dataset_type == "train":
        full_dataset = dataset.get_train_dataset(loader=False)
    elif dataset_type == "validate":
        full_dataset = dataset.get_validate_dataset(loader=False)
    elif dataset_type == "test":
        full_dataset = dataset.get_test_dataset()
    else:
        raise ValueError("dataset_type must be 'train', 'validate', or 'test'.")
    
    shuffle = True if dataset_type == "train" else False

    forget_indices = [
        i for i, datapoint in enumerate(full_dataset) if datapoint[1] == forget_class
    ]
    retain_indices = [
        i for i, datapoint in enumerate(full_dataset) if datapoint[1] != forget_class
    ]

    forget_subset = Subset(full_dataset, forget_indices)
    retain_subset = Subset(full_dataset, retain_indices)

    forget_loader = DataLoader(forget_subset, batch_size=batch_size, shuffle=shuffle)
    retain_loader = DataLoader(retain_subset, batch_size=batch_size, shuffle=shuffle)

    return forget_loader, retain_loader