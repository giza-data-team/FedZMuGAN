import os
import csv
import torch
import random
import numpy as np
from config_manager import ConfigurationManager
from client_utils.models.model_factory import ModelFactory


def set_seed():
    config_manager = ConfigurationManager()
    seed = config_manager.get_seed()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_model(path, device):
    if os.path.isdir(path):
        raise ValueError(f"Expected a file path, got directory: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    config = ConfigurationManager()
    model_factory = ModelFactory()
    model = model_factory.create_model().to(device)

    model.load_state_dict(torch.load(path, map_location=device))

    return model


# Define CSV column headers
csv_headers = [
    "Model",
    "Forget Accuracy",
    "Retain Accuracy",
    "Membership Inference Attack",
    "Anamnesis Index",
]


def log_results_to_csv(path, model_name, forget_acc, retain_acc, mia, ain=None):
    """Append unlearning evaluation metrics to a CSV file."""
    file_exists = os.path.isfile(path)

    with open(path, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)

        # Write headers if the file is new
        if not file_exists:
            writer.writerow(csv_headers)

        # Log model results
        writer.writerow(
            [
                model_name,
                f"{forget_acc:.2f}",
                f"{retain_acc:.2f}",
                f"{mia:.2f}",
                f"{ain:.4f}" if ain is not None else "N/A",
            ]
        )
