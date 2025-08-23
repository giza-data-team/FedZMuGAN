import os
import csv


class FileController:
    def __init__(self, file_name, path="client_results", client=True):
        self.file_path = os.path.join(path, file_name)
        os.makedirs(path, exist_ok=True)
        if client:
            self.columns = [
                "dataset_name",
                "round_number",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "test_loss",
                "test_acc",
                "training_time",
                "test_time",
                "anamnesis_index",
                "training_status",
                "mia",
                "forget_accuracy",
                "retain_accuracy",
            ]
        else:
            self.columns = [
                "dataset_name",
                "round_number",
                "train_agg_loss",
                "train_agg_acc",
                "val_agg_loss",
                "val_agg_acc",
                "test_agg_loss",
                "test_agg_acc",
                "weights_agg_time",
                "loss_agg_time",
                "avg_anamnesis_index",
                "homogeneous",
                "unlearning_time",
                "impair_loss",
                "repair_loss",
                "training_status",
                "avg_mia",
                "avg_forget_acc",
                "avg_retain_acc",
            ]
        # Check if the file exists, if not create it with header
        if not os.path.isfile(self.file_path):
            with open(self.file_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(self.columns)

    def save(self, info: dict = {}):
        with open(self.file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([info.get(k, "") for k in self.columns])
