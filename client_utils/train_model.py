import torch
import torch.optim as optim
import torch.nn as nn
from config_manager import ConfigurationManager
from client_utils.general_utils import set_seed


class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, device):
        """
        Initializes the model trainer with the given parameters.
        """
        # Set seeds for reproducibility
        set_seed()

        self.config = ConfigurationManager()
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = self.config.get_epochs_original()
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

        # Create optimizer with deterministic settings
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.get_lr_original(),
            momentum=self.config.get_optim_momentum(),
            weight_decay=self.config.get_optim_weight_decay_original(),
        )

        # Create scheduler with deterministic settings
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.get_scheduler_t_max()
        )
        self.patience = self.config.get_patience()

    def train(self, model):
        """
        Train the model and validate at the end of each epoch.
        """
        total_train_loss, total_val_loss = 0, 0
        total_train_accuracy, total_val_accuracy = 0, 0
        best_val_loss = 1e6  # Initialize best validation loss with a large number
        best_val_acc, best_train_loss, best_train_acc = 0, 0, 0
        patience_counter = 0  # Counter for early stopping
        self.model = model
        self.best_model_state = None
        for epoch in range(self.num_epochs):
            train_loss, train_accuracy = self._train_one_epoch()
            val_loss, val_accuracy = self._validate_one_epoch()

            self.scheduler.step()

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()
                best_train_loss = train_loss
                best_train_acc = train_accuracy
                best_val_acc = val_accuracy
                patience_counter = 0  # Reset counter when improvement occurs
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print("Early stopping triggered. Stopping training.")
                break  # Stop training if patience is exceeded
            print(
                f"Epoch {epoch + 1}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
            )
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return (
            best_train_loss,
            best_val_loss,
            best_train_acc,
            best_val_acc,
        )

    def _train_one_epoch(self):
        """
        Performs one training epoch and returns the loss and accuracy.
        """
        self.model.train()
        total_loss, correct, total_samples = 0, 0, 0

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total_samples
        return avg_loss, accuracy

    def _validate_one_epoch(self):
        """
        Performs validation and returns the loss and accuracy.
        """
        self.model.eval()
        total_loss, correct, total_samples = 0, 0, 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total_samples
        return avg_loss, accuracy
