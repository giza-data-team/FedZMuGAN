import torch
import torch.nn as nn


class ValidateData:
    def __init__(self, validateloader, device):
        self.validateloader = validateloader
        self.device = device

    def validate(
        self, net, target_class=None, compute_separate=False, compute_anamnesis=False
    ):
        """
        Validate the model on the test set.

        Args:
            net (torch.nn.Module): The model to validate.
            target_class (int, optional): The target class to compute specific metrics for.
            compute_separate (bool, optional): If True, compute separate metrics for target class and other classes.

        Returns:
            Tuple: (total_loss, metrics) where metrics contains overall, target, and other class accuracies.
        """
        net.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        class_correct = {}
        class_total = {}

        with torch.no_grad():
            for inputs, labels in self.validateloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                for label, prediction in zip(labels, predicted):
                    if label.item() not in class_correct:
                        class_correct[label.item()] = 0
                        class_total[label.item()] = 0
                    if label.item() == prediction.item():
                        class_correct[label.item()] += 1
                    class_total[label.item()] += 1

        accuracy = 100 * correct / total
        average_loss = total_loss / len(self.validateloader)
        class_accuracy = {
            cls: 100 * class_correct[cls] / class_total[cls] for cls in class_correct
        }
        target_class_accuracy = class_accuracy.pop(target_class, 0.0)
        other_classes_accuracy = (
            sum(class_accuracy.values()) / len(class_accuracy)
            if class_accuracy
            else 0.0
        )
        metrics = accuracy
        if compute_separate:
            metrics = {
                "overall_accuracy": accuracy,
                "target_accuracy": target_class_accuracy,
                "other_accuracy": other_classes_accuracy,
            }
        return average_loss, metrics
