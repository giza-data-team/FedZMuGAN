import torch
import os
from config_manager import ConfigurationManager
from Machine_Unlearning.Unlearning.metrics.dataset import prepare_forget_retain_dataloaders
from data_loader import DatasetFactory
from Machine_Unlearning.Unlearning.metrics.forget_retain_accuracy import forget_retain_accuracy

def get_correct_and_total(model, dataloader, device):
    """
    Compute the total correct predictions and total samples for a given dataloader.
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct, total


def forget_retain_accuracy(model, forget_dataloader, retain_dataloader, device):
    """
    Compute forget and retain accuracies for unlearned model.
    """
    model.to(device)
    model.eval()

    correct_forget, total_forget = get_correct_and_total(
        model, forget_dataloader, device
    )
    correct_retain, total_retain = get_correct_and_total(
        model, retain_dataloader, device
    )

    forget_accuracy = 100 * correct_forget / total_forget if total_forget > 0 else 0
    retain_accuracy = 100 * correct_retain / total_retain if total_retain > 0 else 0

    return forget_accuracy, retain_accuracy


def __main__():
    from client_utils.models.model_factory import ModelFactory
    config_manager = ConfigurationManager()
    original_model_path=os.path.join(
                    config_manager.get_models_path_original(), 
                    f"{str(config_manager.get_model_name()).lower()}_{config_manager.get_dataset_name()}.pth"
                )
    dataset_loader = DatasetFactory.create_dataset(
        dataset_name=config_manager.get_dataset_name(),
    )
    dataset = dataset_loader
    forget_test_loader, retain_test_loader = prepare_forget_retain_dataloaders(
        dataset=dataset, 
        batch_size=64, 
        forget_class=3, 
        dataset_type="test"
    )
    device="cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the model
    state_dict = torch.load(original_model_path)
    model = ModelFactory.create_model(
    )  # Get the model instance
    model.load_state_dict(state_dict)  # Load the state dictionary
    model.to(device)
    
    forget_accuracy, retain_accuracy = forget_retain_accuracy(
        model=model,
        forget_dataloader=forget_test_loader,
        retain_dataloader=retain_test_loader,
        device=device
    )
    print(f"Forget accuracy: {forget_accuracy}, Retain accuracy: {retain_accuracy}")
    
if __name__ == "__main__":
    __main__()
