from client_utils.weights_controller import WeightsController
from client_utils.test_model import ValidateData
from Machine_Unlearning.Unlearning.metrics.membership_inference_attack import get_membership_attack_prob
from Machine_Unlearning.Unlearning.metrics.anamnesis_index import anamnesis_index
from Machine_Unlearning.Unlearning.metrics.forget_retain_accuracy import forget_retain_accuracy
from copy import deepcopy
from config_manager import ConfigurationManager
from client_utils.general_utils import set_seed

set_seed()


def get_metrics(
    original_model_path,
    unlearned_model_path,
    scratch_model_path,
    forget_test_loader,
    retain_test_loader,
    device,
    forget_train_loader,
    retain_train_loader,
    test_loader,
    alpha,
):

    dev = device
    weight_controller = WeightsController()
    original_model = weight_controller.load_model(original_model_path, device)
    unlearned_model = weight_controller.load_model(unlearned_model_path, device)
    scratch_model = weight_controller.load_model(scratch_model_path, device)
    config = ConfigurationManager()

    validate_data = ValidateData(
                validateloader= test_loader,
                device=dev
                
            )
    print(f'device: {dev}')
    
    
    
    # Calculate original model's retain and forget accuracies
    original_model.eval()  # Ensure model is in eval mode
    print(f"Original model device: {next(original_model.parameters()).device}")
    print(f"Test loader device: {next(iter(forget_test_loader))[0].device}")
    
    loss, original_accuracy = validate_data.validate(net=original_model)
    print(f"Original model overall accuracy: {original_accuracy:.4f}")
    original_forget_accuracy, original_retain_accuracy = forget_retain_accuracy(
        original_model, forget_test_loader, retain_test_loader, device
    )
    print(f"Original model forget accuracy: {original_forget_accuracy:.4f}")
    print(f"Original model retain accuracy: {original_retain_accuracy:.4f}")

    mia = get_membership_attack_prob(
        retain_train_loader, forget_train_loader, test_loader, unlearned_model, device
    )
    print(f"Membership inference attack: {mia:.4f}")

    # Calculate Anamnesis Index

    unlearned_model_ain = deepcopy(unlearned_model).to(device)

    ain = anamnesis_index(
        model_u=unlearned_model_ain,
        model_s=scratch_model,
        train_loader=forget_train_loader,
        forget_val_loader =forget_test_loader,
        learning_rate=config.get_lr_unlearn(),
        original_accuracy=original_accuracy,
        alpha=alpha,
        device=device,
        max_steps=100,
    )
    print(f"Anamnesis Index (AIN): {ain:.4f}")


    forget_accuracy, retain_accuracy = forget_retain_accuracy(
        unlearned_model, forget_test_loader, retain_test_loader, device
    )

    metrics = {
        "mia": mia,
        "anamnesis_index": ain,
        "forget_acc": forget_accuracy,
        "retain_acc": retain_accuracy,
        "original_forget_acc": original_forget_accuracy,
        "original_retain_acc": original_retain_accuracy,
        "original_accuracy": original_accuracy,
    }

    return metrics