import torch
import torch.optim as optim
import torch.nn as nn
from client_utils.general_utils import set_seed

set_seed()

def relearn_time(
    model,
    train_loader,
    forget_val_loader,
    learning_rate,
    target_accuracy_min,
    device,
    max_epochs,
):
    """
    Return the number of training steps needed for the model to reach an accuracy within
    alpha% of its original accuracy on the forget test data.
    
    Uses the same optimizer, scheduler, and loss function as defined in ModelTrainer.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    print("Starting relearn process...\n")

    steps = 0

    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}...")
        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate after each batch
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for val_inputs, val_labels in forget_val_loader:
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    val_outputs = model(val_inputs)
                    _, predicted = torch.max(val_outputs, 1)
                    correct += (predicted == val_labels).sum().item()
                    total += val_labels.size(0)
            
            accuracy = (correct / total) * 100 
            
            if steps>0 and steps%10 == 0:
                print(f"[Step {steps}] Training Loss: {loss.item():.4f} Forget Val Accuracy: {accuracy:.2f}%")

            if accuracy >= target_accuracy_min:
                print(f"[Step {steps}] Relearn threshold reached with accuracy: {accuracy:.2f}%")
                return steps

    print(f"Reached max epochs without achieving target accuracy.")
    return steps


def anamnesis_index(
    model_u,
    model_s,
    train_loader,
    forget_val_loader,
    learning_rate,
    original_accuracy,
    alpha,
    device,
    max_steps,
):
    """
    Compute the Anamnesis Index (AIN) as the ratio of relearn times between an unlearned model and a scratch-trained model on the forget dataset.
    """

    target_accuracy_min = original_accuracy * (1 - alpha / 100)

    print(f"Original Model Accuracy: {original_accuracy:.2f}%")
    print(f"Minimum Target Accuracy: {target_accuracy_min:.2f}%")


    print("\nCalculating relearn time for the unlearned model...")
    rt_u = relearn_time(
        model_u,
        train_loader,
        forget_val_loader,
        learning_rate,
        target_accuracy_min,
        device,
        max_steps,
    )
    print(f"Relearn time for unlearned model: {rt_u} steps\n")

    print("Calculating relearn time for the scratch-trained model...")
    rt_s = relearn_time(
        model_s,
        train_loader,
        forget_val_loader,
        learning_rate,
        target_accuracy_min,
        device,
        max_steps,
    )
    print(f"Relearn time for scratch-trained model: {rt_s} steps\n")

    ain = rt_u / rt_s
    return ain