"""
Knowledge Extraction with No Observable Data (NeurIPS 2019)

Authors:
- Jaemin Yoo (jaeminyoo@snu.ac.kr), Seoul National University
- Minyong Cho (chominyong@gmail.com), Seoul National University
- Taebum Kim (k.taebum@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
import numpy as np
import torch
from Machine_Unlearning.zMuGAN.utils import (
    sample_noises,
    sample_labels,
    visualize_images,
)
from Machine_Unlearning.zMuGAN.loss import ReconstructionLoss, DiversityLoss
from client_utils.early_stopping import EarlyStopping
import torch.optim as optim
import os
import csv
from client_utils.general_utils import set_seed

set_seed()


def train(
    networks,
    num_classes,
    noise_dim,
    batch_size,
    num_batches,
    lr,
    alpha,
    beta,
    epochs,
    patience,
    min_delta,
    device,
    output_dir,
    model_name,
    dataset_name,
    save_path,
    generator_index=None,
):
    """
    Update generator and decoder networks for multiple epochs.
    """
    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)

    generator, classifier, decoder = networks
    print("Loaded networks: generator, classifier, decoder")

    cls_loss = ReconstructionLoss(method="kld").to(device)
    dec_loss = ReconstructionLoss(method="l2").to(device)
    div_loss = DiversityLoss(metric="l1").to(device)

    params = list(generator.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr)

    generator.train()

    logs_path = os.path.join(output_dir, "results")
    os.makedirs(logs_path, exist_ok=True)
    print(f"Created logs directory at {logs_path}")

    # Include generator index in CSV file name to avoid overwriting
    index_suffix = f"_{generator_index}" if generator_index is not None else ""
    csv_file = os.path.join(
        logs_path, f"{dataset_name}_{model_name}_training_log{index_suffix}.csv"
    )
    print(f"CSV file for logging: {csv_file}")

    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Epoch", "Cls Loss", "Dec Loss", "Div Loss", "Total Loss", "Accuracy"]
        )

        for epoch in range(epochs):
            list_bs, list_loss, list_corr = [], [], []
            print(f"Starting epoch {epoch+1}/{epochs}")

            for batch_idx in range(num_batches):
                noise_size = batch_size, noise_dim
                noises = sample_noises(noise_size).to(device)
                labels = sample_labels(batch_size, num_classes, dist="onehot").to(
                    device
                )
                print(f"Batch {batch_idx+1}/{num_batches}: Sampled noises and labels")

                optimizer.zero_grad()

                images = generator(labels, noises)
                outputs = classifier(images)

                loss1 = cls_loss(outputs, labels)
                loss2 = dec_loss(decoder(images), noises) * alpha
                loss3 = div_loss(noises, images) * beta
                loss = loss1 + loss2 + loss3
                print(
                    f"Calculated losses: Cls={loss1.item()}, Dec={loss2.item()}, Div={loss3.item()}, Total={loss.item()}"
                )

                loss.backward()
                optimizer.step()

                corrects = torch.sum(outputs.argmax(dim=1).eq(labels.argmax(dim=1)))
                print(f"Batch {batch_idx+1}: Correct predictions={corrects.item()}")

                list_bs.append(batch_size)
                list_loss.append(
                    (loss1.item(), loss2.item(), loss3.item(), loss.item())
                )
                list_corr.append(corrects.item())

            loss = np.average(list_loss, axis=0, weights=list_bs)
            accuracy = (np.sum(list_corr) / np.sum(list_bs)) * 100
            print(f"Epoch {epoch+1} completed: Loss={loss}, Accuracy={accuracy:.2f}%")

            writer.writerow(
                [
                    epoch + 1,
                    f"{loss[0]:.4f}",
                    f"{loss[1]:.4f}",
                    f"{loss[2]:.4f}",
                    f"{loss[3]:.4f}",
                    f"{accuracy:.2f}",
                ]
            )
            file.flush()
            print("Logged epoch results to CSV")

            # Save sample images every 5 epochs
            if (epoch + 1) % 5 == 0:
                sample_image_dir = os.path.join(
                    output_dir, "samples", dataset_name, model_name + index_suffix
                )
                os.makedirs(sample_image_dir, exist_ok=True)

                sample_image_path = os.path.join(
                    sample_image_dir, f"sample_epoch_{epoch+1}.png"
                )
                visualize_images(generator, sample_image_path, device)
                print(f"Saved sample images for epoch {epoch+1} at {sample_image_path}")

            stop = early_stopper(loss[0])
            print(f"Checked early stopping: {stop}")

            if early_stopper.save_weights:
                print(f"Saved trained generator to {save_path}.")
                torch.save(generator.state_dict(), save_path)
                early_stopper.save_weights = False

            if stop:
                print(
                    f"Early stopping triggered at epoch {epoch + 1}. Training stopped."
                )
                break
