import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import logging
import os
from config_manager import ConfigurationManager
from client_utils.weights_controller import WeightsController

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DeviceDataLoader:
    """
    A wrapper around DataLoader to automatically move data to the specified device.
    """
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
    
    def __iter__(self):
        for batch in self.dataloader:
            # Handle different batch structures
            if isinstance(batch, (list, tuple)):
                yield [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch]
            else:
                yield batch.to(self.device)
    
    def __len__(self):
        return len(self.dataloader)


class UNSIR_noise(torch.nn.Module):
    """
    UNSIR noise module for generating adversarial noise.
    """
    def __init__(self, *dim):
        super().__init__()
        self.noise = torch.nn.Parameter(torch.randn(*dim), requires_grad=True)

    def forward(self):
        return self.noise


class EMMNUnlearning:
    """
    Enhanced Machine unlearning via Maximizing Neuron Mixing (EMMN) implementation.
    """
    
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config_manager = ConfigurationManager()
        logger.info(f"Initialized EMMN with device: {self.device}")
    
    def UNSIR_noise_train(self, noise, model, forget_class_label, num_epochs, noise_batch_size):
        """
        Train UNSIR noise for the forget class.
        """
        opt = torch.optim.Adam(noise.parameters(), lr=0.1)
        logger.info(f"Training UNSIR noise for forget class {forget_class_label}")

        for epoch in range(num_epochs):
            total_loss = []
            inputs = noise()
            labels = torch.zeros(noise_batch_size).to(self.device) + forget_class_label
            
            # Handle different model output formats
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = -F.cross_entropy(outputs, labels.long()) + 0.1 * torch.mean(
                torch.sum(inputs**2, [1, 2, 3])
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss.append(loss.cpu().detach().numpy())
            
            if epoch % 5 == 0:
                logger.info(f"UNSIR Epoch {epoch}, Loss: {np.mean(total_loss):.4f}")

        return noise

    def emmn_noise_train(self, noise, model, target_class_label, num_epochs, noise_batch_size):
        """
        Train EMMN noise for retain classes.
        """
        opt = torch.optim.Adam(noise.parameters(), lr=0.1)
        logger.info(f"Training EMMN noise for retain class {target_class_label}")

        for epoch in range(num_epochs):
            total_loss = []
            inputs = noise()
            labels = torch.zeros(noise_batch_size).to(self.device) + target_class_label
            
            # Handle different model output formats
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = F.cross_entropy(outputs, labels.long())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss.append(loss.cpu().detach().numpy())
            
            if epoch % 5 == 0:
                logger.info(f"EMMN Epoch {epoch}, Loss: {np.mean(total_loss):.4f}")

        return noise

    def emmc_create_noisy_loader(self, noise_dict, batch_size=64, num_noise_batches=80):
        """
        Create a DataLoader with noisy data for all classes.
        """
        noisy_data = []
        logger.info(f"Creating noisy loader with {num_noise_batches} batches per class")
        
        for i in range(num_noise_batches):
            for class_label, noise_module in noise_dict.items():
                batch = noise_module()
                for k in range(batch.size(0)):
                    noisy_data.append(
                        (
                            batch[k].detach().cpu(),
                            torch.tensor(class_label),
                            torch.tensor(class_label)
                        )
                    )
        
                noisy_loader = DataLoader(noisy_data, batch_size=batch_size, shuffle=True)
        logger.info(f"Created noisy loader with {len(noisy_data)} samples")
        return noisy_loader

    def fit(self, epochs, lr, model, train_loader, opt_func=torch.optim.Adam):
        """
        Train model with noisy data (no validation).
        """
        optimizer = opt_func(model.parameters(), lr)
        logger.info(f"Starting EMMN training for {epochs} epochs with lr={lr}")
        
        for epoch in range(epochs):
            model.train()
            train_losses = []
            
            for batch in train_loader:
                if hasattr(model, 'training_step'):
                    # If model has training_step method
                    loss = model.training_step(batch)
                else:
                    # Standard training step
                    inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = F.cross_entropy(outputs, labels)
                
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            # Just log training loss
            avg_train_loss = torch.stack(train_losses).mean().item()
            logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}")
        
        logger.info("EMMN training completed")

    def emmn_fit(self):
        """
        EMMN unlearning fit phase - performs noise training and model fine-tuning.
        Loads model from disk and gets all parameters from config manager.
        
        Returns:
            model: The unlearned model
        """
        # Get all parameters from config manager
        forget_class = self.config_manager.get_forget_class()
        dataset_name = self.config_manager.get_dataset_name()
        model_name = self.config_manager.get_model_name()
        learning_rate = self.config_manager.get_lr_unlearn()
        batch_size = self.config_manager.get_batch_size_unlearn()
        num_classes = self.config_manager.get_num_classes()
        img_shape = self.config_manager.get_image_size()
        num_channels = self.config_manager.get_img_channels()
        
        # Load the saved model from disk
        # Get the model path (assuming it's the aggregated model)
        model_path = os.path.join(
            self.config_manager.get_models_path_original(),
            f"{model_name.lower()}_{dataset_name}.pth"
        )
        
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}")
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load the model
        weight_controller = WeightsController()
        model = weight_controller.load_model(model_path, self.device)
        
        logger.info(f"Starting EMMN fit phase for forget class {forget_class}")
        logger.info(f"Using image shape: {img_shape}x{img_shape}, Channels: {num_channels}")
        logger.info(f"Loaded model from: {model_path}")
        
        noise_batch_size = batch_size
        
        # Create and train UNSIR noise for forget class
        forget_class_label = forget_class
        noise = UNSIR_noise(noise_batch_size, num_channels, img_shape, img_shape).to(self.device)
        noise = self.UNSIR_noise_train(
            noise, model, forget_class_label, 400, noise_batch_size
        )

        # Create and train EMMN noises for retain classes
        retain_noises = {
            i_: UNSIR_noise(noise_batch_size, num_channels, img_shape, img_shape).to(self.device)
            for i_ in [j for j in range(num_classes) if j != forget_class]
        }
        
        for i_, n_ in retain_noises.items():
            retain_noises[i_] = self.emmn_noise_train(n_, model, i_, 400, noise_batch_size)

        # Combine all noises
        noises = retain_noises
        noises[forget_class] = noise

        # Create noisy data loader
        noisy_loader = self.emmc_create_noisy_loader(
            noises,
            batch_size=noise_batch_size,
        )
        
        # Convert to DeviceDataLoader
        noisy_loader = DeviceDataLoader(noisy_loader, self.device)

        # Fine-tune the model with noisy data
        logger.info("Fine-tuning model with noisy data (no validation)")
        self.fit(
            epochs=2, 
            lr=learning_rate, 
            model=model,  
            train_loader=noisy_loader,
            opt_func=torch.optim.Adam
        )

        # Save the unlearned model
        unlearned_model_path = os.path.join(
            self.config_manager.get_models_path_unlearn(),
            f"unlearned_{model_name.lower()}_{dataset_name}_forget_class_{forget_class}_emmn.pth"
        )
        
        os.makedirs(os.path.dirname(unlearned_model_path), exist_ok=True)
        torch.save(model.state_dict(), unlearned_model_path)
        logger.info(f"EMMN unlearned model saved to: {unlearned_model_path}")

        logger.info("EMMN fit phase completed successfully")
        return model



    def emmn(self):
        """
        Complete EMMN unlearning method.
        Loads model from disk and gets all parameters from config manager.
        
        Returns:
            model: The unlearned model
        """
        return self.emmn_fit() 