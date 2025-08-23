import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
import logging
from config_manager import ConfigurationManager


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield self.to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

    def to_device(self, data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list, tuple)):
            return [self.to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0, device="cuda"):
        self.std = std
        self.mean = mean
        self.device = device

    def __call__(self, tensor):
        _max = tensor.max()
        _min = tensor.min()
        tensor = (
            tensor + torch.randn(tensor.size()).to(self.device) * self.std + self.mean
        )
        tensor = torch.clamp(tensor, min=_min, max=_max)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class LipschitzUnlearning:
    def __init__(
        self,
        model,
        device,
        forget_dataloader,
        opt_func=torch.optim.Adam,
        learning_rate=0.001,
        noise_std=0.1,
    ):
        self.model = model
        self.device = device
        self.forget_dataloader = DeviceDataLoader(forget_dataloader, device)
        self.opt_func = opt_func
        self.learning_rate = learning_rate
        self.noise_std = noise_std
        self.config_manager = ConfigurationManager()

        logger.info(
            f"Initializing Lipschitz Unlearning with lr={learning_rate}, noise_std={noise_std}"
        )

    def lipschitz_unlearn(self):
        """
        Perform lipschitz unlearning on the model using the forget dataset.
        """
        logger.info("Starting Lipschitz unlearning")

        noise = AddGaussianNoise(std=self.noise_std, device=self.device)
        optimizer = self.opt_func(self.model.parameters(), lr=self.learning_rate)

        self.model.train()

        for batch_idx, sample in enumerate(self.forget_dataloader):
            x = sample[0].to(self.device)

            # Ensure proper dimensionality
            image = x.unsqueeze(0) if x.dim() == 3 else x

            # Forward pass on original image
            out, *_ = self.model(image)
            loss = torch.tensor(0.0, device=self.device)

            # Build comparison images (100 noisy versions)
            for _ in range(100):
                img2 = noise(deepcopy(x))
                image2 = img2.unsqueeze(0) if img2.dim() == 3 else img2

                with torch.no_grad():
                    out2, *_ = self.model(image2)

                # Ignore batch dimension
                flatimg = image.view(image.size()[0], -1)
                flatimg2 = image2.view(image2.size()[0], -1)

                # Calculate norms
                in_norm = torch.linalg.vector_norm(flatimg - flatimg2, dim=1)
                out_norm = torch.linalg.vector_norm(out - out2, dim=1)

                # Lipschitz constraint loss
                K = ((out_norm / in_norm).sum()).abs()
                loss += K

            # Average the loss over all noise comparisons
            loss /= 100

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                logger.info(
                    f"Batch {batch_idx}/{len(self.forget_dataloader)}, Loss: {loss.item():.6f}"
                )

        logger.info("Lipschitz unlearning completed")
        return self.model
