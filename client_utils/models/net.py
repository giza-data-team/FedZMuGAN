import torch.nn as nn
import torch.nn.functional as F
import torch
from .base_model import BaseModel


class Net(BaseModel):
    def __init__(self):
        super(Net, self).__init__()
        input_channels = self.dataset_config["img_channels"]
        input_size = self.dataset_config.get(
            "img_size", 28
        )  # Default to 28 if not specified

        self.conv1 = nn.Conv2d(
            input_channels, 32, kernel_size=3, padding=1
        )  # Increased filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Increased filters
        self.pool = nn.MaxPool2d(2, 2)
        fc_input_size = (
            (input_size // 4) * (input_size // 4) * 64
        )  # Two pooling layers reduce size by a factor of
        self.fc1 = nn.Linear(fc_input_size, 512)  # Increased hidden units
        self.fc2 = nn.Linear(512, self.dataset_config["num_classes"])

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No softmax here, CrossEntropyLoss does it
        return x
