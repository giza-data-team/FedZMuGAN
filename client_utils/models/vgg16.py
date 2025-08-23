import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(BaseModel):
    def __init__(self, vgg_name, return_activations=False, dropout_rate=0.5):
        super(VGG, self).__init__()
        self.features = self._make_layers(
            cfg[vgg_name], channels=self.dataset_config["img_channels"]
        )
        self.dropout = nn.Dropout(dropout_rate)

        # Use adaptive pooling to handle any input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Now the input to first linear layer will always be 512 regardless of input image size
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, self.dataset_config["num_classes"]),
        )
        self.return_activations = return_activations

    def forward(self, x):
        if not self.return_activations:
            out = self.features(x)
            out = self.adaptive_pool(out)  # Apply adaptive pooling to get fixed size
            out = out.view(out.size(0), -1)  # Flatten
            out = self.dropout(out)
            out = self.classifier(out)
            return out

        activation_list = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.Conv2d) and (x.numel() > 0):
                activation_list.append(x)

        # Apply adaptive pooling before flattening
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        # Apply the classifier layers one by one to collect activations
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            if isinstance(layer, nn.ReLU) and (x.numel() > 0):
                activation_list.append(x)

        return x, activation_list

    def _make_layers(self, cfg, channels):
        layers = []
        in_channels = channels
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                # Add dropout after some conv layers (not after every layer to preserve features)
                if x >= 256:  # Add dropout after layers with larger feature maps
                    layers += [
                        nn.Dropout2d(0.2)
                    ]  # Spatial dropout for convolutional layers
                in_channels = x
        # Remove the last AvgPool layer - we'll use adaptive pooling instead
        return nn.Sequential(*layers)


class VGG16(VGG):
    def __init__(self, return_activations=False, dropout_rate=0.5):
        super(VGG16, self).__init__(
            vgg_name="VGG16",
            return_activations=return_activations,
            dropout_rate=dropout_rate,
        )


class VGG13(VGG):
    def __init__(self, return_activations=False, dropout_rate=0.5):
        super(VGG13, self).__init__(
            vgg_name="VGG13",
            return_activations=return_activations,
            dropout_rate=dropout_rate,
        )
