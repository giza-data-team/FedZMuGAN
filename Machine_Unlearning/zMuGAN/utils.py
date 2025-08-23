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
from torchvision.utils import save_image
from client_utils.general_utils import set_seed

set_seed()


def sample_noises(size):
    """
    Sample noise vectors (z).
    """
    return torch.randn(size)


def sample_labels(num_data, num_classes, dist):
    """
    Sample label vectors (y).
    """
    if dist == "onehot":
        init_labels = np.random.randint(0, num_classes, num_data)
        labels = np.zeros((num_data, num_classes), dtype=int)
        labels[np.arange(num_data), init_labels] = 1
        return torch.tensor(labels, dtype=torch.float32)
    elif dist == "uniform":
        labels = np.random.uniform(size=(num_data, num_classes))
        return torch.tensor(labels, dtype=torch.float32)
    else:
        raise ValueError(dist)


def visualize_images(generator, path, device, repeats=10):
    """
    Generate and visualize data for a generator.
    """
    generator.eval()
    nz = generator.num_noises
    ny = generator.num_classes

    noises = sample_noises(size=(repeats, nz))
    noises[0, :] = 0
    noises = np.repeat(noises.detach().numpy(), repeats=ny, axis=0)
    noises = torch.tensor(noises, dtype=torch.float32, device=device)

    labels = np.zeros((ny, ny))
    labels[np.arange(ny), np.arange(ny)] = 1
    labels = np.tile(labels, (repeats, 1))
    labels = torch.tensor(labels, dtype=torch.float32, device=device)

    images = generator(labels, noises)
    images = images.view(repeats, -1, *images.shape[1:])
    images = images.view(-1, *images.shape[2:])

    save_image(images.detach(), path, nrow=repeats, normalize=True)
