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

from torch import nn
from client_utils.general_utils import set_seed

set_seed()


class Decoder(nn.Module):
    """
    Decoder for both unstructured and image datasets.
    """

    def __init__(self, in_features, out_targets, n_layers, units=120):
        """
        Class initializer.
        """
        super(Decoder, self).__init__()

        layers = [nn.Linear(in_features, units), nn.ELU(), nn.BatchNorm1d(units)]

        for _ in range(n_layers):
            layers.extend([nn.Linear(units, units), nn.ELU(), nn.BatchNorm1d(units)])

        layers.append(nn.Linear(units, out_targets))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward propagation.
        """
        out = x.view((x.size(0), -1))
        out = self.layers(out)
        return out
