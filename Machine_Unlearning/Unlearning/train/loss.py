import torch.nn.functional as F
import torch.nn as nn


class DistillKL(nn.Module):
    """
    Compute a temperature-scaled KL divergence loss between two models' outputs.
    """

    def __init__(self, T):
        """
        Initialize with a temperature T to scale logits for softening distributions.
        """

        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        """
        Return the KL divergence loss between the softened outputs the two models.
        """

        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss
