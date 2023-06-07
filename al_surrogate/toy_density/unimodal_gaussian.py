from typing import List
import numpy as np
import torch

from density import Density


class UnimodalGaussian(Density):
    """Unimodal Gaussian distribution"""

    def __init__(
        self,
        mean: List[float],
        covar: List[List[float]],
        device=torch.device('cpu')
    ):
        assert mean is not None and covar is not None, 'mean and covar must be specified'
        assert len(covar) == len(covar[0]), 'covar must be a square matrix'
        assert len(mean) == len(covar), 'mean and covar must have the same dimensionality'

        # Convert mu and sigma to tensors and move them to the specified device
        self.mean = torch.tensor(mean, device=device)
        self.covar = torch.tensor(covar, device=device)

        # Dimensionality
        super().__init__(self.mean.shape[0])

    def log_prob(self, x):
        return -0.5 * (self.dimension * np.log(2 * np.pi) + torch.logdet(self.covar)) - self.energy(x)

    def energy(self, x, beta=1.0):
        batch_size = x.shape[0]
        # Reshape tensors to be compatible with bmm
        x = x.view(batch_size, self.dimension, 1)  # (batch_size, d, 1)
        mean = self.mean.view(1, self.dimension, 1).expand(batch_size, self.dimension, 1)  # (batch_size, d, 1)

        x_minus_mu = x - mean  # (batch_size, d, 1)

        # Compute the inverse of sigma
        covar_inv = self.covar.inverse()  # (d, d)

        # Reshape sigma_inv to be compatible with bmm
        covar_inv = covar_inv.view(1, self.dimension, self.dimension).expand(batch_size, self.dimension, self.dimension)  # (batch_size, d, d)

        # Compute energy
        energy = torch.bmm(x_minus_mu.transpose(1, 2), covar_inv)  # (batch_size, 1, d)
        energy = torch.bmm(energy, x_minus_mu)  # (batch_size, 1, 1)

        return beta * energy.squeeze()  # (batch_size,)

    def sample(self, num_samples):
        return torch.randn(num_samples, self.dimension) @ self.covar.sqrt() + self.mean
