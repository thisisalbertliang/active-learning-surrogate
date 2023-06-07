from typing import List
from torch import nn


class Density(nn.Module):
    """Base class for densities"""

    def __init__(self, dimension: int):
        super().__init__()
        self._dimension = dimension

    def log_prob(self, x):
        """ Log pdf of x
        Args:
            x: (batch_size, d)
        Returns:
            (batch_size,)
        """
        raise NotImplementedError()

    def energy(self, x, beta=1.0):
        """ Energy of x
        Args:
            x: (batch_size, d)
            beta: float (default: 1.0) Inverse temperature
        Returns:
            (batch_size,)
        """
        raise NotImplementedError()

    def sample(self, num_samples):
        """ Sample from the density
        Args:
            num_samples: int
        Returns:
            (num_samples, d)
        """
        raise NotImplementedError()

    def forward(self, x, beta=1.0):
        return self.energy(x, beta=beta)

    @property
    def dimension(self):
        return self._dimension
