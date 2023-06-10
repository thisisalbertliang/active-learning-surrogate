import os
from typing import Tuple, Union
import matplotlib.pyplot as plt
import torch
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


def plot_2d_density(
    density: Density, plot_energy: bool = True,
    xlim: Tuple[float, float] = (-10, 10), ylim: Tuple[float, float] = (-10, 10),
    num_points: int = 100, figsize: Tuple[int, int] = (10, 10),
    output_path: str = None,
    device: torch.device = torch.device('cpu'),
) -> Union[Tuple[plt.Figure, plt.Axes], None]:
    """ Plots a 2D-density

    Args:
        density: Density object
        plot_energy: bool (default: True) If True, plot the negative energy function. Otherwise plot the density.
        xlim: Tuple[float, float] (default: (-10, 10)) x-axis limits
        ylim: Tuple[float, float] (default: (-10, 10)) y-axis limits
        num_points: int (default: 100) Number of points (per dimension) to evaluate the density at
        figsize: Tuple[int, int] (default: (10, 10)) Size of the figure
        output_path: str (default: None) If None, return the figure. Otherwise save the figure to the given path.
        device: torch.device (default: torch.device('cpu')) Device to evaluate the density on
    """
    assert density.dimension == 2, 'Density must be 2D'

    # Create 2D grid of points
    x = torch.linspace(xlim[0], xlim[1], num_points, device=device)
    y = torch.linspace(ylim[0], ylim[1], num_points, device=device)
    X, Y = torch.meshgrid(x, y)

    # Calculate density at each point
    Z = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    Z = -density.energy(Z) if plot_energy else density.log_prob(Z).exp()
    Z = Z.reshape(num_points, num_points)

    # Plot the density
    X, Y, Z = X.cpu().numpy(), Y.cpu().numpy(), Z.cpu().numpy()
    fig, ax = plt.subplots(figsize=figsize)
    c = ax.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.colorbar(c, ax=ax)

    if output_path is None:
        return fig, ax
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
