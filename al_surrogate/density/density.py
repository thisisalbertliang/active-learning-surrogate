import argparse
import ast
import os
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np


class Density(nn.Module, ABC):
    """Base class for densities"""

    def __init__(self, input_dimension: int):
        super().__init__()
        assert input_dimension > 0, f'input_dimension must be positive, but is {input_dimension}'
        self._input_dimension = input_dimension

    @abstractmethod
    def log_prob(self, x):
        """ Log pdf of x
        Args:
            x: (batch_size, d)
        Returns:
            (batch_size,)
        """
        raise NotImplementedError()

    @abstractmethod
    def energy(self, x, beta=1.0):
        """ Energy of x
        Args:
            x: (batch_size, d)
            beta: float (default: 1.0) Inverse temperature
        Returns:
            (batch_size,)
        """
        raise NotImplementedError()

    @abstractmethod
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
    def input_dimension(self):
        return self._input_dimension


def create_density(args: argparse.Namespace, device: torch.device = torch.device('cpu')) -> Density:
    if args.target == 'gaussian-mixture':
        from .toy_density.gaussian_mixture import GaussianMixture
        means = ast.literal_eval(args.means)
        covars = ast.literal_eval(args.covars)
        mixture_probs = ast.literal_eval(args.mixture_probs)
        return GaussianMixture(
            means=means, covars=covars, mixture_probs=mixture_probs,
            device=device,
        )
    elif args.target == 'unimodal-gaussian':
        from .toy_density.unimodal_gaussian import UnimodalGaussian
        mean = torch.tensor(ast.literal_eval(args.mean), device=device)
        covar = torch.tensor(ast.literal_eval(args.covar), device=device)
        return UnimodalGaussian(
            mean=mean, covar=covar, device=device,
        )
    elif args.target == 'two-moons':
        from .toy_density.two_moons import TwoMoons
        return TwoMoons(dimension=2)
    else:
        raise ValueError(f'Unknown target: {args.target}')


def plot_density(
    args: argparse.Namespace,
    density: Density, plot_energy: bool = True,
    num_points: int = 100, figsize: Tuple[int, int] = (10.0, 10.0),
    output_path: str = None,
    scatter_points: torch.Tensor = None, scatter_point_label: str = None,
    device: torch.device = torch.device('cpu'),
) -> Optional[Tuple[Figure, Axes]]:
    """ Plots 1D or 2D density

    If density is not 1D or 2D, simply does nothing and returns None.

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
    if density.input_dimension == 2:
        assert 'x' in args.input_ranges and 'y' in args.input_ranges, \
            'For 2D densities, the input ranges must be specified as "x=[x_min, x_max], y=[y_min, y_max]"'
        return plot_2d_density(
            density=density, plot_energy=plot_energy,
            xlim=args.input_ranges['x'], ylim=args.input_ranges['y'],
            num_points=num_points, figsize=figsize,
            output_path=output_path,
            scatter_point_label=scatter_point_label, scatter_points=scatter_points,
            device=device,
        )
    elif density.input_dimension == 1:
        assert 'x' in args.input_ranges, \
            'For 1D densities, the input ranges must be specified as "x=[x_min, x_max]"'
        return plot_1d_density(
            density=density, plot_energy=plot_energy,
            xlim=args.input_ranges['x'],
            num_points=num_points, figsize=figsize,
            output_path=output_path,
            scatter_point_label=scatter_point_label, scatter_points=scatter_points,
            device=device,
        )
    else:
        print(f"Cannot plot {density.input_dimension}D density")
        return None


def plot_2d_density(
    density: Density, plot_energy: bool = True,
    xlim: Tuple[float, float] = (-10.0, 10.0), ylim: Tuple[float, float] = (-10.0, 10.0),
    num_points: int = 100, figsize: Tuple[int, int] = (10.0, 10.0),
    output_path: str = None,
    scatter_points: torch.Tensor = None, scatter_point_label: str = None,
    device: torch.device = torch.device('cpu'),
) -> Optional[Tuple[Figure, Axes]]:
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
    assert density.input_dimension == 2, 'Density must be 2D'

    # Create 2D grid of points
    x = torch.linspace(xlim[0], xlim[1], num_points, device=device)
    y = torch.linspace(ylim[0], ylim[1], num_points, device=device)
    X, Y = torch.meshgrid(x, y)

    # Calculate density at each point
    Z = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    with torch.no_grad():
        Z = -density.energy(Z) if plot_energy else density.log_prob(Z).exp()
    Z = Z.reshape(num_points, num_points)

    # Plot the density
    X, Y, Z = X.cpu().numpy(), Y.cpu().numpy(), Z.cpu().numpy()
    fig, ax = plt.subplots(figsize=figsize)
    c = ax.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.colorbar(c, ax=ax)

    # Optionally plot the scatter points
    if scatter_points is not None:
        assert scatter_points.shape[1] == 2, 'Scatter points must be 2D'
        ax.scatter(scatter_points[:, 0].cpu(), scatter_points[:, 1].cpu(), color='red', label=scatter_point_label)

    if output_path is None:
        return fig, ax
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)


def plot_1d_density(
    density: Density, plot_energy: bool = True,
    xlim: Tuple[float, float] = (-10, 10),
    num_points: int = 100, figsize: Tuple[int, int] = (10, 10),
    output_path: str = None,
    scatter_points: torch.Tensor = None, scatter_point_label: str = None,
    device: torch.device = torch.device('cpu'),
) -> Optional[Tuple[Figure, Axes]]:
    assert density.input_dimension == 1, 'Density must be 1D'

    # Create a line of points
    x = torch.linspace(xlim[0], xlim[1], num_points, device=device)
    x = x.reshape(-1, 1)

    # Calculate density at each point
    with torch.no_grad():
        y = -density.energy(x) if plot_energy else density.log_prob(x).exp()

    # Plot the density
    x, y = x.cpu().numpy(), y.cpu().numpy()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, color='blue')
    ax.set_xlabel('x')
    ax.set_ylabel('Energy' if plot_energy else 'Density')

    # Optionally plot the scatter points
    if scatter_points is not None:
        assert scatter_points.shape[1] == 1, 'Scatter points must be 1D'
        scatter_points = scatter_points.cpu().numpy()
        ax.scatter(scatter_points, np.zeros_like(scatter_points), color='red', label=scatter_point_label)

    if output_path is None:
        return fig, ax
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
