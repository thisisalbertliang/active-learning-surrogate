import argparse
from typing import List, Tuple
from abc import ABC, abstractmethod
import torch


class Surrogate(ABC):
    """Abstract class for all surrogate models"""

    def __init__(
        self, args: argparse.Namespace, input_dimension: int, device: torch.device = torch.device('cpu'),
    ):
        """
        Args:
            args (argparse.Namespace): Command line arguments.
            device (torch.device): Device to use for training and prediction.
        """
        self._args = args
        self._device = device
        self._input_dimension = input_dimension

    @abstractmethod
    def train(self, train_X: torch.Tensor, train_y: torch.Tensor) -> Tuple[torch.nn.Module, List[float]]:
        """Trains a surrogate model from scratch on the given data.

        Args:
            args (argparse.Namespace): Command line arguments.
            train_X (torch.Tensor): A tensor of shape (num_samples, input_dimension) containing the samples.
            train_y (torch.Tensor): A tensor of shape (num_samples,) containing the corresponding scalar outputs.

        Returns:
            Tuple[torch.nn.Module, List[float]]: A tuple containing the trained surrogate model and a list of training losses.
        """
        pass

    @property
    def input_dimension(self):
        return self._input_dimension


def create_surrogate(args: argparse.Namespace, input_dimension: int, device: torch.device = torch.device('cpu')):
    """Creates a surrogate model based on the command line arguments.

    Args:
        args (argparse.Namespace): Command line arguments.
        input_dimension (int): Dimension of the input space.
        device (torch.device): Device to use for training and prediction.

    Returns:
        Surrogate: A surrogate model.
    """
    if args.surrogate == 'gaussian-process':
        from .gaussian_process import GaussianProcess
        return GaussianProcess(args, input_dimension, device)
    else:
        raise ValueError(f'Unknown surrogate model: {args.surrogate}')