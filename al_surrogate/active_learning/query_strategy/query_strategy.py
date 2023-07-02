import argparse
from typing import Dict, Tuple
from abc import ABC, abstractmethod
import torch


class QueryStrategy(ABC):
    """Abstract class for population-based query strategies"""

    def __init__(self, input_dimension: int):
        """
        Args:
            input_dimension (int): Dimension of the input space.
        """
        self._input_dimension = input_dimension

    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """Sample from the search space.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: A tensor of shape (num_samples, input_dimension) containing the samples.
        """
        pass

    @property
    def input_dimension(self):
        return self._input_dimension


def create_query_strategy(args: argparse.Namespace, device: torch.device) -> QueryStrategy:
    """Creates a query strategy based on the command line arguments.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        QueryStrategy: A query strategy.
    """
    if args.query_strategy == 'uniform-sampling':
        from .uniform_sampling import UniformSampling
        return UniformSampling(args.input_ranges)
    elif args.query_strategy == 'greedy-sampling':
        from .greedy_sampling import GreedySampling
        return GreedySampling(args.input_ranges, device=device)
    else:
        raise ValueError(f'Unknown query strategy: {args.query_strategy}')