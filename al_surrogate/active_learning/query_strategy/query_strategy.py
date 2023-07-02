import argparse
from typing import Dict, Tuple
from abc import ABC, abstractmethod
import torch

from al_surrogate.density import Density
from ..surrogate import Surrogate


class QueryStrategy(ABC):
    """Abstract class for population-based query strategies"""

    def __init__(
        self,
        input_ranges: Dict[str, Tuple[float, float]],
        target: torch.nn.Module,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Args:
            input_dimension (int): Dimension of the input space.
        """
        self.input_ranges: Dict[str, Tuple[float, float]] = input_ranges
        self.input_dimension: int = len(input_ranges)
        self.target: Density = target
        self.device: torch.device = device

    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from the search space.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of tensors containing the input and output samples.
        """
        pass


def create_query_strategy(
    args: argparse.Namespace,
    target: torch.nn.Module,
    surrogate: Surrogate,
    device: torch.device = torch.device('cpu'),
) -> QueryStrategy:
    """Creates a query strategy based on the command line arguments.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        QueryStrategy: A query strategy.
    """
    if args.query_strategy == 'uniform-sampling':
        from .uniform_sampling import UniformSampling
        return UniformSampling(
            input_ranges=args.input_ranges,
            target=target,
            device=device,
        )
    elif args.query_strategy == 'greedy-sampling':
        from .greedy_sampling import GreedySampling
        return GreedySampling(
            input_ranges=args.input_ranges,
            target=target,
            device=device,
        )
    elif args.query_strategy == 'improved-greedy-sampling':
        from .improved_greedy_sampling import ImprovedGreedySampling
        return ImprovedGreedySampling(
            input_ranges=args.input_ranges,
            target=target,
            surrogate=surrogate,
            device=device,
        )
    else:
        raise ValueError(f'Unknown query strategy: {args.query_strategy}')
