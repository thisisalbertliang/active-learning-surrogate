import argparse
from typing import Dict, Tuple
import torch
from hyperopt import hp
import hyperopt.pyll.stochastic

from .query_strategy import QueryStrategy


class UniformSampling(QueryStrategy):
    """Population-based Uniform Sampling

    This is the most basic acquisition function. It simply samples uniformly from the population.
    """
    def __init__(
        self,
        input_ranges: Dict[str, Tuple[float, float]],
        target: torch.nn.Module,
        device: torch.device = torch.device('cpu'),
    ):
        """Initialize the class with the search space."""
        super().__init__(input_ranges, target, device)

        # overwrite the input_ranges with hyperopt's format
        self.input_ranges = {
            k: hp.uniform(k, low, high) for k, (low, high) in input_ranges.items()
        }
        self.input_order = list(input_ranges.keys())

    def sample(self, num_samples: int) -> torch.Tensor:
        samples = []
        for _ in range(num_samples):
            sample_dict = hyperopt.pyll.stochastic.sample(self.input_ranges)
            # Ensure the inputs are in the same order as the input ranges
            sample_list = [sample_dict[input] for input in self.input_order]
            samples.append(sample_list)

        query_X = torch.tensor(samples, device=self.device)
        query_Y = self.target(query_X)
        return query_X, query_Y


if __name__ == '__main__':
    # Example usage of the `UniformSampling` class
    import matplotlib.pyplot as plt

    # Define the ranges for two parameters: 'x' and 'y'
    param_ranges = {'x': (0, 1), 'y': (-1, 1)}

    # Create an instance of UniformSampling with the defined ranges
    sampler = UniformSampling(param_ranges)

    # Generate 5 samples from the search space
    samples = sampler.sample(5)

    # Convert the tensor to a numpy array for plotting
    samples_np = samples.numpy()

    # Create a scatter plot of the samples
    plt.scatter(samples_np[:, 0], samples_np[:, 1])

    # Set the labels for the axes
    plt.xlabel('x')
    plt.ylabel('y')

    # Show the plot
    plt.show()
