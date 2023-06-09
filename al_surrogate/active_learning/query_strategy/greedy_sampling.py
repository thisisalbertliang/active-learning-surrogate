import argparse
from typing import Dict, Tuple
import torch
from pyDOE2 import lhs

from .query_strategy import QueryStrategy


class GreedySampling(QueryStrategy):
    """Population-based Greedy Sampling (GS)

    Mathmatically, GS selects a sample that maximizes the minimum distance
        x_new = argmax_x min_{x in \mathcal{X}} d(x)
        d(x) = min_{k=1,...,N} || \sigma(x) - \sigma(x_k) ||_2^2
    where
        * \mathcal{X} is the bounded input space
        * N is the number of training samples
        * \sigma is the scaling function mapping from \mathcal{X} to [-1, 1]^d
    """

    def __init__(
        self,
        input_ranges: Dict[str, Tuple[float, float]],
        target: torch.nn.Module,
        device: torch.device = torch.device('cpu')
    ):
        """Initializes the Greedy Sampling query strategy."""
        super().__init__(input_ranges, target, device)

        self.past_X = torch.empty((0, len(input_ranges))).to(device=self.device)
        self.min_vals = torch.tensor([v[0] for v in self.input_ranges.values()]).float().to(device=self.device)
        self.max_vals = torch.tensor([v[1] for v in self.input_ranges.values()]).float().to(device=self.device)
        self.range_widths = self.max_vals - self.min_vals


    def sample(self, num_samples: int) -> torch.Tensor:
        """Sample from the search space.

        Directly optimizing argmax_x min_{x in \mathcal{X}} d(x) is intractable.
        Thus, we approximate the argmax by generating random samples from a Latin Hypercube
        and selecting the sample that maximizes the minimum distance.

        Args:
            num_samples (int): Number of new samples to generate.

        Returns:
            torch.Tensor: A tensor of shape (num_samples, input_dimension) containing the samples.
        """
        query_X = torch.empty((num_samples, self.input_dimension)).to(device=self.device)
        for i in range(num_samples):
            if len(self.past_X) == 0:
                # If no past samples, just sample randomly
                new_x = torch.rand(self.input_dimension).to(device=self.device)
                new_x = self.min_vals + self.range_widths * new_x
            else:
                # Generate Latin Hypercube Samples from the unit hypercube [0, 1]^d
                candidates = torch.from_numpy(
                    lhs(self.input_dimension, samples=1000)
                ).float().to(device=self.device)

                # Scale the LHS to the correct input ranges
                candidates = self.min_vals + self.range_widths * candidates

                # Compute the minimum distance to the past samples for each candidate
                scaled_past_samples = scale(self.past_X, self.input_ranges)
                scaled_candidates = scale(candidates, self.input_ranges)
                distances = (scaled_candidates[:, None, :] - scaled_past_samples[None, :, :]).norm(dim=-1).min(dim=-1)[0]

                # Select the candidate that maximizes the minimum distance
                new_x = candidates[distances.argmax()]

            # Add the new sample to be returned
            query_X[i, :] = new_x

            # Add the new sample to all past samples
            self.past_X = torch.cat(
                (self.past_X, new_x.view(1, -1)),
                dim=0,
            )

        query_Y = self.target.energy(query_X)
        return query_X, query_Y


def scale(x: torch.Tensor, input_ranges: Dict[str, Tuple[float, float]]) -> torch.Tensor:
    """Scales the input from the input ranges to [-1, 1]^d

    Args:
        x (torch.Tensor): Input tensor of shape (num_samples, input_dimension).
        input_ranges (Dict[str, Tuple[float, float]]): Dictionary of input ranges.

    Returns:
        torch.Tensor: Scaled input tensor of shape (num_samples, input_dimension).
    """
    # convert the input_ranges into tensors
    min_vals = torch.tensor([v[0] for v in input_ranges.values()]).to(x.device)
    max_vals = torch.tensor([v[1] for v in input_ranges.values()]).to(x.device)

    # Reshape to (1, input_dimension) for broadcastin
    min_vals = min_vals.view(1, -1)
    max_vals = max_vals.view(1, -1)
    # Apply the scaling operation
    x_scaled = 2 * (x - min_vals) / (max_vals - min_vals) - 1

    return x_scaled
