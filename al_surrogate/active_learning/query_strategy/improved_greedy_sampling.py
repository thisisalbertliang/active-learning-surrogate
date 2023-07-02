from typing import Dict, Tuple
import torch
from pyDOE2 import lhs

from ..surrogate import Surrogate
from .query_strategy import QueryStrategy
from .greedy_sampling import scale


class ImprovedGreedySampling(QueryStrategy):
    """Population-based Improved Greedy Sampling (iGS)

    iGS selects a sample that maximizes the product of:
    1. The minimum distance to the existing samples in the input X space
    2. The minimum distance to the existing samples in the target Y space
    Mathmatically:
        x_new = argmax_x min_{x in \mathcal{X}} d_x(x) * d_y(x)
        d_x(x) = min_{k=1,...,N} || \sigma(x) - \sigma(x_k) ||_2^2
        d_y(x) = min_{k=1,...,N} || model(x) - y_k ||_2^2
    where
        * \mathcal{X} is the bounded input space
        * N is the number of training samples
        * \sigma is the scaling function mapping from \mathcal{X} to [-1, 1]^d
        * model is the current surrogate
    """

    def __init__(
        self,
        input_ranges: Dict[str, Tuple[float, float]],
        target: torch.nn.Module,
        surrogate: Surrogate,
        device: torch.device = torch.device('cpu')
    ):
        """Initializes the Improved Greedy Sampling query strategy.

        Args:
            input_ranges (Dict[str, Tuple[float, float]]): Dictionary of input ranges.
        """
        super().__init__(input_ranges, target, device)
        self.surrogate: Surrogate = surrogate

        self.past_X: torch.Tensor = None
        self.past_y: torch.Tensor = None
        self.min_vals = torch.tensor([v[0] for v in self.input_ranges.values()]).float().to(device=self.device)
        self.max_vals = torch.tensor([v[1] for v in self.input_ranges.values()]).float().to(device=self.device)
        self.range_widths = self.max_vals - self.min_vals


    def sample(self, num_samples: int) -> torch.Tensor:
        """Sample from the search space.

        Directly optimizing argmax_x min_{x in \mathcal{X}} d_x(x) * d_y(x) is intractable.
        Thus, we approximate the argmax by generating random samples from a Latin Hypercube
        and selecting the sample that maximizes the minimum distance.

        Args:
            num_samples (int): Number of new samples to generate.

        Returns:
            torch.Tensor: A tensor of shape (num_prev_samples + num_samples, input_dimension) containing the samples.
        """
        if self.past_X is None:
            # If no past samples, just sample randomly
            query_X = torch.rand((num_samples, self.input_dimension)).to(device=self.device)
            query_X = self.min_vals + self.range_widths * query_X
            query_y = self.target.energy(query_X).squeeze()
            self.past_X, self.past_y = query_X, query_y
        else:
            query_X = torch.empty((num_samples, self.input_dimension)).to(device=self.device)
            query_y = torch.empty(num_samples).to(device=self.device)

            for i in range(num_samples):
                # Generate random samples from a Latin Hypercube
                candidates = torch.from_numpy(
                    lhs(self.input_dimension, samples=1000)
                ).float().to(device=self.device)

                # Scale the candidates to the input ranges
                candidates = self.min_vals + self.range_widths * candidates

                # Compute the minimum distance to the existing samples in the input X space
                min_d_x = torch.min(torch.cdist(candidates, self.past_X), dim=1).values
                # Compute the minimum distance to the existing targets in the Y space
                pred_y = self.surrogate.predict(candidates)
                min_d_y = torch.min(torch.cdist(pred_y.unsqueeze(1), self.past_y.unsqueeze(1)), dim=1).values

                # Compute the product of the minimum distances
                min_d = min_d_x * min_d_y

                # Select the candidate with the maximum minimum distance
                best_idx = torch.argmax(min_d)
                new_x = candidates[best_idx, :]
                new_y = pred_y[best_idx]

                # Add the new sample to be returned
                query_X[i, :] = new_x
                query_y[i] = new_y

                # Add the new sample to the past samples
                self.past_X = torch.cat((self.past_X, new_x.unsqueeze(0)), dim=0)
                self.past_y = torch.cat((self.past_y, new_y.unsqueeze(0)), dim=0)

        return query_X, query_y
