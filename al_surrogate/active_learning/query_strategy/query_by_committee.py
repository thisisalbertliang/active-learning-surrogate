import argparse
from typing import Dict, Tuple, List
import torch
from pyDOE2 import lhs

from ..surrogate import Surrogate
from .query_strategy import QueryStrategy


class QueryByCommittee(QueryStrategy):
    """Population-based Query-by-Committee (QBC)

    Mathematically, QBC selects a sample that maximizes the prediction variance among a committee of bootstrapped surrogate models:
        x_new = argmax_x \sum_{k=1}^K || y_k - \bar{y} ||_2^2
    where
        * K is the number of bootstrapped surrogate models in the committee
        * y_k is the prediction of the k-th surrogate model
        * \bar{y} is the average prediction of the committee

    See section 3.1 of https://arxiv.org/pdf/1808.04245.pdf for more details.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        input_ranges: Dict[str, Tuple[float, float]],
        target: torch.nn.Module,
        surrogate: Surrogate,
        device: torch.device = torch.device('cpu')
    ):
        """Initializes the Query-by-Committee query strategy."""
        super().__init__(input_ranges, target, device)
        # gets the class of the surrogate, needed for bootstrapping
        self.surrogate_cls = surrogate.__class__
        self.args = args

        self.past_X: torch.Tensor = None
        self.past_y: torch.Tensor = None
        self.min_vals = torch.tensor([v[0] for v in self.input_ranges.values()]).float().to(device=self.device)
        self.max_vals = torch.tensor([v[1] for v in self.input_ranges.values()]).float().to(device=self.device)
        self.range_widths = self.max_vals - self.min_vals

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from the search space.

        Directly optimizing argmax_x \sum_{k=1}^K || y_k - \bar{y} ||_2^2 is intractable.
        Thus, we approximate the argmax by generating random samples from a Latin Hypercube
        and selecting the sample that maximizes the minimum distance.
        """
        if self.past_X is None:
            # If no past samples, just sample randomly
            query_X = torch.rand((num_samples, self.input_dimension)).to(device=self.device)
            query_X = self.min_vals + self.range_widths * query_X
            query_y = self.target.energy(query_X).squeeze()
            self.past_X, self.past_y = query_X, query_y
        else:
            query_X = torch.empty((num_samples, self.input_dimension)).to(device=self.device)
            query_y = torch.empty((num_samples)).to(device=self.device)

            for i in range(num_samples):
                # Generate Latin Hypercube Samples from the unit hypercube [0, 1]^d
                candidates = torch.from_numpy(
                    lhs(self.input_dimension, samples=1000)
                ).float().to(device=self.device)

                # Scale the LHS to the correct input ranges
                candidates = self.min_vals + self.range_widths * candidates

                # Create K surrogate models by bootstrapping the past samples
                bootstrapped_surrogates: List[Surrogate] = []
                for k in range(self.args.num_committee_members):
                    bootstrap_indices = torch.randint(
                        0, self.past_X.shape[0], (self.past_X.shape[0],)
                    ).to(device=self.device) # sample with replacement
                    bootstrap_X = self.past_X[bootstrap_indices]
                    bootstrap_y = self.past_y[bootstrap_indices]
                    # train the surrogate model on the bootstrap samples
                    surrogate = self.surrogate_cls(
                        args=self.args, input_dimension=len(self.min_vals), device=self.device
                    )
                    surrogate.train(bootstrap_X, bootstrap_y)
                    bootstrapped_surrogates.append(surrogate)

                # Compute the variance of the predictions of the committee
                variances = torch.var(
                    torch.stack([s.predict(candidates) for s in bootstrapped_surrogates], dim=0), # shape: (K, num_candidates)
                    dim=0
                )
                assert variances.shape == (candidates.shape[0],)

                # Select the candidate that maximizes the prediction variance
                best_candidate_idx = torch.argmax(variances)

                # Add the selected candidate to the query set to be returned
                query_X[i] = candidates[best_candidate_idx]
                query_y[i] = self.target.energy(query_X[i]).squeeze()

                # Add the selected candidate to the past samples
                self.past_X = torch.cat((self.past_X, query_X[i].unsqueeze(0)), dim=0)
                self.past_y = torch.cat((self.past_y, query_y[i].unsqueeze(0)), dim=0)

        return query_X, query_y
