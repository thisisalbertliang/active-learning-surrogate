from typing import List
import numpy as np
import torch

from density import Density
from unimodal_gaussian import UnimodalGaussian


class GaussianMixture(Density):

    def __init__(
        self,
        means: List[List[float]],
        covars: List[List[List[float]]],
        mixture_prob: List[float],
        device=torch.device('cpu')
    ):
        assert means is not None and covars is not None and mixture_prob is not None, 'means, covars and mixture_prob must be specified'
        assert len(means) == len(covars) == len(mixture_prob), 'must have the same number of means, covars and mixture_prob'
        assert np.isclose(np.sum(mixture_prob), 1.0), 'mixture_prob must sum to 1.0'
        assert all(len(mean) == len(covar) == len(means[0]) for mean, covar in zip(means, covars)), 'must have same dimensionality for all means and covars'

        # Create the unimodal Gaussian distributions
        self.gaussians = [UnimodalGaussian(mean, covar, device=device) for mean, covar in zip(means, covars)]
        self.mixture_prob = torch.tensor(mixture_prob, device=device)

        # Dimensionality
        super().__init__(self.gaussians[0].dimension)

        # Save the device
        self.device = device

    def log_prob(self, x):
        # Evaluate the log probabilities under each of the individual Gaussians
        log_probs = [gaussian.log_prob(x) for gaussian in self.gaussians]
        log_probs = torch.stack(log_probs, dim=-1)  # Shape: [num_samples, num_components]

        # Take the log of the mixture probabilities. This is a tensor of shape [num_components].
        log_mixture_prob = self.mixture_prob.log()

        # Add an extra dimension to the start of the tensor to make it compatible with log_probs.
        # This changes its shape from [num_components] to [1, num_components].
        log_mixture_prob = log_mixture_prob.unsqueeze(0)

        # Now we can weight the log probabilities by the mixing proportions. Broadcasting will automatically
        # replicate the mixture probabilities across the extra dimension, so weighted_log_probs will also
        # have shape [num_samples, num_components].
        weighted_log_probs = log_mixture_prob + log_probs

        # Compute the total log probability under the mixture model
        # This is done by using the log-sum-exp trick to prevent underflow during the summation
        log_prob_total = torch.logsumexp(weighted_log_probs, dim=-1) # Shape: [num_samples]

        return log_prob_total

    def energy(self, x, beta=1.0):
        return -beta * self.log_prob(x)

    def sample(self, num_samples: int):
        # Sample component indices according to the mixture probabilities
        component_indices = torch.multinomial(self.mixture_prob, num_samples, replacement=True)

        # Sample from the selected Gaussian distributions
        samples = torch.stack([self.gaussians[i].sample(1) for i in component_indices]) # Shape: [num_samples, dimension]

        return samples

    @property
    def means(self):
        return torch.stack([gaussian.mean for gaussian in self.gaussians]).to(self.device)

    @property
    def num_modes(self):
        return len(self.gaussians)
