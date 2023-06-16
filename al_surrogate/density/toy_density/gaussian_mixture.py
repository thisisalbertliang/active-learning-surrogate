from typing import List
import numpy as np
import torch
import torch.distributions as D

from ..density import Density


class GaussianMixture(Density):

    def __init__(
        self,
        means: List[List[float]],
        covars: List[List[List[float]]],
        mixture_probs: List[float],
        device=torch.device('cpu')
    ):
        assert means is not None and covars is not None and mixture_probs is not None, 'means, covars and mixture_prob must be specified'
        assert len(means) == len(covars) == len(mixture_probs), 'must have the same number of means, covars and mixture_prob'
        assert all(len(mean) == len(covar) == len(means[0]) for mean, covar in zip(means, covars)), 'must have same dimensionality for all means and covars'

        # Create a batch of MultivariateNormal distributions
        components = D.MultivariateNormal(
            loc=torch.tensor(means, device=device),
            covariance_matrix=torch.tensor(covars, device=device)
        )

        # Create a categorical distribution for the mixture probabilities
        mix = D.Categorical(torch.tensor(mixture_probs, device=device))

        # Create the mixture distribution
        self.gaussian_mixture = D.MixtureSameFamily(mix, components)

        # Dimensionality
        super().__init__(input_dimension=len(means[0]))

    def log_prob(self, x):
        return self.gaussian_mixture.log_prob(x)

    def energy(self, x, beta=1.0):
        return -beta * self.log_prob(x)

    def sample(self, num_samples: int):
        return self.gaussian_mixture.sample((num_samples,))

    @property
    def means(self):
        return torch.stack([gaussian.mean for gaussian in self.gaussians])

    @property
    def num_modes(self):
        return len(self.gaussians)


if __name__ == '__main__':
    import os
    from al_surrogate.density import plot_2d_density

    mean1, covar1 = [10, 10], [[2.0, 0.0], [0.0, 2.0]]
    mean2, covar2 = [-10, -10], [[1.0, 0.0], [0.0, 1.0]]
    bimodal_gaussian = GaussianMixture(
        means=[mean1, mean2],
        covars=[covar1, covar2],
        mixture_probs=[0.5, 0.5],
    )

    plot_2d_density(
        density=bimodal_gaussian, plot_energy=False,
        xlim=(-20, 20), ylim=(-20, 20),
        num_points=100, figsize=(10, 10),
        output_path=os.path.join(
            'al_surrogate', 'toy_density', 'figures', 'bimodal_gaussian.png'
        )
    )
