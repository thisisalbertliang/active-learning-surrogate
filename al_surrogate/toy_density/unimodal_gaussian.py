from typing import List
import torch
import torch.distributions as D

from al_surrogate.density import Density


class UnimodalGaussian(Density):
    """Unimodal Gaussian distribution"""

    def __init__(
        self,
        mean: List[float],
        covar: List[List[float]],
        device=torch.device('cpu')
    ):
        assert mean is not None and covar is not None, 'mean and covar must be specified'
        assert len(covar) == len(covar[0]), 'covar must be a square matrix'
        assert len(mean) == len(covar), 'mean and covar must have the same dimensionality'

        # Implements the Gaussian distribution using PyTorch distributions
        self.gaussian = D.MultivariateNormal(
            loc=torch.tensor(mean, dtype=torch.float32, device=device),
            covariance_matrix=torch.tensor(covar, dtype=torch.float32, device=device),
        )

        # Dimensionality
        super().__init__(self.gaussian.mean.shape[0])

    def log_prob(self, x):
        return self.gaussian.log_prob(x)

    def energy(self, x, beta=1.0):
        return -beta * self.log_prob(x)

    def sample(self, num_samples):
        return self.gaussian.sample((num_samples,))


if __name__ == '__main__':
    import os
    from al_surrogate.density import plot_2d_density

    mean, covar = [6, 3], [[1, 0], [0, 1]]
    unimodal_gaussian = UnimodalGaussian(mean=mean, covar=covar)

    plot_2d_density(
        density=unimodal_gaussian, plot_energy=False,
        xlim=(-5, 15), ylim=(-5, 15),
        num_points=100, figsize=(10, 10),
        output_path=os.path.join(
            'al_surrogate', 'toy_density', 'figures', 'unimodal_gaussian.png'
        )
    )
