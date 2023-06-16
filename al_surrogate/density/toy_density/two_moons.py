import torch

from ..density import Density


class TwoMoons(Density):
    """
    Bimodal two-dimensional distribution with two moons.
    """

    def __init__(self, dimension):
        super().__init__(input_dimension=2)
        if dimension != 2:
            raise ValueError("TwoMoons is only defined for dimension 2.")

    def log_prob(self, z):
        """
        ```
        log(p) = - 1/2 * ((norm(z) - 2) / 0.2) ** 2
                 + log(  exp(-1/2 * ((z[0] - 2) / 0.3) ** 2)
                       + exp(-1/2 * ((z[0] + 2) / 0.3) ** 2))
        ```

        Args:
          z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        a = torch.abs(z[:, 0])
        log_prob = (
            -0.5 * ((torch.norm(z, dim=1) - 2) / 0.2) ** 2
            - 0.5 * ((a - 2) / 0.3) ** 2
            + torch.log(1 + torch.exp(-4 * a / 0.09))
        )
        return log_prob

    def energy(self, z, beta=1.0):
        return -beta * self.log_prob(z)


if __name__ == '__main__':
    import os
    from al_surrogate.density import plot_2d_density

    two_moons = TwoMoons(dimension=2)
    plot_2d_density(
        density=two_moons, plot_energy=False,
        xlim=(-3, 3), ylim=(-3, 3),
        output_path=os.path.join(
          'al_surrogate', 'toy_density', 'figures', 'two_moons.png'
        )
    )