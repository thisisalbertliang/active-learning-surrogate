import argparse
from tqdm import tqdm
from typing import Tuple, List

import torch

import gpytorch

from .surrogate import Surrogate


class GaussianProcess(Surrogate):
    """Gaussian process surrogate model"""

    def __init__(self, args: argparse.Namespace, input_dimension: int, device: torch.device = torch.device('cpu')):
        super().__init__(args, input_dimension, device)

    def train(self, train_X: torch.Tensor, train_y: torch.Tensor) -> Tuple[torch.nn.Module, List[float]]:
        # assume zero noise
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=torch.zeros_like(train_y)
        )
        # use exact inference for GP
        model = ExactGPModel(
            train_X=train_X,
            train_y=train_y,
            likelihood=likelihood
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=self._args.lr)

        # marginal log likelihood loss
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        model.train(); likelihood.train()
        losses = []
        for _ in tqdm(range(self._args.num_training_iterations)):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # inference on the entire training dataset
            output = model(train_X)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        return model, losses


class ExactGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_X, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_X, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
        self._input_dimension = train_X.shape[1]

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def energy(self, x, beta=1):
        self.eval()
        self.likelihood.eval()
        y_pred = self.likelihood(self(x))

        return y_pred.mean

    @property
    def input_dimension(self) -> int:
        return self._input_dimension
