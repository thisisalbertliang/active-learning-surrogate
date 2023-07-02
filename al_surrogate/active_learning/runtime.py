import argparse
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .query_strategy import QueryStrategy
from .surrogate import Surrogate
from al_surrogate.utils.logging import log_scalars
from al_surrogate.density import plot_density


def run_active_learning(
    args: argparse.Namespace,
    surrogate: Surrogate, target: nn.Module,
    query_strategy: QueryStrategy,
    tb_writer: SummaryWriter,
    device: torch.device = torch.device('cpu'),
) -> Tuple[Surrogate, torch.Tensor, torch.Tensor]:
    """ Runs the population-based active learning loop.

    Args:
        args (argparse.Namespace): Command line arguments.
        surrogate (nn.Module): Surrogate model.
        target (nn.Module): Target model.
        query_strategy (QueryStrategy): Query strategy.

    Returns:
        Tuple[nn.Module, torch.Tensor, torch.Tensor]: Final surrogate model, training samples, training labels.
    """
    target.requires_grad_(False) # don't track gradients for target oracle

    # initialize the dataset
    train_X = query_strategy.sample(args.batch_size)
    train_y = target(train_X)

    # initialize the surrogate model
    model, losses = surrogate.train(train_X, train_y); log_scalars(tb_writer, 'train_losses_al-loop-0', losses)

    # helper function for plotting the surrogate density and training samples
    def plot_surrogate_density_and_samples(iteration: int):
        fig, _ = plot_density(
            args=args, density=model, plot_energy=True,
            scatter_points=train_X, scatter_point_label='training samples',
            device=device,
        )
        if fig is not None:
            tb_writer.add_figure(
                tag='surrogate_density', figure=fig, global_step=iteration, close=True,
            )

    # plot initial surrogate density and training samples
    plot_surrogate_density_and_samples(iteration=0)

    for it in range(args.num_active_learning_iterations):
        # Query new labeled samples
        query_X = query_strategy.sample(args.batch_size)
        query_y = target(query_X)
        # Update training set
        train_X = torch.cat([train_X, query_X], dim=0)
        train_y = torch.cat([train_y, query_y], dim=0)

        # Re-train surrogate from scratch
        model, losses = surrogate.train(train_X, train_y); log_scalars(tb_writer, f"train_losses_al-loop-{it+1}", losses)

        # Plot current surrogate density and training samples
        plot_surrogate_density_and_samples(iteration=it+1)

    return model, train_X, train_y
