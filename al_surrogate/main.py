import argparse

import torch

from al_surrogate.active_learning.query_strategy import create_query_strategy
from al_surrogate.active_learning.surrogate import create_surrogate
from al_surrogate.active_learning.runtime import run_active_learning
from al_surrogate.utils.logging import create_tensorboard_writer
from al_surrogate.arguments import parse_args
from al_surrogate.density import create_density, plot_density


def main(args: argparse.Namespace):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tb_writer = create_tensorboard_writer(args)

    target = create_density(args, device)

    # Plot the target density
    target_fig, _ = plot_density(
        args=args, density=target, plot_energy=True, device=device,
    )
    if target_fig is not None:
        print(f"Succesfully plotted target density")
        tb_writer.add_figure(
            tag='target_density', figure=target_fig, global_step=0, close=True,
        )

    surrogate = create_surrogate(args, target.input_dimension, device)
    query_strategy = create_query_strategy(
        args=args, surrogate=surrogate, target=target, device=device,
    )

    train_X, _ = run_active_learning(
        args=args, surrogate=surrogate,
        query_strategy=query_strategy,
        tb_writer=tb_writer,
        device=device,
    )

    # density plot of the final surrogate and training samples
    surrogate_and_samples_fig, _ = plot_density(
        args=args, density=surrogate, plot_energy=True,
        scatter_points=train_X, scatter_point_label='training samples',
        device=device,
    )
    if surrogate_and_samples_fig is not None:
        print(f"Successfully plotted final surrogate and training samples")
        tb_writer.add_figure(
            tag='final_surrogate', figure=surrogate_and_samples_fig,
            global_step=args.num_active_learning_iterations, close=True,
        )

    tb_writer.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
