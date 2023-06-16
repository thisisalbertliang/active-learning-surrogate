import argparse

import torch

from al_surrogate.active_learning.query_strategy import create_query_strategy
from al_surrogate.active_learning.surrogate import create_surrogate
from al_surrogate.active_learning.runtime import run_active_learning
from al_surrogate.utils.logging import create_tensorboard_writer
from al_surrogate.arguments import parse_args
from al_surrogate.density import create_density, plot_2d_density, plot_1d_density


def main(args: argparse.Namespace):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tb_writer = create_tensorboard_writer(args)

    target = create_density(args, device)

    target_fig = None
    if target.input_dimension == 2:
        assert 'x' in args.input_ranges and 'y' in args.input_ranges, \
            'For 2D densities, the input ranges must be specified as "x=[x_min, x_max], y=[y_min, y_max]"'
        target_fig, _ = plot_2d_density(
            density=target, plot_energy=True,
            xlim=args.input_ranges['x'], ylim=args.input_ranges['y'],
            device=device,
        )
    elif target.input_dimension == 1:
        assert 'x' in args.input_ranges, \
            'For 1D densities, the input ranges must be specified as "x=[x_min, x_max]"'
        target_fig, _ = plot_1d_density(
            density=target, plot_energy=True,
            xlim=args.input_ranges['x'],
            device=device,
        )

    if target_fig is not None:
        tb_writer.add_figure(
            tag='target_density', figure=target_fig, global_step=0, close=True,
        )

    surrogate = create_surrogate(args, target.input_dimension, device)
    query_strategy = create_query_strategy(args)

    model, train_X, _ = run_active_learning(
        args=args, surrogate=surrogate, target=target,
        query_strategy=query_strategy,
        tb_writer=tb_writer,
    )

    # density plot of the final surrogate and training samples
    fig = None
    if target.input_dimension == 2:
        fig, _ = plot_2d_density(
            density=model, plot_energy=True,
            xlim=args.input_ranges['x'], ylim=args.input_ranges['y'],
            scatter_points=train_X, scatter_point_label='training samples',
            device=device,
        )
    elif target.input_dimension == 1:
        fig, _ = plot_1d_density(
            density=model, plot_energy=True,
            xlim=args.input_ranges['x'],
            scatter_points=train_X, scatter_point_label='training samples',
            device=device,
        )

    if fig is not None:
        tb_writer.add_figure(
            tag='final_surrogate', figure=fig, global_step=args.num_active_learning_iterations, close=True,
        )

    tb_writer.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
