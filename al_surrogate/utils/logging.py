import os
from time import gmtime, strftime
import pprint
from typing import List

from torch.utils.tensorboard import SummaryWriter


def get_experiment_name(args):
    name = [
        args.surrogate,
        f"target={args.target}",
        f"query-strategy={args.query_strategy}",
    ] if args.experiment_name is None else [
        args.experiment_name,
    ]

    name.append(strftime('%Y-%m-%d-%H-%M-%S', gmtime()))

    return '_'.join(name)


def create_tensorboard_writer(args):
    tb_writer = SummaryWriter(
        log_dir=os.path.join('output', 'runs', get_experiment_name(args)),
        flush_secs=10,
    )
    args_str = pprint.pformat(vars(args))
    print(f"args: {args_str}")
    tb_writer.add_text('args', args_str)
    print(f"Tensorboard logs will be saved to {tb_writer.log_dir}")
    return tb_writer


def log_scalars(tb_writer: SummaryWriter, tag: str, scalars: List[float]):
    for it, scalar in enumerate(scalars):
        tb_writer.add_scalar(tag=tag, scalar_value=scalar, global_step=it)
