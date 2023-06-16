import argparse
import ast


def parse_args():
    """ Parses command line arguments.

    Returns:
        argparse.Namespace: Command line arguments.
    """
    parser = argparse.ArgumentParser()

    # Target oracle
    parser.add_argument('--target', type=str, choices=[
        'gaussian-mixture', 'unimodal-gaussian', 'two-moons', # TODO: we could add 'hartmann3', 'hartmann6', 'branin', 'six-hump-camel', 'ackley', 'levy', 'michalewicz', 'rastrigin', 'rosenbrock', 'schwefel', 'styblinski-tang'
    ], required=True)
    # Gaussian mixtures
    parser.add_argument('--means', type=str, default='[[10.0, 10.0], [-10.0, -10.0]]')
    parser.add_argument('--covars', type=str, default='[[[2.0, 0.0], [0.0, 2.0]], [[1.0, 0.0], [0.0, 1.0]]]')
    parser.add_argument('--mixture-probs', type=str, default='[0.5, 0.5]')
    # Unimodal Gaussian
    parser.add_argument('--mean', type=str, default='[6.0, 3.0]')
    parser.add_argument('--covar', type=str, default='[[1.0, 0.0], [0.0, 1.0]]')

    parser.add_argument('--surrogate', type=str, choices=['gaussian-process'], required=True)

    # Training configs
    parser.add_argument('--num-active-learning-iterations', type=int, default=100,)
    parser.add_argument('--num-training-iterations', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)

    # Query strategy
    parser.add_argument('--query-strategy', type=str, choices=['uniform-sampling'], required=True)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--input-ranges', action=ParseKwargs, default={})

    return parser.parse_args()


class ParseKwargs(argparse.Action):
    """ Parses keyword arguments from command line arguments."""

    def __call__(self, parser: argparse.ArgumentParser, args: argparse.Namespace, dict_str, option_string=None):
        try:
            kwargs = ast.literal_eval(dict_str)
        except ValueError:
            raise argparse.ArgumentError(self, f"{self.dest} must be a valid dictionary.")

        setattr(args, self.dest, kwargs)
