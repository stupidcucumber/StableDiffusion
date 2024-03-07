import argparse, pathlib
from src.model import Pipeline


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prompt', type=str, required=True,
                        help='Text prompt to the diffusion model.')
    parser.add_argument('-n', type=int, default=1,
                        help='Number of examples to generate.')
    parser.add_argument('--config', type=pathlib.Path, default=None,
                        help='Path to the config of the generating model.')
    parser.add_argument('-d', '--device', type=str, default='cpu',
                        help='Device on which model and inputs will be located.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    models = Pipeline(config_path=args.config, device=args.device)