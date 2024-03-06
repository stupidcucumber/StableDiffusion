import argparse, pathlib
from src.model import StableDiffusion
from src.decoder import StableDiffusionDecoder


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prompt', type=str, required=True,
                        help='Text prompt to the diffusion model.')
    parser.add_argument('-n', type=int, default=1,
                        help='Number of examples to generate.')
    parser.add_argument('--weights', type=pathlib.Path, default=None,
                        help='Path to the weights of the generating model.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    model = StableDiffusion()