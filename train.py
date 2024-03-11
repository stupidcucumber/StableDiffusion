import argparse, pathlib, yaml
import torch
from src.model.diffusion import Pipeline


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-c', '--config', type=pathlib.Path, default=pathlib.Path('configs', 'default.yaml'),
                        help='Configuration file of the model.')
    parser.add_argument('-d', '--device', type=str, default='cpu',
                        help='Device on which model and dataset will be placed on.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    
    model = Pipeline(
        config=args.config,
        device=args.device
    )

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    