import argparse
import pathlib


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d', '--device', type=str, default='cpu',
                        help='Device on which model and dataset will be placed on.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
