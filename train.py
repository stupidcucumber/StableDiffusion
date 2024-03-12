import argparse, pathlib
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.model.diffusion import Pipeline
from src.data import StableDiffusionDataset
from src import Trainer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=pathlib.Path, required=True,
                        help='Path to the dataframe containing data for the training. ' +
                        'Data must contain columns: ["image_path", "prompt"]')
    parser.add_argument('-c', '--config', type=pathlib.Path, default=pathlib.Path('configs', 'default.yaml'),
                        help='Configuration file of the model.')
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-d', '--device', type=str, default='cpu',
                        help='Device on which model and dataset will be placed on.')
    parser.add_argument('--batch-size', type=int, default=12,
                        help='Batchsize for the StableDiffusion')
    parser.add_argument('--output', type=pathlib.Path, default='runs/run_0',
                        help='Path to the output folder, where all the weights will be contained.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    
    model = Pipeline(
        config=args.config,
        device=args.device
    )
    model.to(args.device)
    data = pd.read_csv(str(args.data))
    dataset = StableDiffusionDataset(
        class_data=data[data['type'] == 'class'].reset_index(drop=True),
        instance_data=data[data['type'] == 'instance'].reset_index(drop=True),
        tokenizer=model.tokenizer
    )
    dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=args.batch_size)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    trainer = Trainer(
        model=model, 
        optimizer=optimizer,
        output_dir=args.output,
        device=args.device
    )
    trainer.fit(epochs=args.epochs, train_loader=dataloader)