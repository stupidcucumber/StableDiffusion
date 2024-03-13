import argparse, pathlib
import torch
from src.model import Pipeline
from diffusers import StableDiffusionPipeline


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prompt', type=str, required=True,
                        help='Text prompt to the diffusion model.')
    parser.add_argument('-n', '--number', type=int, default=1,
                        help='Number of examples to generate.')
    parser.add_argument('--config', type=pathlib.Path, default=None,
                        help='Path to the config of the generating model.')
    parser.add_argument('-d', '--device', type=str, default='cpu',
                        help='Device on which model and inputs will be located.')
    parser.add_argument('-o', '--output-dir', type=pathlib.Path, default=pathlib.Path('runs', 'output.png'),
                        help='Output of the model.')
    return parser.parse_args()


def setup_output(dir_path: pathlib.Path) -> None:
    if dir_path.exists():
        raise ValueError('Directory %s already exists!' % str(dir_path))
    else:
        dir_path.mkdir()

if __name__ == '__main__':
    args = parse_arguments()
    setup_output(dir_path=args.output_dir)

    model = Pipeline(config=args.config, device=args.device)
    pipeline = StableDiffusionPipeline(vae=model.vae, text_encoder=model.text_encoder, 
                                       tokenizer=model.tokenizer, unet=model.unet,
                                       scheduler=model.scheduler, safety_checker=None,
                                        feature_extractor=None, requires_safety_checker=False)
    pipeline.to(args.device)
    generator = torch.Generator(args.device).manual_seed(0)
    for index in range(args.number):
        image = pipeline(prompt=args.prompt, generator=generator).images[0]
        print(image.save(args.output_dir.joinpath('generated_image_%d.png' % index)))