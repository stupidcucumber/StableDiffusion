import argparse, pathlib
import torch
from src.model import Pipeline
from diffusers import StableDiffusionPipeline


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
    model = Pipeline(config=args.config, device=args.device)
    pipeline = StableDiffusionPipeline(vae=model.vae, text_encoder=model.text_encoder, 
                                       tokenizer=model.tokenizer, unet=model.unet,
                                       scheduler=model.scheduler, safety_checker=None,
                                        feature_extractor=None, requires_safety_checker=False)
    pipeline.to(args.device)
    generator = torch.Generator(args.device).manual_seed(0)
    image = pipeline(prompt=args.prompt, generator=generator).images[0]
    print(image.save('output.png'))