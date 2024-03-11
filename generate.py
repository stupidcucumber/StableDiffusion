import pathlib, argparse
from src.model import Pipeline
from diffusers import StableDiffusionPipeline
from src.generator import Generator


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=pathlib.Path, required=True,
                        help='Path to the directory containing instance images.')
    parser.add_argument('-c', '--config', type=pathlib.Path, default=pathlib.Path('configs', 'default.yaml'),
                        help='Config file of the model pipeline.')
    parser.add_argument('--instance-prompt', type=str, required=True,
                        help='Prompt to the images of your instance.')
    parser.add_argument('--class-prompt', type=str, required=True,
                        help='Prompt from which to generate images of the class instances.')
    parser.add_argument('--ratio', type=float, default=100,
                        help='Ratio between class images and instance images. As a default it will,' +
                        ' generate 10 class images per one instance image.')
    parser.add_argument('--output-dir', type=pathlib.Path, default=pathlib.Path('output'),
                        help='Path to the folder where dataset will be stored. Will be created if not exist.')
    parser.add_argument('-d', '--device', type=str, default='cpu',
                        help='Device on which inputs and model itself will be stored.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    
    model = Pipeline(config=args.config, device=args.device)
    pipeline = StableDiffusionPipeline(vae=model.vae, text_encoder=model.text_encoder, 
                                       tokenizer=model.tokenizer, unet=model.unet,
                                       scheduler=model.scheduler, safety_checker=None,
                                        feature_extractor=None, requires_safety_checker=False)
    pipeline.to(args.device)

    generator = Generator(
        pipeline=pipeline,
        output_dir=args.output_dir,
        ratio=args.ratio,
        input_dir=args.input_dir,
        device=args.device
    )
    generator.start(
        instance_prompt=args.instance_prompt,
        class_prompt=args.class_prompt
    )

