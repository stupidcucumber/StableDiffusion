import pathlib, torch
import pandas as pd
from diffusers import StableDiffusionPipeline
from PIL import Image


class Generator: 
    def __init__(self, pipeline: StableDiffusionPipeline, output_dir: pathlib.Path,
                 ratio: float, input_dir: pathlib.Path, device: str):
        self.output_class_dir, self.output_instance_dir = self._setup_output_dir(output_dir=output_dir)
        self.output_dir = output_dir
        self.ratio = ratio
        self.input_dir = input_dir
        self.pipeline = pipeline
        self.device = device

    def _setup_output_dir(self, output_dir: pathlib.Path):
        if output_dir.exists():
            raise ValueError('Directory %s already exists!' % str(output_dir))
        output_dir.mkdir()
        output_class_dir_path = output_dir.joinpath('class')
        output_instance_dir_path = output_dir.joinpath('instance')
        output_class_dir_path.mkdir()
        output_instance_dir_path.mkdir()
        return output_class_dir_path, output_instance_dir_path
    
    def _generate_folder(self, prompts: list[str], output: pathlib.Path, 
                         input: list[pathlib.Path] | None = None) -> list[dict]:
        result = []
        for index, prompt in enumerate(prompts):
            if input is None:
                generator = torch.Generator(self.device).manual_seed(index)
                image = self.pipeline(prompt=prompt, generator=generator).images[0]
            else:
                image = Image.open(input[index])
            save_path = output.joinpath('image_%d.png' % index)
            image.save(save_path)
            result.append({
                    'image_path': save_path,
                    'prompt': prompt,
                    'type': 'class' if input is None else 'instance'
            })
        return result
    
    def start(self, instance_prompt: str, class_prompt: str) -> pd.DataFrame:
        instance_images_paths = list(self.input_dir.glob('*'))
        _class_number = int(len(instance_images_paths) * self.ratio)
        entries = []

        instance_entries = self._generate_folder(
            prompts=len(instance_images_paths) * [instance_prompt],
            output=self.output_instance_dir,
            input=instance_images_paths
        )

        class_entries = self._generate_folder(
            prompts=_class_number * [class_prompt],
            output=self.output_class_dir
        )
        entries.extend(instance_entries)
        entries.extend(class_entries)
        dataframe = pd.DataFrame(entries)
        dataframe.to_csv(str(self.output_dir.joinpath('data.csv')), index=False)
        return dataframe