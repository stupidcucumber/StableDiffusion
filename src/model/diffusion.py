import torch
import numpy as np
import pathlib, itertools
from .utils import (
    load_models,  
    get_target,
    prior_preserving_loss
)


class Pipeline(torch.nn.Module):
    def __init__(self, config: pathlib.Path, device: str = 'cpu') -> None:
        super(Pipeline, self).__init__()
        models = load_models(config_path=config, device=device)
        self.vae = models['vae']
        self.unet = models['unet']
        self.scheduler = models['scheduler']
        self.tokenizer = models['tokenizer']
        self.text_encoder = models['text_encoder']
        self.device = device
        self.train()

    def _tokenize(self, strings: list[str]) -> np.ndarray:
        return self.tokenizer(
            strings,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.numpy()
    
    def _to_latent(self, tensor: torch.Tensor):
        return self.vae.encode(tensor).latent_dist.sample() * 0.18215

    def train(self) -> None:
        self.unet.train()
        self.text_encoder.train()

    def eval(self) -> None:
        self.unet.eval()
        self.text_encoder.eval()

    def parameters(self) -> None:
        return itertools.chain(self.unet.parameters(), self.text_encoder.parameters())
    
    def save(self, output_dir: pathlib.Path) -> None:
        unet_dir = output_dir.joinpath('unet')
        text_encoder_dir = output_dir.joinpath('text_encoder')
        unet_dir.mkdir()
        text_encoder_dir.mkdir()
        self.unet.save_pretrained(str(unet_dir))
        self.text_encoder.save_pretrained(str(text_encoder_dir))

    def forward(self, input: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        '''
            :input param: accepts tuple, where on the index 1 are placed tokenized prompts and
        on the index 0 are placed processed images.
        '''
        prompt_ids = input[1]
        images = input[0].type(dtype=torch.bfloat16)
        encoder_hidden_states = self.text_encoder(prompt_ids)[0]
        latents = self._to_latent(images)

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps, 
            (latents.shape[0], ), 
            device=self.device
        ).long()
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        predicted = self.unet(
            noisy_latents.to(self.device),
            timesteps.to(self.device),
            encoder_hidden_states.to(self.device)
        ).sample
        target = get_target(scheduler=self.scheduler, noise=noise,
                            latents=latents, timesteps=timesteps)
        return prior_preserving_loss(
            model_pred=predicted, target=target, weight=0.5
        )