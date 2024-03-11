import cv2, torch
import numpy as np
import pathlib
from .utils import load_models, generate_gaussian_noise


class Pipeline:
    def __init__(self, config: pathlib.Path, device: str = 'cpu') -> None:
        models = load_models(config_path=config, device=device)
        self.vae = models['vae']
        self.unet = models['unet']
        self.scheduler = models['scheduler']
        self.tokenizer = models['tokenizer']
        self.text_encoder = models['text_encoder']
        self.device = device

    def _tokenize(self, string: str) -> np.ndarray:
        return self.tokenizer(
            string,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.numpy()
    
    def _to_latent(self, image: torch.Tensor):
        return self.vae

    def inference(self, prompt: str) -> cv2.Mat:
        token_ids = self._tokenize(string=prompt)
        text_embeddings = self.text_encoder(token_ids)[0]

        image = generate_gaussian_noise(shape=(3, 512, 512), 
                                        device=self.device,
                                        generator=torch.Generator(device=self.device).manual_seed(0))
        