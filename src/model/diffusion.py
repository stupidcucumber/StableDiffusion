import cv2
import pathlib
from .utils import load_models


class Pipeline:
    def __init__(self, config: pathlib.Path, device: str = 'cpu') -> None:
        models = load_models(config_path=config, device=device)
        self.vae = models['vae']
        self.unet = models['unet']
        self.scheduler = models['scheduler']
        self.tokenizer = models['tokenizer']
        self.text_encoder = models['text_encoder']

    def inference(self, prompt: str) -> cv2.Mat:
        pass