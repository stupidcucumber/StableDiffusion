from typing import Any
from torch import Tensor
from .base import Decoder


class StableDiffusionDecoder(Decoder):
    def __init__(self):
        super(StableDiffusionDecoder, self).__init__()

    def __call__(self, logits: Tensor) -> Any:
        return super().__call__(logits)