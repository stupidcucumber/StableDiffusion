import torch
from typing import Any


class Decoder:
    def __call__(self, logits: torch.Tensor) -> Any:
        raise NotImplementedError('This method must be implemented in the derived class!')