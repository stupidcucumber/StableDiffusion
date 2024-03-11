import torch, pathlib
from torchvision import transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class StableDiffusionDataset(Dataset):
    '''
        Accepts a dataframe object with columns: ['image_path', 'prompt']
    '''
    def __init__(self, data: pd.DataFrame, tokenizer) -> None:
        super(StableDiffusionDataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=(768, 768)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            ]
            
        )

    def __len__(self) -> int:
        return len(self.data)
    
    def _tokenize(self, prompt: str) -> np.ndarray:
        return self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.numpy()
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.data.iloc[index]
        image = Image.open(item['image_path'])
        image_tensor = self.transform(image)
        prompt_tensor = self._tokenize(item['prompt'])
        return image_tensor.unsqueeze(0), prompt_tensor
