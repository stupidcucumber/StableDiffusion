import torch
from torchvision import transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class StableDiffusionDataset(Dataset):
    '''
        Accepts a dataframe object with columns: ['image_path', 'prompt']
    '''
    def __init__(self, class_data: pd.DataFrame, instance_data: pd.DataFrame, tokenizer) -> None:
        super(StableDiffusionDataset, self).__init__()
        self.class_data = class_data
        self.instance_data = instance_data
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
        return len(self.class_data)
    
    def _tokenize(self, prompt: str) -> np.ndarray:
        return self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.numpy()
    
    def _choose_object(self, index: int, _type: str) -> tuple[torch.Tensor, np.ndarray]:
        dataframe = self.class_data if _type == 'class' else self.instance_data
        entry = dataframe.iloc[index % len(dataframe)]
        image_tensor = self.transform(Image.open(entry['image_path']))
        prompt_ids = self._tokenize(entry['prompt'])
        return image_tensor, prompt_ids

    def __getitem__(self, index) -> tuple[torch.Tensor, np.ndarray]:
        _instance = \
              self._choose_object(index=index, _type='instance')
        _class = \
              self._choose_object(index=index, _type='class')
        return *_instance, *_class
