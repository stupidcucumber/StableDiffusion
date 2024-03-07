import torch, pathlib
import pandas as pd
from torch.utils.data import Dataset
from ..model.utils import load_model


class StableDiffusionDataset(Dataset):
    def __init__(self, data: pd.DataFrame, text_encoder) -> None:
        super(StableDiffusionDataset, self).__init__()
        self.data = data
        self.text_encoder = text_encoder

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> tuple:
        entry = self.data.iloc[index]
