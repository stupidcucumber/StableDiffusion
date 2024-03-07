import torch
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim,
                 loss_fn) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def _train_step(self):
        pass

    def _val_step(self):
        pass

    def _epoch_pass(self, dataloader: DataLoader, partition: str = 'train') -> dict:
        pass

    def fit(self, epochs: int, train_loader: DataLoader, val_loader: DataLoader,
            test_loader: DataLoader | None = None) -> None:
        for epoch in range(epochs):
            metrics = self._epoch_pass(dataloader=train_loader)
            with torch.no_grad():
                metrics = self._epoch_pass(dataloader=val_loader, partition='val')
        if test_loader:
            self._epoch_pass(dataloader=test_loader, partition='test')