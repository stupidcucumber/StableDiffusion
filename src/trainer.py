import torch
from torch.utils.data import DataLoader
from robustprinter import Printer
from robustprinter.formatter import DefaultFormatter


class Trainer:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim,
                 loss_fn) -> None:
        self.model = model
        self.model.eval()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.rprinter = Printer(formatter=DefaultFormatter(max_columns=1, precision=4))

    def _train_step(self, loss):
        self.model.train()
        self.optimizer.zero_grad()
        loss.backward()
        return loss

    def _val_step(self, loss):
        self.model.eval()
        return loss

    def _epoch_pass(self, epoch: int, dataloader: DataLoader, partition: str = 'train') -> None:
        data = dict()
        data['max_steps'] = len(dataloader)
        data['epoch'] = epoch
        data['partition'] = partition
        for index, (images, prompts) in enumerate(dataloader):
            metrics = dict()
            data['step'] = index
            loss = self.model((images, prompts))
            if partition == 'train':
                loss = self._train_step(loss=loss)
            else:
                loss = self._val_step(loss=loss)
            metrics['loss'] = loss
            metrics.update(data)
            self.rprinter.print(data=data)

    def fit(self, epochs: int, train_loader: DataLoader) -> None:
        print('Start training StableDiffusion...')
        self.rprinter.start()
        for epoch in range(epochs):
            self._epoch_pass(epoch=epoch, dataloader=train_loader, partition='train')
            self.rprinter.break_loop()

        print('Ended training. Saving weights...')
            