import torch, pathlib
from torch.utils.data import DataLoader
from robustprinter import Printer
from robustprinter.formatter import DefaultFormatter
from .model import Pipeline


class Trainer:
    def __init__(self, model: Pipeline, optimizer: torch.optim,
                 output_dir: pathlib.Path, device: str = 'cpu') -> None:
        self.model = model
        self.model.eval()
        self.optimizer = optimizer
        self.rprinter = Printer(formatter=DefaultFormatter(max_columns=1, precision=4))
        self.output_dir = output_dir
        self.device = device
        self._setup_output_dir()

    def _setup_output_dir(self) -> None:
        if self.output_dir.exists():
            raise ValueError('%s already exists!' % self.output_dir)
        self.output_dir.mkdir(parents=False)

    def _train_step(self, loss) -> None:
        self.model.train()
        self.optimizer.zero_grad()
        loss.backward()
        return loss

    def _val_step(self, loss):
        self.model.eval()
        return loss
    
    def _move_to_device(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        return [tensor.to(self.device) for tensor in tensors]

    def _epoch_pass(self, epoch: int, dataloader: DataLoader, partition: str = 'train') -> None:
        data = dict()
        data['max_steps'] = len(dataloader)
        data['epoch'] = epoch
        data['partition'] = partition
        data['metrics'] = dict()
        for index, (instance_images, instance_ids, class_images, class_ids) in enumerate(dataloader):
            metrics = dict()
            data['step'] = index
            images = torch.cat([instance_images, class_images])
            prompt_ids = torch.cat([instance_ids, class_ids])
            images, prompt_ids = self._move_to_device(tensors=[images, prompt_ids])
            loss = self.model((images, prompt_ids))
            if partition == 'train':
                loss = self._train_step(loss=loss)
            else:
                loss = self._val_step(loss=loss)
            metrics['loss'] = loss
            data['metrics'].update(metrics)
            self.rprinter.print(data=data)
            torch.cuda.empty_cache()

    def fit(self, epochs: int, train_loader: DataLoader) -> None:
        print('Start training StableDiffusion...')
        self.rprinter.start()
        for epoch in range(epochs):
            self._epoch_pass(epoch=epoch, dataloader=train_loader, partition='train')
            self.rprinter.break_loop()
            self.model.save(output_dir=self.output_dir)
        print('Ended training. Saving weights...')
            