import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import torch
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from typing import Optional, Any


class SkinCancerDataset(pl.LightningDataModule):

    def __init__(self, data_path: str = './'):
        super().__init__()
        self.data_path = data_path
        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )

    def prepare_data(self) -> None:
        MNIST(root=self.data_path, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        mnist_all = MNIST(
            root=self.data_path,
            train=True,
            transform=self.transform,
            download=False
        )

        self.train, self.val = random_split(
            mnist_all, [55_000, 5_000], generator=torch.Generator().manual_seed(1)
        )

        self.test = MNIST(
            root=self.data_path,
            train=False,
            transform=self.transform,
            download=False
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=64, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=64, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.train, batch_size=64, num_workers=4)

