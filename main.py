import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import wandb

import torch
from torch import nn
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
from typing import Optional, Any

from dataset import SkinCancerDataset
from model import CNN

def main(args):
    dataset = SkinCancerDataset()
    net = CNN()
    callbacks = [ModelCheckpoint(save_top_k=1, mode='max', monitor="valid_acc")]  # save top 1 model
    
    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(project='cnn-mnist-number')        
    
    trainer = pl.Trainer(max_epochs=args.epochs, 
                          callbacks=callbacks, 
                          accelerator=args.accelerator,
                          devices=1,
                          logger = wandb_logger
                        )
    trainer.fit(model=net, datamodule=dataset)
    trainer.test(model=net, datamodule=dataset, ckpt_path='best')

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--accelerator", default="cpu", type=str)
    args = parser.parse_args()

    main(args)
