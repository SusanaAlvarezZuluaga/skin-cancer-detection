from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset import MNISTDataModule
from datasetv2 import SkinCancerDataModule
from model import CNN


def main(args):
    datamodule = SkinCancerDataModule()
    net = CNN()

    callbacks = [
        ModelCheckpoint(save_top_k=1, mode="max", monitor="valid_acc")
    ]  # save top 1 model

    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(project="cnn-mnist-number")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        accelerator="mps" if torch.backends.mps.is_available() else "cpu",
        devices=-1,
        logger=wandb_logger,
    )
    trainer.fit(model=net, datamodule=datamodule)
    trainer.test(model=net, datamodule=datamodule, ckpt_path="best")

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--epochs", default=1, type=int)
    args = parser.parse_args()

    main(args)
