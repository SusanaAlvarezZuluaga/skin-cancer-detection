import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from argparse import ArgumentParser

import wandb

from datasetv2 import SkinCancerDataModule
from dataset import MNISTDataModule
from model import CNN

def main(args):
    datamodule = SkinCancerDataModule()
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
    trainer.fit(model=net, datamodule=datamodule)
    trainer.test(model=net, datamodule=datamodule, ckpt_path='best')

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--accelerator", default="cpu", type=str)
    args = parser.parse_args()

    main(args)
