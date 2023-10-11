from argparse import ArgumentParser
import os
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset import SkinCancerDataModule
from model import CNN

import torch.multiprocessing


IN_CHANNELS = 3  # colored images
NUM_CLASSES = 7

IMG_DIR = "../data/HAM10000_images_part_1/"
LABELS_DIR = "./data/HAM10000_metadata.csv"

LABELS = {"akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5, "vasc": 6}


def main(args):
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    pl.seed_everything(42)
    torch.multiprocessing.set_sharing_strategy("file_system")

    datamodule = SkinCancerDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_dir=IMG_DIR,
        labels_dir=LABELS_DIR,
        labels=LABELS,
    )

    class_weights = datamodule.class_weights

    net = CNN(
        lr=args.lr,
        momentum=args.lr,
        n_lables=NUM_CLASSES,
        in_channels=IN_CHANNELS,
        class_weights=class_weights,
    )

    callbacks = [ModelCheckpoint(save_top_k=1, mode="max", monitor="valid_acc")]

    wandb_logger = WandbLogger(project="skin_cancer_dataset_trial")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        accelerator="mps" if torch.backends.mps.is_available() else "cpu",
        devices=-1,
        logger=wandb_logger,
        log_every_n_steps=1,
    )
    trainer.fit(model=net, datamodule=datamodule)
    trainer.test(model=net, datamodule=datamodule, ckpt_path="best")

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)

    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--batch_size", default=64, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    args = parser.parse_args()

    main(args)
