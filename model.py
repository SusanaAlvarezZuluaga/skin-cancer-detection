import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import torch
from torch import nn
from torchmetrics import Accuracy


class CNN(pl.LightningModule):

    def __init__(
            self,
            cnn_out_channels=None,
            n_lables: int = 10 ## number of labels in de dataset (10 numbers)
    ):
        super().__init__()

        if cnn_out_channels is None:
            cnn_out_channels = [16, 32, 64]
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.valid_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        in_channels = 1 ## the input will have one color chanel (the dataset is black and white )
        # if it were colored images(RGB) it should be set up to three

        cnn_block = list()
        for out_channel in cnn_out_channels:
            cnn_block.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cnn_block.append(nn.ReLU())
            cnn_block.append(nn.MaxPool2d((2, 2))) # reduce dimensionality- we are taking the maximum of each channel
            in_channels = out_channel #for the next iteration

        self.cnn_block = nn.Sequential(*cnn_block) ## output = emmbeding
        self.classifier = nn.Sequential( # does the objetive ej: classify,
            nn.Flatten(),
            nn.Linear(cnn_out_channels[-1] * 3 * 3, n_lables) #n lables = 10
        )

        # save hyper-parameters to self.hparamsm auto-logged by wandb
        self.save_hyperparameters()


    def forward(self, x) -> torch.Tensor:
        x = self.cnn_block(x)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)

        # log metrics to wandb
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute())
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x) ## output of the 
        loss = nn.functional.cross_entropy(logits, y)  ## softmax 
        preds = torch.argmax(logits, dim=1)
        self.valid_acc.update(preds, y)
        self.log("valid_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("valid_acc", self.valid_acc.compute(), prog_bar=True)
        self.valid_acc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
