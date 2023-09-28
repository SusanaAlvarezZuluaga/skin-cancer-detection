import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy, F1Score,ConfusionMatrix

class CNN(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        lr,
        momentum,
        n_lables,
        cnn_out_channels= None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.lr = lr
        self.momentum = momentum
        self.n_labels = n_lables

        if cnn_out_channels is None:
            cnn_out_channels = [16, 32, 64]

        regular_metric_kwargs = {
            "task": "multiclass",
            "num_classes": n_lables,
        }

    
        # Calcualte metrics
        self.train_acc = Accuracy(**regular_metric_kwargs)
        self.valid_acc = Accuracy(**regular_metric_kwargs)
        self.test_acc = Accuracy(**regular_metric_kwargs)


        self.train_f1 = F1Score(**regular_metric_kwargs)
        self.valid_f1 = F1Score(**regular_metric_kwargs)
        self.test_f1 = F1Score(**regular_metric_kwargs)

        #self.test_confusion_matrix = ConfusionMatrix(num_classes=self.n_labels)


        in_channels = self.in_channels  ## the input will have one color chanel (the dataset is black and white )
        # if it were colored images(RGB) it should be set up to three
        cnn_block = list()
        for out_channel in cnn_out_channels:
            conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            cnn_block.append(conv_layer)
            cnn_block.append(nn.ReLU())
            cnn_block.append(
                nn.MaxPool2d((2, 2))
            )  # reduce dimensionality- we are taking the maximum of each channel
            in_channels = out_channel  # for the next iteration

        self.cnn_block = nn.Sequential(*cnn_block)  ## output = emmbeding
        self.classifier = nn.Sequential(  # does the objetive ej: classify,
            nn.Flatten(), nn.Linear(268800, self.n_labels)  # n lables = 10
        )

        # save hyper-parameters to self.hparamsm auto-logged by wandb
        self.save_hyperparameters()

    def forward(self, x) -> torch.Tensor:
        x = self.cnn_block(x)
        return self.classifier(x)
    
    def on_train_epoch_start(self):
        """Resets the metrics at the start of every epoch."""
        self.train_f1.reset()
        self.train_acc.reset()
   
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        return {
            "loss": loss,
            "preds": preds,
            "y": y 
        }
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Logs training loss after every batch and update metrics"""
        self.log(
            "train_loss",
            outputs["loss"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        # Update metrics
        self.train_acc.update(outputs["preds"], outputs["y"])
        self.train_f1.update(outputs["preds"], outputs["y"])
 
    def on_train_epoch_end(self):
        """Computes the metrics at the end of every epoch and logs it."""
        self.log(
            "train_acc",
            self.train_acc.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
         
        self.log(
            "train_f1",
            self.train_f1.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )



    def on_validation_epoch_start(self):
        """Resets the metrics at the start of every epoch."""
        self.valid_f1.reset()
        self.valid_acc.reset()
    
    def validation_step(self, batch, batch_idx):
        """Mimics training step."""
        return self.training_step(batch, batch_idx)

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        """Logs training loss after every batch and update metrics"""
        self.log(
            "valid_loss",
            outputs["loss"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        # Update metrics
        self.valid_acc.update(outputs["preds"], outputs["y"])
        self.valid_f1.update(outputs["preds"], outputs["y"])

    def on_validation_epoch_end(self):
        """Computes the metrics at the end of every epoch and logs it."""
        self.log(
            "valid_acc",
            self.valid_acc.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
         
        self.log(
            "valid_f1",
            self.valid_f1.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )


    def on_test_epoch_start(self):
        """Resets the metrics at the start of every epoch."""
        self.test_acc.reset()
        self.test_f1.reset()
   

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
    
        return {
            "loss": loss,
            "preds": preds,
            "y": y 
        }
 
    def on_test_batch_end(self, outputs, batch, batch_idx):
        # Update metrics
        self.test_acc.update(outputs["preds"], outputs["y"])
        self.test_f1.update(outputs["preds"], outputs["y"])

    def on_test_epoch_end(self):
        """Computes metrics at the end of every epoch and logs it."""
        self.log(
            "test_f1",
            self.test_f1.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_acc",
            self.test_acc.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        return optimizer
