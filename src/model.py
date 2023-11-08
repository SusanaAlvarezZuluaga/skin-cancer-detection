import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy
from torchvision.models import (
    alexnet,
    AlexNet_Weights,
    resnet18,
    ResNet18_Weights,
    resnet50,
    ResNet50_Weights,
)
from torchmetrics.classification import MulticlassF1Score
import wandb
from torchmetrics.classification import MulticlassConfusionMatrix
import matplotlib.pyplot as plt


class CNN(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        lr,
        momentum,
        n_lables,
        class_weights,
        params_freeze_fraction,
        weighted_sampling,
        adaptative_loss,
        weight_decay,
        optimizer,
        cnn_out_channels=None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.lr = lr
        self.momentum = momentum
        self.n_labels = n_lables
        self.params_freeze_fraction = params_freeze_fraction
        self.cnn_out_channels = cnn_out_channels
        self.class_weights = None
        self.weight_decay = weight_decay
        self.adaptative_loss = adaptative_loss
        self.optimizer = optimizer

        if weighted_sampling == False:
            self.class_weights = class_weights

        if self.cnn_out_channels is None:
            self.cnn_out_channels = [16, 32, 64]

        # initialize metrics

        regular_metric_kwargs = {
            "task": "multiclass",
            "num_classes": n_lables,
        }
        self.train_acc = Accuracy(**regular_metric_kwargs)
        self.valid_acc = Accuracy(**regular_metric_kwargs)
        self.test_acc = Accuracy(**regular_metric_kwargs)

        multi_class_metric_kwargs = {"num_classes": n_lables, "average": None}
        self.train_f1 = MulticlassF1Score(**multi_class_metric_kwargs)
        self.valid_f1 = MulticlassF1Score(**multi_class_metric_kwargs)
        self.test_f1 = MulticlassF1Score(**multi_class_metric_kwargs)

        multi_class_weighted_metric_kwargs = {
            "num_classes": n_lables,
            "average": "weighted",
        }
        self.train_f1_weighted = MulticlassF1Score(**multi_class_weighted_metric_kwargs)
        self.valid_f1_weighted = MulticlassF1Score(**multi_class_weighted_metric_kwargs)
        self.test_f1_weighted = MulticlassF1Score(**multi_class_weighted_metric_kwargs)

        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=self.n_labels)

        self.test_outputs = {"y": [], "preds": []}

        # self.model = self.load_personal_model()
        self.model = self.load_resnet()

        self.save_hyperparameters()

    def load_personal_model(self):
        in_channels = self.in_channels
        cnn_block = list()

        for out_channel in self.cnn_out_channels:
            conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            cnn_block.append(conv_layer)
            cnn_block.append(nn.ReLU())
            cnn_block.append(nn.MaxPool2d((2, 2)))
            in_channels = out_channel

        cnn_block = nn.Sequential(*cnn_block)
        classifier = nn.Sequential(nn.Flatten(), nn.Linear(268800, self.n_labels))

        model = nn.Sequential(cnn_block, classifier)
        return model

    def load_cengil_2021(self):
        # Load a pretrained AlexNet model from torchvision
        model = alexnet(weights=AlexNet_Weights.DEFAULT)

        # Remove the last fully connected layer (classification layer)
        feature_extractor = model.features

        # Create an SVM classifier as the new classification layer
        svm_classifier = nn.Sequential(nn.Flatten(), nn.Linear(56576, self.n_labels))

        # Combine the feature extractor and SVM classifier
        model = nn.Sequential(feature_extractor, svm_classifier)

        return model

    def load_resnet(self):
        # Load a pre-trained ResNet model from torchvision
        model = resnet18(ResNet18_Weights.DEFAULT)

        # Calculate the total number of parameters in the model
        total_params = sum(p.numel() for p in model.parameters())

        # Calculate the number of parameters to freeze (70% of total_params)
        params_to_freeze = int(self.params_freeze_fraction * total_params)

        if params_to_freeze > 0:
            # Freeze the first 70% of parameters
            frozen_params = 0
            for param in model.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
                if frozen_params >= params_to_freeze:
                    break

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.n_labels)
        return model

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def on_train_epoch_start(self):
        """Resets the metrics at the start of every epoch."""
        self.train_f1.reset()
        self.train_f1_weighted.reset()
        self.train_acc.reset()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        new_class_weights = self.class_weights

        if self.adaptative_loss == True and self.global_step > 0:
            new_class_weights = self.get_adaptative_class_weights()

        loss = nn.functional.cross_entropy(logits, y, new_class_weights)

        return {"loss": loss, "preds": preds, "y": y}

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

        self.train_acc.update(outputs["preds"], outputs["y"])
        self.train_f1.update(outputs["preds"], outputs["y"])
        self.train_f1_weighted.update(outputs["preds"], outputs["y"])

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
            "train_f1_weighted",
            self.train_f1_weighted.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        f1 = self.train_f1.compute()

        for i in range(f1.size()[0]):
            self.log(
                f"train_f1_class_{i}",
                f1[i].item(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def on_validation_epoch_start(self):
        """Resets the metrics at the start of every epoch."""
        self.valid_f1.reset()
        self.valid_f1_weighted.reset()
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
        self.valid_f1_weighted.update(outputs["preds"], outputs["y"])

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
            "valid_f1_weighted",
            self.valid_f1_weighted.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        f1 = self.valid_f1.compute()

        for i in range(f1.size()[0]):
            self.log(
                f"valid_f1_class_{i}",
                f1[i].item(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def on_test_epoch_start(self):
        """Resets the metrics at the start of every epoch."""
        self.test_acc.reset()
        self.test_f1.reset()
        self.test_f1_weighted.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y, self.class_weights)
        preds = torch.argmax(logits, dim=1)

        self.test_outputs["loss"] = loss
        self.test_outputs["y"].extend(y.cpu().numpy())
        self.test_outputs["preds"].extend(preds.cpu().numpy())
        return {"loss": loss, "preds": preds, "y": y}

    def on_test_batch_end(self, outputs, batch, batch_idx):
        # Update metrics
        self.test_acc.update(outputs["preds"], outputs["y"])
        self.test_f1.update(outputs["preds"], outputs["y"])
        self.confusion_matrix.update(outputs["preds"], outputs["y"])

    def on_test_epoch_end(self):
        class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

        # Create a Matplotlib figure
        cm_fig, ax = plt.subplots(figsize=(8, 8))
        # Plot the confusion matrix on the figure
        self.confusion_matrix.plot(ax=ax)

        # Save the plot as an image
        cm_fig.savefig("confusion_matrix.png")

        # Log the image with wandb
        wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})

        # Close the Matplotlib figure to avoid memory leaks
        plt.close(cm_fig)

        """Computes metrics at the end of every epoch and logs it."""
        self.log(
            "test_acc",
            self.test_acc.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "test_f1_weighted",
            self.test_f1_weighted.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        f1 = self.test_f1.compute()

        for i in range(f1.size()[0]):
            self.log(
                f"test_f1_class_{i}",
                f1[i].item(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def configure_optimizers(self):
        if self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )

        else:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        return optimizer

    def get_adaptative_class_weights(self):
        f1_scores = self.valid_f1.compute() + 0.1

        max_f1_score = torch.max(f1_scores).item()

        f1_scores_normalized = f1_scores / max_f1_score

        if self.class_weights != None:
            new_class_weights = self.class_weights / f1_scores_normalized
            new_class_weights[new_class_weights == float("inf")] = 0
            max_class_weight = torch.max(new_class_weights).item()
            return new_class_weights / max_class_weight
        else:
            return f1_scores_normalized
