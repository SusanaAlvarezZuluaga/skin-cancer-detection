import os

import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image

from torchvision.transforms.functional import to_pil_image
import torch


class SkinCancerDataset(Dataset):
    def __init__(
        self, img_anotations, img_dir, labels, transform=None, target_transform=None
    ):
        super(SkinCancerDataset, self).__init__()
        self.img_labels = img_anotations
        self.img_dir = img_dir
        self.transform = transform
        self.labels = labels
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = (
            os.path.join(self.img_dir, self.img_labels.iloc[idx]["image_id"]) + ".jpg"
        )
        image = read_image(img_path).float()

        label = self.labels[self.img_labels.iloc[idx]["dx"]]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class SkinCancerDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, img_dir, labels_dir, labels):
        super().__init__()

        # init values
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.labels = labels
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomEqualize(),
                transforms.RandomRotation((-180, 180)),  # Image rotate
                transforms.RandomAffine((-180, 180)),  # image translate
                transforms.ToTensor(),
            ]
        )

        self.class_weights = self.calculate_weights()

    def setup(self, stage: str):
        df = pd.read_csv(self.labels_dir)
        test_size = 0.30
        train, test = train_test_split(df, test_size=test_size, stratify=df["dx"])

        validation_proportion = 0.30
        train, validation = train_test_split(
            train, test_size=validation_proportion, stratify=train["dx"]
        )

        if stage == "fit":
            self.train_dataset = SkinCancerDataset(
                img_anotations=train,
                img_dir=self.img_dir,
                labels=self.labels,
                transform=self.transform,
            )

            self.valid_dataset = SkinCancerDataset(
                img_anotations=validation,
                img_dir=self.img_dir,
                labels=self.labels,
                transform=self.transform,
            )

        if stage == "test":
            self.test_dataset = SkinCancerDataset(
                img_anotations=test,
                img_dir=self.img_dir,
                labels=self.labels,
                transform=self.transform,
            )

    def calculate_weights(self):
        test_size = 0.30
        df = pd.read_csv(self.labels_dir)
        train, test = train_test_split(df, test_size=test_size, stratify=df["dx"])

        proportions = train["dx"].value_counts(normalize=True)
        weights = proportions[0] / proportions
        weights_ordered = weights[self.labels.keys()]
        return torch.tensor(weights_ordered, dtype=torch.float32, device="mps")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            self.batch_size,
            num_workers=self.num_workers,
        )
