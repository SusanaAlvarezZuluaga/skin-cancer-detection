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
    def __init__(self, img_anotations, img_dir, labels, transform=None):
        super(SkinCancerDataset, self).__init__()
        self.img_labels = img_anotations
        self.img_dir = img_dir
        self.transform = transform
        self.labels = labels

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
        return image, label


class SkinCancerDataModule(pl.LightningDataModule):
    def __init__(
        self, batch_size, num_workers, img_dir, labels_dir, labels, weighted_sampling
    ):
        super().__init__()

        # init values
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.labels = labels
        self.weighted_sampling = weighted_sampling
        self.train_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomRotation((-180, 180)),  # Image rotate
                transforms.RandomAffine((-180, 180)),  # image translate
                transforms.ToTensor(),
            ]
        )

        self.valid_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        self.train_data, self.validation_data, self.test_data = self.get_data_sets()

        weights = self.calculate_weights()

        self.class_weights = torch.tensor(weights, dtype=torch.float32, device="mps")

        self.trainSampler = None

        if self.weighted_sampling == True:
            class_weights = [weights[label] for label in self.train_data["dx"]]
            self.trainSampler = torch.utils.data.WeightedRandomSampler(
                weights=class_weights,
                num_samples=len(self.train_data),
                replacement=True,
            )

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = SkinCancerDataset(
                img_anotations=self.train_data,
                img_dir=self.img_dir,
                labels=self.labels,
                transform=self.train_transform,
            )

            self.valid_dataset = SkinCancerDataset(
                img_anotations=self.validation_data,
                img_dir=self.img_dir,
                labels=self.labels,
                transform=self.train_transform,
            )

        if stage == "test":
            self.test_dataset = SkinCancerDataset(
                img_anotations=self.test_data,
                img_dir=self.img_dir,
                labels=self.labels,
                transform=self.train_transform,
            )

    def get_data_sets(self):
        df = pd.read_csv(self.labels_dir)
        test_size = 0.30
        train, test = train_test_split(df, test_size=test_size, stratify=df["dx"])

        validation_proportion = 0.30
        train, validation = train_test_split(
            train, test_size=validation_proportion, stratify=train["dx"]
        )

        return train, validation, test

    def calculate_weights(self):
        proportions = self.train_data["dx"].value_counts(normalize=True)
        weights = proportions[0] / proportions
        weights_ordered = weights[self.labels.keys()]

        return weights_ordered

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True if self.trainSampler is None else False,
            sampler=self.trainSampler,
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
