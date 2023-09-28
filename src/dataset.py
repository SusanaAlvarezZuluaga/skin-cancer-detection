import os

import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image


LABELS = {"akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5, "vasc": 6}

class SkinCancerDataset(Dataset):
    def __init__(
        self, img_anotations, img_dir, transform=None, target_transform=None
    ):  # run once when instantiating the Dataset object.
        super(SkinCancerDataset, self).__init__()
        self.img_labels = img_anotations
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = (
            os.path.join(self.img_dir, self.img_labels.iloc[idx]["image_id"]) + ".jpg"
        )  # get images to path
        image = read_image(img_path).float()

        label = LABELS[self.img_labels.iloc[idx]["dx"]]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class SkinCancerDataModule(pl.LightningDataModule):
    def __init__(self, 
                batch_size, 
                num_workers,
                img_dir,
                labels_dir):
        super().__init__()
        
        # init values
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224))
            ]
        )

    def setup(self, stage: str):
        # split data
        df = pd.read_csv(self.labels_dir)
        test_size = 0.30
        train, test = train_test_split(df, test_size=test_size)

        validation_proportion = 0.30
        train, validation = train_test_split(train, test_size=validation_proportion)

        if stage == "fit":
            self.train_dataset = SkinCancerDataset(
                img_anotations=train, img_dir = self.img_dir
            )

            self.valid_dataset = SkinCancerDataset(
                img_anotations=validation, img_dir = self.img_dir
            )

        if stage == "test":
            self.test_dataset = SkinCancerDataset(
                img_anotations=test, img_dir = self.img_dir
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = True,
        )

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, self.batch_size, num_workers = self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, num_workers = self.num_workers)
