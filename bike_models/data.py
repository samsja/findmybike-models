from typing import Callable, Optional, Tuple

import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
_IMAGE_SHAPE = (244, 244)


def get_transforms(image_shape: Tuple[int, int]) -> Callable:

    all_transform = [
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=_MEAN, std=_STD),
    ]

    return torchvision.transforms.Compose(all_transform)


class BikeDataSet(ImageFolder):
    pass


class BikeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):

        _VAL_PERCENTAGE = 0.1

        dataset = BikeDataSet(self.data_path, get_transforms(_IMAGE_SHAPE))

        val_len = int(_VAL_PERCENTAGE * len(dataset))

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset,
            [
                len(dataset) - val_len,
                val_len,
            ],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
