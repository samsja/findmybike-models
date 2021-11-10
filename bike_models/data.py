import copy
from typing import Callable, Optional, Tuple

import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from vision_transformers.data.utils.augmentation import (
    get_transforms,
    get_transforms_val,
)

_IMAGE_SHAPE = (244, 244)


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

        dataset_original = BikeDataSet(self.data_path, get_transforms(_IMAGE_SHAPE))

        dataset_for_val = copy.deepcopy(dataset_original)

        val_len = int(_VAL_PERCENTAGE * len(dataset_original))

        def get_split(dataset):
            return torch.utils.data.random_split(
                dataset,
                [
                    len(dataset) - val_len,
                    val_len,
                ],
                generator=torch.Generator().manual_seed(42),
            )

        self.train_dataset, _ = get_split(dataset_original)
        _, self.val_dataset = get_split(dataset_for_val)

        dataset_for_val.transform = get_transforms_val(_IMAGE_SHAPE)

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
