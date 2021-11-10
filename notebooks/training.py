# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: bike_models
#     language: python
#     name: bike_models
# ---

# # Train a bike model

# %load_ext autoreload
# %autoreload 2

# ## Import

import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

os.chdir("..")

from vision_transformers.data.utils.utils_plot import imshow
from vision_transformers.transformers.resnet import ResNetModule
from vision_transformers.utils import ProgressiveImageResizing

from bike_models.data import BikeDataModule

# ## Model

## PARAM
batch_size = 12
num_workers = 8
patience = 3
model_path = "data/models/v1"
epochs = 20

data = BikeDataModule("data/dataset_0", batch_size, num_workers)

data.setup()

model = ResNetModule(3, 1e-3)

imshow(data.val_dataset[0][0])

data.train_dataset

imshow(data.train_dataset[0][0])

# + [markdown] tags=[]
# ## Training
# -

increase_image_shape = ProgressiveImageResizing(
    data, epoch_final_size=5, n_step=5, init_size=100, final_size=224
)

callbacks = [
    EarlyStopping(monitor="val_loss", mode="min", patience=patience, strict=False),
    increase_image_shape,
    LearningRateMonitor(logging_interval="step"),
    ModelCheckpoint(
        dirpath=model_path,
        monitor="val_acc",
        save_top_k=3,
        filename="{epoch}-{val_abs:.2f}-{val_loss:.2f}",
        save_last=True,
    ),
]

trainer = pl.Trainer(
    gpus=1,
    max_epochs=epochs,
    check_val_every_n_epoch=1,
    log_every_n_steps=1,
    callbacks=callbacks,
    #  precision=16,
)

trainer.fit(model, data)
