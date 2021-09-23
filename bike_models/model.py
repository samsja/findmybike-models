import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class CaptchaModule(pl.LightningModule):
    def __init__(self, lr: float = 1e-4, pretrained=False):
        super().__init__()
        self.backbone = resnet18(pretrained=pretrained)
        self.fc = nn.Linear(self.backbone.fc.in_features, 1)
        self.backbone.fc = nn.Identity()
        self.lr = lr

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.backbone(x)
        output = 100 * self.fc(embedding).sigmoid()

        return output

    @staticmethod
    def _abs(output, y):
        return torch.abs(output - y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = F.mse_loss(output, y)
        abs_ = CaptchaModule._abs(output, y).mean()

        self.log("train_loss", loss)
        self.log("train_abs", abs_)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = F.mse_loss(output, y)
        abs_ = CaptchaModule._abs(output, y).mean()

        self.log("val_loss", loss)
        self.log("val_abs", abs_)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
