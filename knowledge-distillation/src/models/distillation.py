import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from .mdm.model.v2 import MDMModel
from utils.losses import CriterionPixelWise

class WarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup):
        self.warmup = warmup
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        # lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        else:
            lr_factor = 0.5 ** (epoch - self.warmup) // 25000
        return lr_factor

class DistillationModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-5, weight_decay=0.05):
        super().__init__()
        self.save_hyperparameters()

        self.teacher = MDMModel.from_pretrained()
        self.student = MDMModel.from_pretrained_config()

        self.loss_fn = CriterionPixelWise()

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        lr_scheduler = WarmupScheduler(optimizer=optimizer, warmup=2000)
        return optimizer
    
    def forward(self, color, depth):
        return self.student(color, depth)
    
    def on_train_epoch_start(self):
        self.teacher.eval()
    
    def training_step(self, batch, batch_idx):
        color = batch['color']
        depth = batch['depth']

        pred_s = self(color, depth)
        pred_t = self.teacher(color, depth)

        loss = self.loss_fn(pred_s, pred_t)

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        color = batch['color']
        depth = batch['depth']

        pred_s = self(color, depth)
        pred_t = self.teacher(color, depth)

        loss = self.loss_fn(pred_s, pred_t)

        self.log("validation_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        color = batch['color']
        depth = batch['depth']

        pred_s = self(color, depth)
        pred_t = self.teacher(color, depth)

        loss = self.loss_fn(pred_s, pred_t)

        self.log("test_loss", loss)
        return loss