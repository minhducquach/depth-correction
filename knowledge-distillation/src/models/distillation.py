import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from .mdm.model.v2 import MDMModel
from utils.losses import CriterionPixelWise

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
        optimizer = torch.optim.AdamW(
            self.student.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )

        def lr_lambda(current_step):
            if current_step <= 2000:
                return float(current_step) / 2000.0
            else:
                decay_steps = (current_step - 2000) // 25000
                return 0.5 ** decay_steps

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step" # Updates every batch iteration
            }
        }
    
    def forward(self, image, depth, num_tokens=1200):
        return self.student(image=image, depth=depth, num_tokens=num_tokens)
    
    def on_train_epoch_start(self):
        self.teacher.eval()
    
    def training_step(self, batch, batch_idx):
        color = batch['color']
        depth = batch['depth']

        pred_s = self(image=color, depth=depth)
        with torch.no_grad():
            pred_t = self.teacher(image=color, depth=depth, num_tokens=1200)

        loss = self.loss_fn(pred_s, pred_t)

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        color = batch['color']
        depth = batch['depth']

        pred_s = self(image=color, depth=depth)
        with torch.no_grad():
            pred_t = self.teacher(image=color, depth=depth, num_tokens=1200)

        loss = self.loss_fn(pred_s, pred_t)

        self.log("validation_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        color = batch['color']
        depth = batch['depth']

        pred_s = self(image=color, depth=depth)
        with torch.no_grad():
            pred_t = self.teacher(image=color, depth=depth, num_tokens=1200)

        loss = self.loss_fn(pred_s, pred_t)

        self.log("test_loss", loss)
        return loss