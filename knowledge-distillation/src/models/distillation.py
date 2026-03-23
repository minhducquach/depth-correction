import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from .mdm.model.v2 import MDMModel
from utils.losses import Criterion
from utils.metrics import compute_metrics

from prettytable import PrettyTable

class DistillationModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-5, weight_decay=0.05):
        super().__init__()
        self.save_hyperparameters()

        self.teacher = MDMModel.from_pretrained()
        # self.teacher.compile(mode='default')

        self.student = MDMModel.from_pretrained_config()
        # self.student.compile(mode='default')

        self.loss_fn = Criterion()
        # self.loss_fn.compile(mode='default')

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
                "interval": "step" 
            }
        }
    
    def forward(self, image, depth, num_tokens):
        return self.student(image=image, depth=depth, num_tokens=num_tokens)
    
    def on_train_epoch_start(self):
        self.teacher.eval()

    def extract_and_mask_depth(self, pred_dict, apply_mask=True):
        """Extracts depth/mask from the model dict and applies invalid pixel masking."""
        depth_reg = pred_dict.get('depth_reg', None)
        mask = pred_dict.get('mask', None)

        if depth_reg is None:
            return None

        depth_reg = depth_reg.float()
        
        if mask is not None:
            mask = mask.float()
            mask_binary = mask > 0.5
        else:
            mask_binary = None

        depth = depth_reg

        if apply_mask and mask_binary is not None:
            depth = torch.where(mask_binary, depth, torch.inf)
            
        return depth

    def get_num_tokens(self):
        min_tokens, max_tokens = 1200, 3600
        resolution_level = 3
        return int(min_tokens + (resolution_level / 9.0) * (max_tokens - min_tokens))
    
    def training_step(self, batch, batch_idx):
        color = batch['color']
        depth_in = batch['depth']
        num_tokens = self.get_num_tokens()

        raw_pred_s = self(image=color, depth=depth_in, num_tokens=num_tokens)
        depth_s = self.extract_and_mask_depth(raw_pred_s, apply_mask=True)

        with torch.no_grad():
            raw_pred_t = self.teacher(image=color, depth=depth_in, num_tokens=num_tokens)
            depth_t = self.extract_and_mask_depth(raw_pred_t, apply_mask=True)

        loss = self.loss_fn(depth_s, depth_t)
        # print('Train loss:', loss)

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        color = batch['color']
        depth_in = batch['depth']
        num_tokens = self.get_num_tokens()

        # Student
        raw_pred_s = self(image=color, depth=depth_in, num_tokens=num_tokens)
        depth_s = self.extract_and_mask_depth(raw_pred_s, apply_mask=True)

        # Teacher
        with torch.no_grad():
            raw_pred_t = self.teacher(image=color, depth=depth_in, num_tokens=num_tokens)
            depth_t = self.extract_and_mask_depth(raw_pred_t, apply_mask=True)

        loss = self.loss_fn(depth_s, depth_t)
        self.log("validation_loss", loss)

        # metrics_dict = compute_metrics(depth_s, depth_t) 
        
        # for metric_name, metric_value in metrics_dict.items():
        #     self.log(
        #         f"val/{metric_name}",
        #         metric_value, 
        #         on_step=False,   
        #         on_epoch=True,   
        #         sync_dist=True,
        #         batch_size=color.size(0)
        #     )
        
        return loss
    
    def test_step(self, batch, batch_idx):
        color = batch['color']
        depth_in = batch['depth']
        num_tokens = self.get_num_tokens()

        raw_pred_s = self(image=color, depth=depth_in, num_tokens=num_tokens)
        depth_s = self.extract_and_mask_depth(raw_pred_s, apply_mask=True)

        with torch.no_grad():
            raw_pred_t = self.teacher(image=color, depth=depth_in, num_tokens=num_tokens)
            depth_t = self.extract_and_mask_depth(raw_pred_t, apply_mask=True)

        loss = self.loss_fn(depth_s, depth_t)
        self.log("test_loss", loss)

        # metrics_dict = compute_metrics(depth_s, depth_t)

        # for metric_name, metric_value in metrics_dict.items():
        #     self.log(
        #         f"test/{metric_name}", 
        #         metric_value, 
        #         on_step=False,   
        #         on_epoch=True,   
        #         sync_dist=True,
        #         batch_size=color.size(0)
        #     )

        return loss