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

CKPT_PATH = "/content/drive/MyDrive/depth-correction/mdm-distill-epoch=07-validation_loss=0.0524-clean.ckpt"

class DistillationModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = config["learning_rate"]

        ckpt = torch.load(CKPT_PATH)

        self.teacher = MDMModel.from_pretrained_config_small()
        self.teacher.load_state_dict(ckpt)
        self.teacher.eval()
        # self.teacher.compile(mode='default')

        # self.student = MDMModel.from_pretrained_config()
        self.student = MDMModel.from_pretrained_config()

        # self.student.encoder.enable_gradient_checkpointing()
        # self.student.neck.enable_gradient_checkpointing()
        # self.student.depth_head.enable_gradient_checkpointing()
        # self.student.mask_head.enable_gradient_checkpointing()

        self.loss_fn = Criterion(alpha=config["alpha"], beta=config["beta"], gamma=config["gamma"], delta=config["delta"])
        # self.loss_fn.compile(mode='default')

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def extract_weights_from_ckpt(self, ckpt):
      lightning_state_dict = ckpt["state_dict"]
      state_dict = {}
      for key, weight in lightning_state_dict.items():
          if key.startswith("student."):
              clean_key = key.replace("student.", "", 1)
              state_dict[clean_key] = weight
      return state_dict

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(
        #     self.student.parameters(), 
        #     lr=self.learning_rate, 
        #     weight_decay=self.hparams.config["weight_decay"]
        # )

        backbone_params = []
        decoder_params = []
        for name, param in self.student.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            else:
                decoder_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {
                    'params': backbone_params,
                    'lr': self.hparams.config["learning_rate_backbone"]
                },
                {
                    'params': decoder_params
                }
            ],
            lr=self.learning_rate, 
            betas=self.hparams.config["betas_lr"],
            weight_decay=self.hparams.config["weight_decay"]
        )

        def encoder_lr_lambda(current_step):
            if current_step <= 10:
                return float(current_step) / 10.0
            else:
                decay_steps = (current_step - 10) // 250
                return 0.5 ** decay_steps

        def decoder_lr_lambda(current_step):
            if current_step <= 10:
                return float(current_step) / 10.0
            else:
                decay_steps = (current_step - 10) // 250
                return 0.5 ** decay_steps

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=[encoder_lr_lambda, decoder_lr_lambda]
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(factor=0.5, patience=3),

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step" 
            }
        }
    
    def forward(self, image, depth, num_tokens, extract_layers):
        return self.student(image=image, depth=depth, num_tokens=num_tokens, extract_layers=extract_layers)
    
    def on_train_epoch_start(self):
        self.teacher.eval()

    # def on_load_checkpoint(self, checkpoint):
    #     state_dict = checkpoint.get("state_dict", {})
    #     cleaned_state_dict = {}
    #     for k, v in state_dict.items():
    #         cleaned_state_dict[k.replace("_orig_mod.", "")] = v
    #     checkpoint["state_dict"] = cleaned_state_dict

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

    def get_num_tokens(self, img_height, img_width):
        # min_tokens, max_tokens = 1200, 3600
        # resolution_level = 0
        # return int(min_tokens + (resolution_level / 9.0) * (max_tokens - min_tokens))
        return int(img_height * img_width / 196.0)
    
    def training_step(self, batch, batch_idx):
        color = batch['color']
        depth_in = batch['depth']

        h, w = color.shape[2], color.shape[3]
        num_tokens = self.get_num_tokens(h, w)

        raw_pred_s, intermediate_feat_s, feat_neck_s, cls_token_s = self(image=color, depth=depth_in, num_tokens=num_tokens, extract_layers=(1,2,3,4,5,6,7,8,9,10,11))
        # depth_s = self.extract_and_mask_depth(raw_pred_s, apply_mask=True)
        depth_s, mask_s = raw_pred_s['depth_reg'], raw_pred_s['mask']

        with torch.no_grad():
            raw_pred_t, intermediate_feat_t, feat_neck_t, cls_token_t = self.teacher(image=color, depth=depth_in, num_tokens=num_tokens, extract_layers=(1,2,3,4,5,6,7,8,9,10,11))
            # depth_t = self.extract_and_mask_depth(raw_pred_t, apply_mask=True)
            depth_t, mask_t = raw_pred_t['depth_reg'], raw_pred_t['mask']

        # loss, dl, dssiml, ml, atl = self.loss_fn(depth_s, depth_t, mask_s, mask_t, feat_neck_t, feat_neck_s, intermediate_feat_t, intermediate_feat_s)
        loss, dl, gradl, ml, atl = self.loss_fn(depth_s, depth_t, mask_s, mask_t, feat_neck_t, feat_neck_s, intermediate_feat_t, intermediate_feat_s)
        # print('Train loss:', loss)

        # self.log("train_loss", loss)
        # self.log("depth_loss", dl)
        # self.log("at_loss", atl)
        # self.log("mask_loss", ml)
        # self.log("grad_loss", gradl)
        return loss
    
    def validation_step(self, batch, batch_idx):
        color = batch['color']
        depth_in = batch['depth']
        h, w = color.shape[2], color.shape[3]
        num_tokens = self.get_num_tokens(h, w)

        raw_pred_s, intermediate_feat_s, feat_neck_s, cls_token_s = self(image=color, depth=depth_in, num_tokens=num_tokens, extract_layers=(1,2,3,4,5,6,7,8,9,10,11))
        # depth_s = self.extract_and_mask_depth(raw_pred_s, apply_mask=True)
        depth_s, mask_s = raw_pred_s['depth_reg'], raw_pred_s['mask']

        with torch.no_grad():
            raw_pred_t, intermediate_feat_t, feat_neck_t, cls_token_t = self.teacher(image=color, depth=depth_in, num_tokens=num_tokens, extract_layers=(1,2,3,4,5,6,7,8,9,10,11))
            # depth_t = self.extract_and_mask_depth(raw_pred_t, apply_mask=True)
            depth_t, mask_t = raw_pred_t['depth_reg'], raw_pred_t['mask']

        # loss, dl, dssiml, ml, atl = self.loss_fn(depth_s, depth_t, mask_s, mask_t, feat_neck_t, feat_neck_s, intermediate_feat_t, intermediate_feat_s)
        loss, dl, gradl, ml, atl = self.loss_fn(depth_s, depth_t, mask_s, mask_t, feat_neck_t, feat_neck_s, intermediate_feat_t, intermediate_feat_s)
        # self.log("validation_loss", loss)
        # self.log("depth_loss_val", dl)
        # self.log("at_loss_val", atl)
        # self.log("mask_loss_val", ml)
        # self.log("grad_loss_val", gradl)

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
        h, w = color.shape[2], color.shape[3]
        num_tokens = self.get_num_tokens(h, w)

        raw_pred_s, intermediate_feat_s, feat_neck_s, cls_token_s = self(image=color, depth=depth_in, num_tokens=num_tokens, extract_layers=(1,2,3,4,5,6,7,8,9,10,11))
        # depth_s = self.extract_and_mask_depth(raw_pred_s, apply_mask=True)
        depth_s, mask_s = raw_pred_s['depth_reg'], raw_pred_s['mask']

        with torch.no_grad():
            raw_pred_t, intermediate_feat_t, feat_neck_t, cls_token_t = self.teacher(image=color, depth=depth_in, num_tokens=num_tokens, extract_layers=(1,2,3,4,5,6,7,8,9,10,11))
            # depth_t = self.extract_and_mask_depth(raw_pred_t, apply_mask=True)
            depth_t, mask_t = raw_pred_t['depth_reg'], raw_pred_t['mask']

        # loss, dl, dssiml, ml, atl = self.loss_fn(depth_s, depth_t, mask_s, mask_t, feat_neck_t, feat_neck_s, intermediate_feat_t, intermediate_feat_s)
        loss, dl, gradl, ml, atl = self.loss_fn(depth_s, depth_t, mask_s, mask_t, feat_neck_t, feat_neck_s, intermediate_feat_t, intermediate_feat_s)
        # self.log("test_loss", loss)
        # self.log("depth_loss_test", dl)
        # self.log("at_loss_test", atl)
        # self.log("mask_loss_test", ml)
        # self.log("grad_loss_test", gradl)
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

