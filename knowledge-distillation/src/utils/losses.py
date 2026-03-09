import torch
import torch.nn as nn
from torch.nn import functional as F    

class CriterionPixelWise(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, preds_S, preds_T):
        preds_T[0] = preds_T[0].detach()
        assert preds_S[0].shape == preds_T[0].shape,'the output dim of teacher and student differ'
        N,C,H,W = preds_S[0].shape

        flat_T = preds_T[0].permute(0, 2, 3, 1).contiguous().view(-1, C)
        flat_S = preds_S[0].permute(0, 2, 3, 1).contiguous().view(-1, C)

        softmax_pred_T = F.softmax(flat_T, dim=1)

        loss = self.criterion(flat_S, softmax_pred_T)
        return loss