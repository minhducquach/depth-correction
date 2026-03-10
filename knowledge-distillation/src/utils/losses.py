import torch
import torch.nn as nn

class CriterionPixelWise(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, preds_S, preds_T):
        t_tensor = preds_T['depth_reg'].detach()
        s_tensor = preds_S['depth_reg']

        assert s_tensor.shape == t_tensor.shape, f'Output dims differ: Student {s_tensor.shape} vs Teacher {t_tensor.shape}'

        loss = self.criterion(s_tensor, t_tensor)
        
        return loss