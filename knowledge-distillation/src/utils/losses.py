import torch
import torch.nn as nn

class DSSIM(nn.Module):
    """Layer to compute the DSSIM loss between a pair of images
    """
    def __init__(self):
        super().__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

class Criterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction="none")
        self.dssim = DSSIM()

    def forward(self, pred_S, pred_T, pred_mask_S, pred_mask_T):
        t_tensor = pred_T.detach()
        s_tensor = pred_S

        t_mask_tensor = pred_mask_T.detach()
        s_mask_tensor = pred_mask_S

        loss = 0.85 * self.dssim(s_tensor, t_tensor) + 0.15 * self.l1_loss(s_tensor, t_tensor) 
        loss = t_mask_tensor * loss + self.l1_loss(s_mask_tensor, t_mask_tensor)

        # valid_mask_t = torch.isfinite(t_tensor)
        # valid_mask_s = torch.isfinite(s_tensor)

        # t_valid = torch.where(valid_mask_t, t_tensor, torch.zeros_like(t_tensor))
        # s_valid = torch.where(valid_mask_s, s_tensor, torch.zeros_like(t_tensor))

        # loss = 0.85 * self.dssim(s_valid, t_valid) + 0.15 * self.l1_loss(s_valid, t_valid)
        # print('Train loss:', self.dssim(s_valid, t_valid).mean(), self.l1_loss(s_valid, t_valid).mean(), loss.mean())
        
        # loss = 0.85 * self.dssim(s_tensor, t_tensor) + 0.15 * self.l1_loss(s_tensor, t_tensor)
        # print('Train loss:', self.dssim(s_tensor, t_tensor).mean(), self.l1_loss(s_tensor, t_tensor).mean(), loss.mean())

        return loss.mean()