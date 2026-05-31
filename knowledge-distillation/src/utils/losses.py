import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

        Gx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        Gy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, x, y):
        grad_x = self.filter(x)
        grad_x = torch.mul(grad_x, grad_x)
        grad_x = torch.sum(grad_x, dim=1, keepdim=True)
        grad_x = torch.sqrt(grad_x)

        grad_y = self.filter(y)
        grad_y = torch.mul(grad_y, grad_y)
        grad_y = torch.sum(grad_y, dim=1, keepdim=True)
        grad_y = torch.sqrt(grad_y)

        return grad_x, grad_y

class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 0.001

    def forward(self, input, target, valid_mask=None):
        input_downscaled = [input] + [input[..., :: 2 * i, :: 2 * i] for i in range(1, 4)]
        target_downscaled = [target] + [target[..., :: 2 * i, :: 2 * i] for i in range(1, 4)]
        if valid_mask is not None:
            valid_mask = valid_mask > 0.5
            mask_downscaled = [valid_mask] + [valid_mask[..., :: 2 * i, :: 2 * i] for i in range(1, 4)]
        else:
            mask_downscaled = [torch.ones_like(target, dtype=bool) for target in target_downscaled]

        gradient_loss = 0
        for input, target, mask in zip(input_downscaled, target_downscaled, mask_downscaled):
            N = torch.sum(mask)
            d_diff = torch.mul(input - target, mask)

            v_gradient = torch.abs(d_diff[..., 0:-2, :] - d_diff[..., 2:, :])
            v_mask = torch.mul(mask[..., 0:-2, :], mask[..., 2:, :])
            v_gradient = torch.mul(v_gradient, v_mask)

            h_gradient = torch.abs(d_diff[..., :, 0:-2] - d_diff[..., :, 2:])
            h_mask = torch.mul(mask[..., :, 0:-2], mask[..., :, 2:])
            h_gradient = torch.mul(h_gradient, h_mask)
            gradient_loss += (torch.sum(h_gradient) + torch.sum(v_gradient)) / N

        return gradient_loss

class Criterion(nn.Module):
    def __init__(self, alpha=0.85, beta=0.15, gamma=0.1, delta=0.5):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction="none")
        # self.l2_loss = nn.MSELoss(reduction="none")
        # self.sobel = Sobel()
        self.dssim = DSSIM()
        self.grad_loss = GradientLoss()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def at(self, x):
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

    def at_loss(self, x, y):
        return (self.at(x) - self.at(y)).pow(2).sum(dim=1).mean()

    def forward(self, pred_S, pred_T, pred_mask_S, pred_mask_T, features_T=None, features_S=None, inter_T=None, inter_S=None):
        t_tensor = pred_T.detach()
        s_tensor = pred_S

        t_mask_tensor = pred_mask_T.detach()
        s_mask_tensor = pred_mask_S

        # grad_t, grad_s = self.sobel(t_tensor, s_tensor)
        # grad_loss = self.l1_loss(grad_t, grad_s)

        # L1 version - Sobel
        # loss = 0.85 * (self.dssim(s_tensor, t_tensor) + grad_loss) + 0.15 * self.l1_loss(s_tensor, t_tensor)
        # loss = t_mask_tensor * loss + self.l1_loss(s_mask_tensor, t_mask_tensor)

        # masked_depth_loss = t_mask_tensor * loss

        # valid_pixels = t_mask_tensor.sum() + 1e-8
        # avg_masked_depth_loss = masked_depth_loss.sum() / valid_pixels

        # total_loss = avg_masked_depth_loss + self.l1_loss(s_mask_tensor, t_mask_tensor).mean()


        # L1 version - vanila
        # loss = 0.85 * self.dssim(s_tensor, t_tensor) + 0.15 * self.l1_loss(s_tensor, t_tensor) 
        # loss = t_mask_tensor * loss + self.l1_loss(s_mask_tensor, t_mask_tensor)

        # L1 - Grad Matching
        depth_loss = self.beta * self.l1_loss(s_tensor, t_tensor)
        mask_loss = self.l1_loss(s_mask_tensor, t_mask_tensor)
        grad_loss = self.delta * self.grad_loss(s_tensor, t_tensor, t_mask_tensor)

        # Average depth loss only over valid pixels
        valid_pixels = t_mask_tensor.sum() + 1e-8
        avg_depth_loss = (t_mask_tensor * depth_loss).sum() / valid_pixels
        
        avg_mask_loss = mask_loss.mean()

        if features_T is not None and features_S is not None:
            at_loss = self.gamma * sum(self.at_loss(features_T[i], features_S[i]) for i in range(len(features_S)))
            # print(at_loss)

        total_loss = avg_depth_loss + avg_mask_loss + grad_loss + at_loss

        return total_loss, avg_depth_loss, grad_loss, avg_mask_loss, at_loss

        # L1 version - AT - DSSIM
        # depth_loss = self.beta * self.l1_loss(s_tensor, t_tensor)
        # dssim_loss = self.alpha * self.dssim(s_tensor, t_tensor)
        # mask_loss = self.l1_loss(s_mask_tensor, t_mask_tensor)
        # at_loss = None

        # loss = dssim_loss + depth_loss
        # loss = t_mask_tensor * loss + mask_loss

        # total_loss = loss.mean()

        # if features_T is not None and features_S is not None:
        #     at_loss = self.gamma * sum(self.at_loss(features_T[i], features_S[i]) for i in range(len(features_S)))
        #     sum_at_neck = at_loss
        #     total_loss += sum_at_neck
            
        # if inter_T is not None and inter_S is not None:
        #     sum_at_inter = sum(self.at_loss(inter_T[i], inter_S[i]) for i in range(len(inter_S)))
        #     total_loss += self.gamma * sum_at_inter

        # return total_loss, depth_loss.mean(), dssim_loss.mean(), mask_loss.mean(), at_loss

