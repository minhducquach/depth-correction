import torch

def compute_metrics(pred, gt):
    valid_mask_gt = (gt > 0) & torch.isfinite(gt)
    valid_mask_pred = (pred > 0) & torch.isfinite(pred)

    pred_valid = torch.where(valid_mask_pred, pred, 0)
    gt_valid = torch.where(valid_mask_gt, gt, 0)

    pred_valid = torch.clamp(pred_valid, min=1e-5)
    gt_valid = torch.clamp(gt_valid, min=1e-5)

    diff = pred_valid - gt_valid
    mae = torch.mean(torch.abs(diff))
    rmse = torch.sqrt(torch.mean(diff ** 2))

    abs_rel = torch.mean(torch.abs(diff) / gt_valid)
    ratio = torch.max(gt_valid / pred_valid, pred_valid / gt_valid)

    delta_1 = torch.mean((ratio < 1.25).float())

    return {
        'mae': mae.item(),
        'rmse': rmse.item(),
        'abs_rel': abs_rel.item(),
        'delta_1': delta_1.item() 
    }


def compute_metrics_sum(pred_depth, gt):
    # valid_mask_gt = (gt > 0) & torch.isfinite(gt)
    # valid_mask_pred = (pred_depth > 0) & torch.isfinite(pred_depth)
    
    # # num_valid = valid_mask.sum().item()
    # # if num_valid == 0:
    # #     return None, 0



    # gt_valid = torch.clamp(gt[valid_mask_gt], min=1e-5)
    # pred_valid = torch.clamp(pred_depth[valid_mask_pred], min=1e-5)

    # diff = pred_valid - gt_valid
    # ratio = torch.max(pred_valid / gt_valid, gt_valid / pred_valid)

    # # sums = {
    # #     'abs_err': torch.sum(torch.abs(diff)).item(),
    # #     'sq_err': torch.sum(diff ** 2).item(),
    # #     'abs_rel_err': torch.sum(torch.abs(diff) / gt_valid).item(),
    # #     'delta_1_count': torch.sum((ratio < 1.25).float()).item()
    # # }

    # return sums, num_valid
    pass