import torch
import cv2
import sys
import os
import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm
import matplotlib.pyplot as plt
# from fvcore.nn import FlopCountAnalysis
from ptflops import get_model_complexity_info

from data_modules.datamodule import MyDataModule
from models.mdm.model.v2 import MDMModel
from utils.metrics import compute_metrics

torch.cuda.empty_cache()

torch.manual_seed(42)

CKPT_PATH = "/home/quachmd/Bureau/depth-correction/knowledge-distillation/src/checkpoints/mdm-distill-epoch=08-validation_loss=0.1196.ckpt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_num_tokens():
    min_tokens, max_tokens = 1200, 3600
    resolution_level = 0
    return int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))

def depth_to_color_opencv(depth_map, vmin=0, vmax=30, colormap=cv2.COLORMAP_TURBO):
    """
    Convert depth map to color visualization using OpenCV colormap.

    Args:
        depth_map (np.ndarray): Depth map (H, W)    
        vmin (float): Minimum depth for colormap (auto if None)
        vmax (float): Maximum depth for colormap (auto if None)
        colormap: OpenCV colormap (TURBO, JET, VIRIDIS, etc.)

    Returns:
        np.ndarray: Colored depth map (H, W, 3) in BGR format
    """
    # Handle invalid values
    valid_mask = np.isfinite(depth_map) & (depth_map > 0)
    depth_clean = depth_map.copy()
    depth_clean[~valid_mask] = 0

    # Auto-range if not specified
    if vmin is None:
        vmin = depth_clean[valid_mask].min() if valid_mask.any() else 0
    if vmax is None:
        vmax = depth_clean[valid_mask].max() if valid_mask.any() else 1

    # Normalize to [0, 255]
    depth_normalized = np.clip(
        (depth_clean - vmin) / (vmax - vmin + 1e-8) * 255,
        0, 255
    ).astype(np.uint8)

    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_normalized, colormap)

    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

    # Set invalid pixels to black
    depth_colored[~valid_mask] = [0, 0, 0]

    return depth_colored

def extract_weights_from_ckpt(ckpt):
    lightning_state_dict = ckpt["state_dict"]
    state_dict = {}
    for key, weight in lightning_state_dict.items():
        if key.startswith("student."):
            clean_key = key.replace("student.", "", 1)
            state_dict[clean_key] = weight
    return state_dict

def evaluate_metrics(o_model, d_model, dataloader, device):
    o_model.eval()
    o_model.to(device)

    d_model.eval()
    d_model.to(device)

    num_tokens = get_num_tokens()

    dataset = dataloader.dataset
    len_dataset = len(dataset)

    # Initialize running totals
    total_metrics_o = {
        'mae': 0.0, 'rmse': 0.0, 
        'abs_rel': 0.0, 'delta_1': 0.0
    }

    total_metrics_d = {
        'mae': 0.0, 'rmse': 0.0, 
        'abs_rel': 0.0, 'delta_1': 0.0
    }

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        with torch.no_grad():
            for batch in tqdm(dataloader):
                color = batch['color'].to(device)
                depth = batch['depth'].to(device).squeeze()

                batch_size = depth.shape[0]

                pred_o = o_model.infer(image=color, depth_in=depth, num_tokens=num_tokens)
                pred_d = d_model.infer(image=color, depth_in=depth, num_tokens=num_tokens)

    #             batch_sums_o, num_valid_o = compute_metrics_sum(pred_o_depth, depth)
    #             batch_sums_d, num_valid_d = compute_metrics_sum(pred_d_depth, depth)

    #             if num_valid_o > 0:
    #                 total_valid_pixels_o += num_valid_o
    #                 for k in total_metrics_o.keys():
    #                     total_metrics_o[k] += batch_sums_o[k]

    #             if num_valid_d > 0:
    #                 total_valid_pixels_d += num_valid_d
    #                 for k in total_metrics_d.keys():
    #                     total_metrics_d[k] += batch_sums_d[k]

                for i in range(batch_size):
                    gt_i = depth[i]
                    
                    pred_o_depth_i = pred_o['depth'][i]
                    pred_d_depth_i = pred_d['depth'][i]

                    metrics_o = compute_metrics(pred_o_depth_i, gt_i)
                    for k in total_metrics_o.keys():
                        total_metrics_o[k] += metrics_o[k]

                    metrics_d = compute_metrics(pred_d_depth_i, gt_i)
                    for k in total_metrics_d.keys():
                        total_metrics_d[k] += metrics_d[k]

    mae_o = total_metrics_o['mae'] / len_dataset
    abs_rel_o = total_metrics_o['abs_rel'] / len_dataset
    delta_1_o = total_metrics_o['delta_1'] / len_dataset
    rmse_o = (total_metrics_o['rmse'] / len_dataset)

    table_o = PrettyTable()
    table_o.field_names = ['Metric', 'MAE', 'RMSE', 'ABS_REL', 'DELTA_1']
    table_o.add_row(['Value', mae_o, rmse_o, abs_rel_o, delta_1_o])

    print(table_o)

    mae_d = total_metrics_d['mae'] / len_dataset
    abs_rel_d = total_metrics_d['abs_rel'] / len_dataset
    delta_1_d = total_metrics_d['delta_1'] / len_dataset
    rmse_d = (total_metrics_d['rmse'] / len_dataset)
    
    table_d = PrettyTable()
    table_d.field_names = ['Metric', 'MAE', 'RMSE', 'ABS_REL', 'DELTA_1']
    table_d.add_row(['Value', mae_d, rmse_d, abs_rel_d, delta_1_d])

    print(table_d)

def evaluate_visual(o_model, d_model, dataloader, device):
    o_model.eval()
    o_model.to(device)

    d_model.eval()
    d_model.to(device)
    
    dataset = dataloader.dataset
    len_dataset = len(dataset)
    random_indices = torch.randint(0, len_dataset, (5, ))

    num_tokens = get_num_tokens()
    # num_tokens = 1200

    fig = plt.figure(figsize=(15,12))
    rows, cols = len(random_indices), 4

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            for i in range(len(random_indices)):
                index = random_indices[i].item()

                color = dataset[index]['color'].to(device, dtype=torch.bfloat16)
                depth = dataset[index]['depth'].to(device, dtype=torch.bfloat16)

                pred_o = o_model.infer(image=color, depth_in=depth, num_tokens=num_tokens)['depth'].squeeze().float().cpu().numpy()
                pred_d = d_model.infer(image=color, depth_in=depth, num_tokens=num_tokens)['depth'].squeeze().float().cpu().numpy()
                
                fig.add_subplot(rows, cols, i * cols + 1)
                color = color.squeeze(0).cpu().float().numpy()
                color = color.transpose(1,2,0)
                color = np.clip(color * 255.0, 0, 255).astype(np.uint8)
                plt.imshow(color)
                plt.axis(False)
                if i == 0:
                    plt.title('RGB image')

                depth_raw_color = depth_to_color_opencv(depth.squeeze().float().cpu().numpy())
                fig.add_subplot(rows, cols, i * cols + 2)
                plt.imshow(depth_raw_color)
                plt.axis(False)
                if i == 0:
                    plt.title('Raw/Ground truth depth image')

                depth_pred_color = depth_to_color_opencv(pred_d)
                fig.add_subplot(rows, cols, i * cols + 3)
                plt.imshow(depth_pred_color)
                plt.axis(False)
                if i == 0:
                    plt.title('Student\'s depth')

                depth_ref_color = depth_to_color_opencv(pred_o)
                fig.add_subplot(rows, cols, i * cols + 4)
                plt.imshow(depth_ref_color)
                plt.axis(False)
                if i == 0:
                    plt.title('Teacher\'s depth')


    plt.tight_layout()
    plt.savefig("../results/depth_comparison.png", dpi=150, bbox_inches='tight')
    print("Saved visualization to 'depth_comparison.png'")
    
    plt.show()

def evaluate_time_complexity(o_model, d_model, dataloader, device):
    o_model.eval()
    o_model.to(device)

    d_model.eval()
    d_model.to(device)

    dataset = dataloader.dataset
    input_elem = dataset[0]

    num_tokens = get_num_tokens()
    # num_tokens = 3600

    def input_constructor(input_res):
        image_input = input_elem['color'].unsqueeze(0).to(device)
        depth_input = input_elem['depth'].to(device)
        return dict(image=image_input, num_tokens=num_tokens, depth=depth_input)

    macs_o, params_o = get_model_complexity_info(
        o_model,
        (1,1,1,1), 
        input_constructor=input_constructor,
        as_strings=True,
        print_per_layer_stat=False
    )

    macs_d, params_d = get_model_complexity_info(
        d_model,
        (1,1,1,1), 
        input_constructor=input_constructor,
        as_strings=True,
        print_per_layer_stat=True
    )

    print('Computational complexity of original model: ', macs_o)
    print('Number of parameters of original model: ', params_o)

    print('Computational complexity of distilled model: ', macs_d)
    print('Number of parameters of distilled model: ', params_d)

# def evaluate_time_complexity(o_model, d_model, device):
    # o_model.eval()
    # o_model.to(device)

    # d_model.eval()
    # d_model.to(device)

    # num_tokens = get_num_tokens()
    # # num_tokens = 3600

    # def input_constructor(input_res):
    #     image_input = torch.rand(1,3,504,504).to(device)
    #     depth_input = torch.rand(1,504,504)
    #     return dict(image=image_input, num_tokens=num_tokens, depth=depth_input)

    # macs_o, params_o = get_model_complexity_info(
    #     o_model,
    #     (1,1,1,1), 
    #     input_constructor=input_constructor,
    #     as_strings=True,
    #     print_per_layer_stat=False
    # )

    # macs_d, params_d = get_model_complexity_info(
    #     d_model,
    #     (1,1,1,1), 
    #     input_constructor=input_constructor,
    #     as_strings=True,
    #     print_per_layer_stat=True
    # )

    # print('Computational complexity of original model: ', macs_o)
    # print('Number of parameters of original model: ', params_o)

    # print('Computational complexity of distilled model: ', macs_d)
    # print('Number of parameters of distilled model: ', params_d)

if __name__ == "__main__":
    data_module = MyDataModule()
    data_module.setup(stage='test')
    test_dataset = data_module.test_dataloader()

    ckpt = torch.load(CKPT_PATH, map_location=device)

    original_model = MDMModel.from_pretrained().to(device)
    original_model.eval()

    distilled_model = MDMModel.from_pretrained_config()
    distilled_model.load_state_dict(extract_weights_from_ckpt(ckpt))
    distilled_model = distilled_model.to(device)
    distilled_model.eval()

    # evaluate_metrics(original_model, distilled_model, test_dataset, device)

    evaluate_visual(original_model, distilled_model, test_dataset, device)

    # evaluate_time_complexity(original_model, distilled_model, test_dataset, device)

    







