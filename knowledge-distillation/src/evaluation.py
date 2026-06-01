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
from sklearn.decomposition import PCA
import torch.nn.functional as F

from data_modules.datamodule import MyDataModule
from models.mdm.model.v2 import MDMModel
from utils.metrics import compute_metrics

from pathlib import Path

# torch.cuda.empty_cache()

torch.manual_seed(16)

CKPT_PATH = "/home/quachmd/Bureau/depth-correction/knowledge-distillation/src/checkpoints/mdm-distill-epoch=15-validation_loss=0.5323.ckpt"
CKPT_PATH_2 = "/home/quachmd/Bureau/depth-correction/mdm-distill-epoch=07-validation_loss=0.0524.ckpt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

img_path = "/home/quachmd/Bureau/depth-correction/knowledge-distillation/playground/1739893136_240500000.png"
depth_path = "/home/quachmd/Bureau/depth-correction/knowledge-distillation/playground/1739893136_240500000.npy"

def get_num_tokens():
    # min_tokens, max_tokens = 1200, 3600
    # resolution_level = 0
    # return int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))
    return int(480*848 / 196.0)

def load_depth_map(depth_path, scale=1000.0):
    """
    Load depth map from PNG file (16-bit) and convert to meters.

    Args:
        depth_path (str): Path to depth image
        scale (float): Scale factor to convert to meters
            - 1000.0 for millimeters
            - 1.0 for meters

    Returns:
        np.ndarray: Depth map in meters (H, W), float32
    """
    if not Path(depth_path).exists():
        raise FileNotFoundError(f"Depth map not found: {depth_path}")

    # Read depth map as 16-bit
    # depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    depth_map = np.load(depth_path)
    if depth_map is None:
        raise ValueError(f"Failed to read depth map: {depth_path}")

    # Convert to meters
    depth_map = depth_map.astype(np.float32) / scale                                                    

    # Replace invalid values with 0
    depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)

    return depth_map

def depth_to_color_opencv(depth_map, mask=None, vmin=0, vmax=30, colormap=cv2.COLORMAP_TURBO):
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
    if mask is None:
        valid_mask = np.isfinite(depth_map) & (depth_map > 0)
    else:
        valid_mask = mask
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
                    
                    pred_o_depth_i, pred_o_mask_i = pred_o['depth'][i], pred_o['mask'][i]
                    pred_d_depth_i, pred_d_mask_i = pred_d['depth'][i], pred_d['mask'][i]

                    metrics_o = compute_metrics(pred_o_depth_i, pred_o_mask_i, gt_i)
                    for k in total_metrics_o.keys():
                        total_metrics_o[k] += metrics_o[k]

                    metrics_d = compute_metrics(pred_d_depth_i, pred_d_mask_i, gt_i)
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

def evaluate_visual(o_model, d_model, d_model_2, dataloader, device):
    o_model.eval()
    o_model.to(device)

    d_model.eval()
    d_model.to(device)

    d_model_2.eval()
    d_model_2.to(device)
    
    dataset = dataloader.dataset
    len_dataset = len(dataset)
    random_indices = torch.randint(0, len_dataset, (5, ))

    num_tokens = get_num_tokens()
    # num_tokens = 1200

    fig = plt.figure(figsize=(15,12))
    rows, cols = len(random_indices), 4

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        with torch.no_grad():
            for i in range(len(random_indices)):
                index = random_indices[i].item()

                color = dataset[index]['color'].to(device)
                depth = dataset[index]['depth'].to(device)

                pred_o = o_model.infer(image=color, depth_in=depth, num_tokens=2352)
                pred_s = d_model.infer(image=color, depth_in=depth, num_tokens=2352)
                
                pred_d_o, pred_m_o = pred_o['depth'].squeeze().float().cpu().numpy(), pred_o['mask'].squeeze().cpu().numpy()
                pred_d_s, pred_m_s = pred_s['depth'].squeeze().float().cpu().numpy(), pred_s['mask'].squeeze().cpu().numpy()
                
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

                depth_pred_color = depth_to_color_opencv(pred_d_s, pred_m_s)
                fig.add_subplot(rows, cols, i * cols + 3)
                plt.imshow(depth_pred_color)
                plt.axis(False)
                if i == 0:
                    plt.title('Student\'s depth')

                depth_ref_color = depth_to_color_opencv(pred_d_o, pred_m_o)
                fig.add_subplot(rows, cols, i * cols + 4)
                plt.imshow(depth_ref_color)
                plt.axis(False)
                if i == 0:
                    plt.title('Teacher\'s depth')


    plt.tight_layout()
    plt.savefig("../results/depth_comparison_ad1.png", dpi=150, bbox_inches='tight')
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

def visualize_inter_map(o_model, d_model, device):
    o_model.eval()
    o_model.to(device)

    d_model.eval()
    d_model.to(device)

    color = cv2.imread('/home/quachmd/Bureau/depth-correction/knowledge-distillation/playground/1739893136_240500000.png')
    color = cv2.resize(color, (848, 480))
    color_tensor = torch.tensor(color / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)[None]

    depth_np = np.load('/home/quachmd/Bureau/depth-correction/knowledge-distillation/playground/1739893136_240500000.npy').astype(np.float32)
    depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
    depth_np = cv2.resize(depth_np, (848, 480), interpolation=cv2.INTER_NEAREST)
    depth_tensor = torch.tensor(depth_np, dtype=torch.float32, device=device)[None, None]

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        with torch.no_grad():
            pred_o, o_vit_feats, o_depth_maps, cls_token_o = o_model.forward(color_tensor, 1200, depth_tensor, extract_layers=[5,11,17,23])
            pred_d, d_vit_feats, d_depth_maps, cls_token_d = d_model.forward(color_tensor, 1200, depth_tensor, extract_layers=[2,5,8,11])

            # if isinstance(o_vit_feats, list) and isinstance(d_vit_feats, list):
            #     o_vit_feats = o_vit_feats[-1]
            #     d_vit_feats = d_vit_feats[-1]

            # # print(o_vit_feats.shape, d_vit_feats.shape)

            # fig, axs = plt.subplots(3, 8, figsize=(64,32))
            # for ax in axs.flat:
            #     ax.axis('off')

            # # print(d_vit_feats.cpu().numpy().shape)
            print([m.shape for m in d_depth_maps])
            print([m.shape for m in o_depth_maps])                                                
            # # print(cls_token_d.cpu().numpy().shape)

            # last_depth_map_d = d_depth_maps[-1].squeeze()
            # last_depth_map_o = o_depth_maps[-1].squeeze()

            # for i in range(8):
            #     depth_d = last_depth_map_o[i, :, :].float().cpu().numpy()
            #     depth_d_1 = last_depth_map_o[i+8, :, :].float().cpu().numpy()
            #     depth_d_2 = last_depth_map_o[i+16, :, :].float().cpu().numpy()

            #     axs[0, i].imshow(depth_d)
            #     axs[1, i].imshow(depth_d_1)
            #     axs[2, i].imshow(depth_d_2)

            # for i in range(8, 16):
            #     depth_d = last_depth_map_d[i, :, :].float().cpu().numpy()
            #     depth_o = last_depth_map_o[i*2, :, :].float().cpu().numpy()
            #     depth_o_1 = last_depth_map_o[i*2+1, :, :].float().cpu().numpy()

            #     axs[3, i-8].imshow(depth_d)
            #     axs[4, i-8].imshow(depth_o)
            #     axs[5, i-8].imshow(depth_o_1)

            # fig, axs = plt.subplots(1, 2)

            # B, C_t, H, W = o_vit_feats.shape
            # _, C_s, _, _ = d_vit_feats.shape
            
            # # 1. PCA Visualization (More than 3 components)
            # n_components = 3
            # feat_t_flat = o_vit_feats[-1].reshape(C_t, -1).permute(1, 0).float().cpu().numpy() # (1196, 1024)
            # feat_s_flat = d_vit_feats[-1].reshape(C_s, -1).permute(1, 0).float().cpu().numpy() # (1196, 192)
            
            # pca_t = PCA(n_components=n_components).fit_transform(feat_t_flat)
            # pca_s = PCA(n_components=n_components).fit_transform(feat_s_flat)

            # print(pca_t.shape)
            
            # # Normalize independently
            # t_min, t_max = pca_t.min(axis=0), pca_t.max(axis=0)
            # s_min, s_max = pca_s.min(axis=0), pca_s.max(axis=0)
            # pca_t = (pca_t - t_min) / (t_max - t_min + 1e-8)
            # pca_s = (pca_s - s_min) / (s_max - s_min + 1e-8)
            
            # pca_img_t = pca_t.reshape(H, W, n_components)
            # pca_img_s = pca_s.reshape(H, W, n_components)

            # axs[0].imshow(pca_img_t)
            # axs[0].set_title("Teacher ViT Features (PCA - 1024D)")
            # axs[0].axis('off')
            
            # axs[1].imshow(pca_img_s)
            # axs[1].set_title("Student ViT Features (PCA - 192D)")
            # axs[1].axis('off')
            
            # # Plotting individual components
            # fig_pca, axs_pca = plt.subplots(n_components, 2, figsize=(10, 3 * n_components))
            # for i in range(n_components):
            #     axs_pca[i, 0].imshow(pca_img_t[..., i], cmap='viridis')
            #     axs_pca[i, 0].set_title(f"Teacher PC {i+1}")
            #     axs_pca[i, 0].axis('off')
                
            #     axs_pca[i, 1].imshow(pca_img_s[..., i], cmap='viridis')
            #     axs_pca[i, 1].set_title(f"Student PC {i+1}")
            #     axs_pca[i, 1].axis('off')
                
            # Setup for Attention Transfer Visualization (commented code below)
            # fig, axs = plt.subplots(len(o_vit_feats), 2, figsize=(14, 10))

            # def at(x):
            #     return x.pow(2).mean(0)
            
            # for i in range(len(o_vit_feats)):
            
            #     at_t = cv2.normalize(at(o_vit_feats[i][0]).float().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
            #     # at_t = cv2.resize(at_t, (848, 480), interpolation=cv2.INTER_CUBIC)

            #     at_s = cv2.normalize(at(d_vit_feats[i][0]).float().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
            #     # at_s = cv2.resize(at_s, (848, 480), interpolation=cv2.INTER_CUBIC)

            #     # at_t = cv2.normalize(cv2.resize(at(o_vit_feats[0]).float().cpu().numpy(), (848, 480)), None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
            #     # at_s = cv2.normalize(cv2.resize(at(d_vit_feats).float().cpu().numpy(), (848, 480)), None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)

            #     # print(at_t * 255.0)
            #     # print(at_t)

            #     # heatmap_att = cv2.addWeighted(np.asarray(color[:,:,1], dtype=np.float32), 0.7, np.asarray(at_t, dtype=np.float32), 0.3, 0)

            #     axs[i,0].imshow(at_t, interpolation='bicubic')
            #     # axs[0].set_title("Teacher AT")
            #     axs[i,0].axis('off')
                
            #     axs[i,1].imshow(at_s, interpolation='bicubic')
            #     # axs[1].set_title("Student AT")
            #     axs[i,1].axis('off')
                            
            # plt.tight_layout()
            # plt.show()

def evaluate_mde(color_path, model, device):
    img = cv2.imread(color_path)
    img = cv2.resize(img, (848, 480))
    img_tensor = torch.tensor(img / 255.0, dtype=torch.float32, device=device).permute(2,0,1)[None]
    h, w = img_tensor.shape[2], img_tensor.shape[3]
    depth_holder = torch.zeros((h, w), dtype=torch.float32, device=device)[None]

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        with torch.no_grad():
            output = model.infer(img_tensor, depth_in=depth_holder, num_tokens=1200)
    
    depth_pred = output["depth"].squeeze().float().cpu().numpy()

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img)
    axs[1].imshow(depth_to_color_opencv(depth_pred))
    axs[0].axis('off')
    axs[1].axis('off')
    plt.tight_layout()
    plt.show()

def evaluate_single_img(color_path, depth_path, model, device):
    img = cv2.imread(color_path)
    img = cv2.resize(img, (848, 480))
    img_tensor = torch.tensor(img / 255.0, dtype=torch.float32, device=device).permute(2,0,1)[None].to(device)

    # depth_np = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 100.0
    depth_np = load_depth_map(depth_path)
    depth_np = cv2.resize(depth_np, (848, 480), interpolation=cv2.INTER_NEAREST)

    depth_tensor = torch.tensor(depth_np, dtype=torch.float32, device=device)[None].to(device)

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        with torch.no_grad():
            output = model.infer(img_tensor, depth_in=depth_tensor, num_tokens=1200)
    
    depth_pred = output["depth"].squeeze().float().cpu().numpy()

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(img)
    axs[1].imshow(depth_to_color_opencv(depth_np))
    axs[2].imshow(depth_to_color_opencv(depth_pred))
    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_module = MyDataModule()
    data_module.setup(stage='test')
    test_dataset = data_module.test_dataloader()

    ckpt = torch.load(CKPT_PATH, map_location=device)

    original_model = MDMModel.from_pretrained().to(device)
    original_model.eval()

    distilled_model = MDMModel.from_pretrained_config_small().to(device)
    distilled_model.load_state_dict(extract_weights_from_ckpt(ckpt))
    distilled_model = distilled_model.to(device)
    distilled_model.eval()

    ckpt2 = torch.load(CKPT_PATH_2, map_location=device)
    distilled_model2 = MDMModel.from_pretrained_config_small().to(device)
    distilled_model2.load_state_dict(extract_weights_from_ckpt(ckpt2))
    distilled_model2 = distilled_model2.to(device)
    distilled_model2.eval()
    # evaluate_metrics(original_model, distilled_model, test_dataset, device)

    evaluate_visual(original_model, distilled_model, distilled_model2, test_dataset, device)

    # evaluate_time_complexity(original_model, distilled_model, test_dataset, device)

    # visualize_inter_map(original_model, distilled_model,  device)

    # evaluate_mde(color_path=img_path, model=distilled_model, device=device)

    # evaluate_single_img(color_path=img_path, depth_path=depth_path, model=distilled_model, device=device)
