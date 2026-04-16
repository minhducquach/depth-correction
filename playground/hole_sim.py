import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from math import pi

from numpy import random

rng = np.random.default_rng(seed=42)

SCALE = 1.0

PATH = "/home/quachmd/Bureau/depth-correction/datasets/tartanair/House/Data_omni/P0000/depth_lcam_front/000010_lcam_front_depth.png"

BASELINE = 54.8 if SCALE == 1000 else 0.0548
HORIZONTAL_FOV = 87
FOCAL_LENGTH = 1280 / (2 * np.tan((HORIZONTAL_FOV * pi / 180)/2))

def read_decode_depth(depthpath):
    depth_rgba = cv2.imread(depthpath, cv2.IMREAD_UNCHANGED)
    depth = depth_rgba.view("<f4")
    return np.squeeze(depth, axis=-1)
    # return depth_rgba

def depth_to_color_opencv(depth_map, vmin=0.0, vmax=None, colormap=cv2.COLORMAP_TURBO):
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

    # Set invalid pixels to black
    depth_colored[~valid_mask] = [0, 0, 0]

    return depth_colored

def load_depth_map(depth_path, scale=SCALE):
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

    # Read depth map
    depth_map = read_decode_depth(depth_path)
    depth_map = cv2.resize(depth_map, (1280, 720), interpolation=cv2.INTER_NEAREST)
    if depth_map is None:
        raise ValueError(f"Failed to read depth map: {depth_path}")

    # Convert to meters
    depth_map = depth_map.astype(np.float32) / scale
    depth_map = np.clip(depth_map, 0.0, 3.0)

    # Replace invalid values with 0
    depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)

    return depth_map

def calc_shift(depth_value):
    return (FOCAL_LENGTH * BASELINE) / depth_value

if __name__ == "__main__":
    depth_mat = load_depth_map(PATH)

    # Create black canvas
    imperfect_depth = np.copy(depth_mat)
    buffer = np.zeros_like(depth_mat, dtype=np.float32)

    rows, cols = depth_mat.shape

    idx = np.argsort(depth_mat, axis=None)
    # print(depth_mat.shape)
    sorted_mat = depth_mat.flatten()[idx]

    res = np.unique(sorted_mat)[-2:]
    # print(res)

    background_depth = res[-1] if res[-1] != 65.535 else res[-2]
    # print(background_depth)

    # Disparity holes

    # Fill in buffer for right img (left->right)
    for i in range(rows):
        for j in range(cols):
            depth_val = imperfect_depth[i, j]
            
            if depth_val <= 0:
                continue
                
            shift = calc_shift(depth_val)
            j_right = int(round(j - shift))

            if 0 <= j_right < cols:
                if shift > buffer[i, j_right]:
                    buffer[i, j_right] = shift


    # Check consistency (right->left) 
    for i in range(rows):
        for j in range(cols):
            depth_val = imperfect_depth[i, j]
            
            if depth_val <= 0:
                continue
                
            shift = calc_shift(depth_val)
            j_right = int(round(j - shift))

            # 1. If it shifted off-screen, it's a hole (the border effect)
            if j_right < 0 or j_right >= cols:
                imperfect_depth[i, j] = 0
            else:
                # 2. If the buffer at that spot has a larger shift, this pixel is occluded
                if buffer[i, j_right] > (shift + 0.5):
                    # imperfect_depth[i, j] = 0
                    # Shadow
                    depth_right = (FOCAL_LENGTH * BASELINE) / buffer[i, j_right]
                    shadow = (BASELINE * (background_depth - depth_right)) / depth_right
                    shadow_pixels = (shadow * 1280) / (2 * depth_right * np.tan((HORIZONTAL_FOV * pi / 180)/2))
                    imperfect_depth[i, j - int(round(shadow_pixels)): j] = 0
            
    imperfect_depth[imperfect_depth < 0.53] = 0

    sobel_vert_edges = cv2.Sobel(imperfect_depth, cv2.CV_32F, 1, 0, ksize=7)
    sobel_hori_edges = cv2.Sobel(imperfect_depth, cv2.CV_32F, 0, 1, ksize=7)

    # Lateral noise
    kernel_size = 31
    col_kernel = cv2.getGaussianKernel(kernel_size, kernel_size / 3)
    col_kernel = col_kernel / col_kernel.max()
    row_kernel = np.transpose(col_kernel)
    gaussian_kernel = col_kernel * row_kernel

    edge_mask = edge_mask = (np.abs(sobel_vert_edges) > 2048/4) | (np.abs(sobel_hori_edges) > 2048/4)
    edge_mask = edge_mask.astype(np.float32)

    
    noise_power = 0.0432
    noise_field = cv2.filter2D(edge_mask, -1, gaussian_kernel) * noise_power

    valid_mask = imperfect_depth > 0
    imperfect_depth[valid_mask] += noise_field[valid_mask]
    # Axial noise
    kernel_size = 51
    col_kernel = cv2.getGaussianKernel(kernel_size, kernel_size / 3)
    col_kernel = col_kernel / col_kernel.max()
    row_kernel = np.transpose(col_kernel)
    gaussian_kernel = col_kernel * row_kernel

    for i in range(5000):
        row = rng.integers(0, rows - kernel_size)
        col = rng.integers(0, cols - kernel_size)

        noise_power = (0.001063 + 0.0007278 * imperfect_depth[row+int(round(kernel_size//2-1)), col+int(round(kernel_size//2-1))] + 0.003949 * imperfect_depth[row+int(round(kernel_size//2-1)), col+int(round(kernel_size//2-1))] ** 2)
        patch = imperfect_depth[row:row+kernel_size, col:col+kernel_size]
        noise_matrix = (gaussian_kernel * noise_power)
        # noise_matrix = np.clip(noise_matrix, a_min=None, a_max=2.0)

        mask = patch > 0
        patch[mask] = patch[mask] + noise_matrix[mask]
        # noise_power = (0.001063 + 0.0007278 * imperfect_depth + 0.003949 * imperfect_depth ** 2)
        # global_noise = np.random.normal(0, 1, size=imperfect_depth.shape)
        # mask = imperfect_depth > 0
        # imperfect_depth[mask] += (global_noise * noise_power)[mask]

    # # Lateral Noise
    # kernel_size = 31
    # col_kernel = cv2.getGaussianKernel(kernel_size, kernel_size / 3)
    # col_kernel = col_kernel / col_kernel.max()
    # row_kernel = np.transpose(col_kernel)
    # gaussian_kernel = col_kernel * row_kernel

    # for i in range(5000):
    #     row = rng.integers(kernel_size, rows - kernel_size)
    #     col = rng.integers(kernel_size, cols - kernel_size)

    #     noise_power = (0.001063 + 0.0007278 * imperfect_depth[row+int(round(kernel_size//2)), col+int(round(kernel_size//2))] + 0.003949 * imperfect_depth[row+int(round(kernel_size//2)), col+int(round(kernel_size//2))] ** 2)
    #     patch = imperfect_depth[row:row+kernel_size, col:col+kernel_size]
    #     noise_matrix = (gaussian_kernel * noise_power)
    #     # noise_matrix = np.clip(noise_matrix, a_min=None, a_max=2.0)

    #     mask = patch > 0
    #     patch[mask] = patch[mask] + noise_matrix[mask]

    # Remove vertical edges
    imperfect_depth[np.abs(sobel_vert_edges) > 2048/4] = 0

    # # Sub-sampling noise
    # imperfect_depth_resized = cv2.resize(imperfect_depth, (cols // 3, rows // 3), interpolation=cv2.INTER_NEAREST)
    # resized_rows, resized_cols = imperfect_depth_resized.shape

    # for i in range(1000):
    #     shift = rng.integers(1,4)
    #     row = rng.integers(shift, resized_rows)
    #     col = rng.integers(shift, resized_cols)
    #     imperfect_depth_resized[row, col] = imperfect_depth_resized[row, col - shift]
    # imperfect_depth = cv2.resize(imperfect_depth_resized, (cols, rows), interpolation=cv2.INTER_NEAREST)

    # # Axial noise
    # kernel_size = 31
    # col_kernel = cv2.getGaussianKernel(kernel_size, kernel_size / 3)
    # col_kernel = col_kernel / col_kernel.max()
    # row_kernel = np.transpose(col_kernel)
    # gaussian_kernel = col_kernel * row_kernel

    # for i in range(5000):
    #     row = rng.integers(kernel_size, rows - kernel_size)
    #     col = rng.integers(kernel_size, cols - kernel_size)

    #     noise_power = (0.001063 + 0.0007278 * imperfect_depth[row+int(round(kernel_size//2)), col+int(round(kernel_size//2))] + 0.003949 * imperfect_depth[row+int(round(kernel_size//2)), col+int(round(kernel_size//2))] ** 2)
    #     patch = imperfect_depth[row:row+kernel_size, col:col+kernel_size]
    #     noise_matrix = (gaussian_kernel * noise_power)
    #     # noise_matrix = np.clip(noise_matrix, a_min=None, a_max=2.0)

    #     mask = patch > 0
    #     patch[mask] = patch[mask] + noise_matrix[mask]
        
    # # Lateral Noise
    # kernel_size = 31
    # col_kernel = cv2.getGaussianKernel(kernel_size, kernel_size / 3)
    # col_kernel = col_kernel / col_kernel.max()
    # row_kernel = np.transpose(col_kernel)
    # gaussian_kernel = col_kernel * row_kernel

    # for i in range(5000):
    #     row = rng.integers(kernel_size, rows - kernel_size)
    #     col = rng.integers(kernel_size, cols - kernel_size)

    #     noise_power = (0.001063 + 0.0007278 * imperfect_depth[row+int(round(kernel_size//2)), col+int(round(kernel_size//2))] + 0.003949 * imperfect_depth[row+int(round(kernel_size//2)), col+int(round(kernel_size//2))] ** 2)
    #     patch = imperfect_depth[row:row+kernel_size, col:col+kernel_size]
    #     noise_matrix = (gaussian_kernel * noise_power)
    #     # noise_matrix = np.clip(noise_matrix, a_min=None, a_max=2.0)

    #     mask = patch > 0
    #     patch[mask] = patch[mask] + noise_matrix[mask]

    # hole_mask = imperfect_depth == 0

    # # Gaussian blur
    # imperfect_depth = cv2.GaussianBlur(imperfect_depth, (5, 5), 0)
    # imperfect_depth[hole_mask] = 0

    depth_ori = depth_to_color_opencv(depth_mat)
    depth_sim = depth_to_color_opencv(imperfect_depth)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(depth_ori[:, :, ::-1]) # BGR -> RGB by reading in reverse
    plt.subplot(1, 2, 2)
    plt.imshow(depth_sim[:, :, ::-1])
    plt.show()


