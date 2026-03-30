import pyrealsense2 as rs
import numpy as np
import cv2
import onnxruntime as ort

def depth_to_color_opencv(depth_map, vmin=None, vmax=None, colormap=cv2.COLORMAP_TURBO):
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

pipe = rs.pipeline()
cfg  = rs.config()

cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)

profile = pipe.start(cfg)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

align_to = rs.stream.color
align = rs.align(align_to)

session = ort.InferenceSession("../convert-onnx/model.onnx", providers=['CUDAExecutionProvider'])

while True:
    frame = pipe.wait_for_frames()

    aligned_frames = align.process(frame)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    color_image = np.asanyarray(color_frame.get_data())

    img_np = (color_image / 255.0).astype(np.float32)
    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0)

    depth_np = np.expand_dims(depth_image * depth_scale, axis=0)
    depth_np = np.expand_dims(depth_np, axis=0)

    num_tokens_np = np.array(1200, dtype=np.int64)

    inputs = {
        "image": img_np,
        "num_tokens": num_tokens_np,
        "depth": depth_np
    }

    outputs = session.run(None, inputs)
    depth_out = outputs[0].squeeze()
    mask_out = outputs[1].squeeze()

    binary_mask = mask_out > 0.5

    depth_out = np.where(binary_mask, depth_out, np.inf)
    depth_pred = depth_to_color_opencv(depth_out)

    depth_m = depth_to_color_opencv(depth_image * depth_scale)
    # gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # print(depth_image)
    # print(color_image)
    # break

    cv2.imshow('rgb', color_image)
    cv2.imshow('depth', depth_m)
    cv2.imshow('pred', depth_pred)

    if cv2.waitKey(33) == ord('q'):
        np.save("depth.npy", depth_image * depth_scale)
        cv2.imwrite("depth.png", depth_m)
        cv2.imwrite("color.png", color_image)
        break

pipe.stop()