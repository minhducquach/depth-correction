import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def allocate_buffer(engine):
    inputs, outputs = [], []
    stream = cuda.Stream()

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        volume = trt.volume(shape)

        host_mem = cuda.pagelocked_empty(volume, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append({'name': name, 'host': host_mem, 'device': device_mem, 'shape': shape})
        else:
            outputs.append({'name': name, 'host': host_mem, 'device': device_mem, 'shape': shape})

    return inputs, outputs, stream

def do_inference(context, inputs, outputs, stream):
    for inp in inputs:
        cuda.memcpy_htod_async(inp["device"], inp["host"], stream)
        context.set_tensor_address(inp["name"], int(inp["device"]))

    for out in outputs:
        context.set_tensor_address(out["name"], int(out["device"]))

    context.execute_async_v3(stream_handle=stream.handle)

    for out in outputs:
        cuda.memcpy_dtoh_async(out["host"], out["device"], stream)

    stream.synchronize()

    return [out["host"] for out in outputs]

def load_normal_engine(engine_path: str) -> trt.ICudaEngine:
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as plan:
        engine = runtime.deserialize_cuda_engine(plan.read())
        return engine


def depth_to_color_opencv(depth_map, vmin=0, vmax=10, colormap=cv2.COLORMAP_TURBO):
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

    # print(vmin, vmax)

    # Normalize to [0, 255]
    depth_normalized = np.clip(
        (depth_clean - vmin) / (vmax - vmin + 1e-8) * 255,
        0, 255
    ).astype(np.uint8)

    # print(depth_normalized)

    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_normalized, colormap)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

    # Set invalid pixels to black
    depth_colored[~valid_mask] = [0, 0, 0]

    return depth_colored

# session = ort.InferenceSession("./model.onnx", providers=['CPUExecutionProvider'])

img = cv2.imread('/home/quachmd/Bureau/depth-correction/datasets/darknav/Circular/rgb/1739893136_240500000.png')
img_np = (img / 255.0).astype(np.float32)
img_np = np.transpose(img_np, (2, 0, 1))
img_np = np.expand_dims(img_np, axis=0)

depth_np = np.load("/home/quachmd/Bureau/depth-correction/datasets/darknav/Circular/depth/1739893136_240500000.npy").astype(np.float32) / 1000.0
depth_np = np.expand_dims(depth_np, axis=0)
depth_np = np.expand_dims(depth_np, axis=0)

num_tokens_np = np.array(1200, dtype=np.int64)

input = {
    "image": img_np,
    # "num_tokens": num_tokens_np,
    "depth": depth_np
}

engine = load_normal_engine("/home/quachmd/Bureau/depth-correction/convert-onnx/model.trt")
context = engine.create_execution_context()

inputs, outputs, stream = allocate_buffer(engine)

for inp in inputs:
    if inp["name"] in input:
        data = input[inp["name"]]
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        np.copyto(inp["host"], data.ravel())

results = do_inference(context, inputs, outputs, stream)

depth_out = outputs[0]["host"].reshape(480,848)
mask_out = outputs[1]["host"].reshape(480,848)

print("Depth output shape:", depth_out.shape)
print("Mask output shape:", mask_out.shape)

binary_mask = mask_out > 0.5

depth_out = np.where(binary_mask, depth_out, np.inf)

depth = depth_to_color_opencv(depth_out)
depth_inp = depth_to_color_opencv(depth_np.squeeze())

plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
plt.title("Input RGB")
plt.imshow(img)
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Input depth")
plt.imshow(depth_inp)
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Output depth")
plt.imshow(depth)
plt.axis('off')

plt.tight_layout()
plt.show()