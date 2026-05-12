import pyrealsense2 as rs
import numpy as np
import cv2
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def load_normal_engine(engine_path: str) -> trt.ICudaEngine:
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as plan:
        engine = runtime.deserialize_cuda_engine(plan.read())
        return engine
    
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

def depth_to_color_opencv(depth_map, vmin=0, vmax=20, colormap=cv2.COLORMAP_TURBO):
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

cfg.enable_stream(rs.stream.color, 848,480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 848,480, rs.format.z16, 30)

profile = pipe.start(cfg)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

align_to = rs.stream.color
align = rs.align(align_to)

# session = ort.InferenceSession("../convert-onnx/model.onnx", providers=['CUDAExecutionProvider'])

# engine = load_normal_engine("../convert-onnx/model.trt")

engine = load_normal_engine("/home/quachmd/Bureau/depth-correction/convert-onnx/model.trt")
context = engine.create_execution_context()

inputs, outputs, stream = allocate_buffer(engine)

while True:
    start_time = time.time()
    frame = pipe.wait_for_frames()

    aligned_frames = align.process(frame)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    color_image = np.asanyarray(color_frame.get_data())
    # outputs = engine.run(inputs)
    # print(outputs)

    img_np = (color_image / 255.0).astype(np.float32)
    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0)

    depth_np = np.expand_dims(depth_image * depth_scale, axis=0)
    depth_np = np.expand_dims(depth_np, axis=0)

    num_tokens_np = np.array(1200, dtype=np.int64)

    input = {
        "image": img_np,
        # "num_tokens": num_tokens_np,
        "depth": depth_np
    }

    for inp in inputs:
        name = inp["name"]
        if inp["name"] in input:
            data = input[inp["name"]]
            if not data.flags['C_CONTIGUOUS']:
                data = np.ascontiguousarray(data)
            np.copyto(inp["host"], data.ravel())

    results = do_inference(context, inputs, outputs, stream)

    depth_out = outputs[0]["host"].reshape(480,848)
    mask_out = outputs[1]["host"].reshape(480,848)

    binary_mask = mask_out > 0.5

    depth_out = np.where(binary_mask, depth_out, np.inf)
    depth_pred = depth_to_color_opencv(depth_out)

    depth_m = depth_to_color_opencv(depth_image * depth_scale)
    # gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # print(depth_image)
    # print(color_image)
    # break

    # Calculate and display FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(color_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('rgb', color_image)
    cv2.imshow('depth', depth_m)
    cv2.imshow('pred', depth_pred)

    if cv2.waitKey(1) == ord('q'):
        np.save("depth.npy", depth_image * depth_scale)
        cv2.imwrite("depth.png", depth_m)
        cv2.imwrite("color.png", color_image)
        break

pipe.stop()