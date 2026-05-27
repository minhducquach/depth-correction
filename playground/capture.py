import pyrealsense2 as rs
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import threading
import queue
import time

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

input_queue = queue.Queue(maxsize=2)
output_queue = queue.Queue(maxsize=2)
is_running = True

def depth_to_color(depth_map, colormap=cv2.COLORMAP_TURBO):
    valid_mask = np.isfinite(depth_map) & (depth_map > 0)
    depth_clean = np.where(valid_mask, depth_map, 0).astype(np.float32)
    
    depth_normalized = np.zeros_like(depth_clean, dtype=np.uint8)
    if valid_mask.any():
        cv2.normalize(depth_clean, depth_normalized, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U, mask=valid_mask.astype(np.uint8))
    
    depth_colored = cv2.applyColorMap(depth_normalized, colormap)
    depth_colored[~valid_mask] = [0, 0, 0]
    return depth_colored

def capture_thread_func():
    global is_running
    
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    
    profile = pipe.start(cfg)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)

    while is_running:
        frame = pipe.wait_for_frames()
        aligned_frames = align.process(frame)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)

        # Fast OpenCV Preprocessing
        img_np = cv2.dnn.blobFromImage(color_image, scalefactor=1.0/255.0, size=(848, 480), swapRB=False, crop=False)
        
        depth_np = depth_image * depth_scale
        depth_np = np.expand_dims(depth_np, axis=(0, 1)) # Shape to (1, 1, 480, 848)

        # Bundle data and push to queue. If queue is full, this blocks safely.
        payload = {
            "image": img_np,
            "depth": depth_np,
            "raw_color": color_image,
            "raw_depth": depth_image * depth_scale
        }
        
        try:
            input_queue.put(payload, timeout=1)
        except queue.Full:
            continue # Drop frame if inference is lagging

    pipe.stop()

def inference_thread_func(engine_path):
    global is_running
    
    # 1. Initialize CUDA specifically inside this thread
    cuda.init()
    device = cuda.Device(0)
    cuda_context = device.make_context()
    
    # 2. Load Engine
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as plan:
        engine = runtime.deserialize_cuda_engine(plan.read())
        
    context = engine.create_execution_context()
    
    # 3. Allocate Zero-Copy Buffers
    inputs, outputs = [], []
    stream = cuda.Stream()

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        volume = trt.volume(shape)

        host_mem = cuda.pagelocked_empty(volume, dtype, mem_flags=cuda.host_alloc_flags.DEVICEMAP)
        device_ptr = np.intp(host_mem.base.get_device_pointer())

        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append({'name': name, 'host': host_mem, 'device': device_ptr})
        else:
            outputs.append({'name': name, 'host': host_mem, 'device': device_ptr})

    while is_running:
        try:
            data = input_queue.get(timeout=1)
        except queue.Empty:
            continue

        for inp in inputs:
            if inp["name"] in data:
                np.copyto(inp["host"], data[inp["name"]].ravel())
                context.set_tensor_address(inp["name"], int(inp["device"]))

        for out in outputs:
            context.set_tensor_address(out["name"], int(out["device"]))

        # Execute on GPU
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()

        # Extract results
        depth_out = outputs[0]["host"].reshape(480, 848).copy()
        mask_out = outputs[1]["host"].reshape(480, 848).copy()

        try:
            output_queue.put({
                "depth_pred": depth_out,
                "mask_pred": mask_out,
                "raw_color": data["raw_color"],
                "raw_depth": data["raw_depth"]
            }, timeout=1)
        except queue.Full:
            continue

    # Cleanup CUDA context when shutting down
    cuda_context.pop()

if __name__ == "__main__":
    engine_path = "/home/quachmd/Bureau/depth-correction/convert-onnx/model.trt"

    # Start Threads
    capture_thread = threading.Thread(target=capture_thread_func)
    inference_thread = threading.Thread(target=inference_thread_func, args=(engine_path,))
    
    capture_thread.start()
    inference_thread.start()

    print("Pipeline started. Press 'q' to quit.")

    last_time = time.time()
    
    while is_running:
        try:
            # Get processed data
            res = output_queue.get(timeout=0.1)
        except queue.Empty:
            continue
            
        current_time = time.time()
        fps = 1.0 / (current_time - last_time)
        last_time = current_time

        depth_out = res["depth_pred"]
        mask_out = res["mask_pred"]
        color_image = res["raw_color"]
        
        # Post-process
        binary_mask = mask_out > 0.5
        depth_out = np.where(binary_mask, depth_out, np.inf)
        
        # Fast C++ Color mapping
        depth_pred_colored = depth_to_color(depth_out)
        depth_raw_colored = depth_to_color(res["raw_depth"])

        # Display FPS
        cv2.putText(color_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show images
        cv2.imshow('rgb', color_image)
        cv2.imshow('depth', depth_raw_colored)
        cv2.imshow('pred', depth_pred_colored)

        key = cv2.waitKey(1)
        if key == ord('q'):
            print("Shutting down...")
            is_running = False

            np.save("depth.npy", res["raw_depth"])
            cv2.imwrite("depth.png", depth_raw_colored)
            cv2.imwrite("color.png", color_image)
            break

    # Wait for background threads to safely close
    capture_thread.join()
    inference_thread.join()
    cv2.destroyAllWindows()