import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from typing import Optional

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine(onnx_file_path: str, engine_file_path: str) -> Optional[trt.ICudaEngine]:
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser, \
         builder.create_builder_config() as config:

        print(f"Parsing ONNX file: {onnx_file_path}")
        with open(onnx_file_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                print("ERROR: Failed to parse ONNX model")
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                return None
        print("ONNX model parsed successfully.")

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            input = network.get_input(i)
            if (input.name == "image"):
                min_shape = (1,3,480,848)
                opt_shape = min_shape
                max_shape = min_shape
                profile.set_shape(input.name, min_shape, opt_shape, max_shape)
            # if (input.name == "num_tokens"):
            #     min_shape = [1200]
            #     opt_shape = min_shape
            #     max_shape = min_shape
            #     profile.set_shape_input(input.name, min_shape, opt_shape, max_shape)
            if input.name == "depth":
                min_shape = (1,1,480,848)
                opt_shape = min_shape
                max_shape = min_shape
                profile.set_shape(input.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        config.set_flag(trt.BuilderFlag.BF16)

        print("Building TensorRT engine. This may take a while...")
        engine = builder.build_engine_with_config(network, config)  # <--- Use this method

        if engine is None:
            print("ERROR: Engine build failed!")
            return None

        with open(engine_file_path, 'wb') as f:
            f.write(engine.serialize())
        print(f"TensorRT engine saved to: {engine_file_path}")

        return engine

# Usage
engine = build_engine("./model.onnx", "model.trt")