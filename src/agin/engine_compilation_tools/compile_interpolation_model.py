from diffusers import AutoencoderTiny
from dependencies.interpolation_model import IFNet
from safetensors.torch import load_file
import torch
import tensorrt as trt
import shutil
import os

from dependencies.app_config import AppConfig

# Step 0: prepare

config = AppConfig.from_json("config.json")

engine_path = config.engine_save_path / "interpolation_model.engine"
if os.path.isfile(engine_path):
    print("Engine", engine_path, "already exists.")
    exit()

# Step 1: load model as torch model

def convert(param):
    return {
        k.replace("module.", ""): v
        for k, v in param.items()
        if "module." in k
    }

device = "cuda"
interpolation_model = IFNet()
interpolation_model.load_state_dict(convert(torch.load(config.path_to_models / "interpolation_model/flownet.pkl")))
interpolation_model.to(device, torch.float16)
interpolation_model.eval()

# Step 2: save it as onnx model

torch_dtype = torch.float16

images = torch.randn(1, 6, config.height, config.width).to("cuda", config.torch_dtype)

torch.onnx.export(
    interpolation_model,
    images,
    config.path_to_temp_onnx_models / "interpolation_model.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["images"],
    output_names=["interpolated_image"],
)

# Step 3: configure engine

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) 
builder = trt.Builder(TRT_LOGGER)
engine_config = builder.create_builder_config()
cache = engine_config.create_timing_cache(b"")
engine_config.set_timing_cache(cache, ignore_mismatch=False)


flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(flag)
parser = trt.OnnxParser(network, TRT_LOGGER)

path_onnx_model = config.path_to_temp_onnx_models / "interpolation_model.onnx"
with open(path_onnx_model, "rb") as f:
    if not parser.parse(f.read()):
        print(f"ERROR: Failed to parse the ONNX file {path_onnx_model}")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
            
inputs = [network.get_input(i) for i in range(network.num_inputs)]
outputs = [network.get_output(i) for i in range(network.num_outputs)]

for input in inputs:
    print(input.name, input.shape)
for output in outputs:
    print(output.name, output.shape)


max_batch_size = 1
profile = builder.create_optimization_profile()

for i in range(network.num_inputs):
    input = network.get_input(i)
    input_dims = list(input.shape)
    
    # Extract static dimensions (all except batch dimension)
    static_dims = input_dims[1:]  # Remove batch dimension
    
    # Set dynamic batch dimensions for min/opt/max
    min_shape = [1] + static_dims
    opt_shape = [max_batch_size] + static_dims
    max_shape = [max_batch_size] + static_dims
    
    profile.set_shape(input.name, min_shape, opt_shape, max_shape)

engine_config.add_optimization_profile(profile)

engine_config.set_flag(trt.BuilderFlag.FP16)

# Step 4: compile engine

engine_bytes = builder.build_serialized_network(network, engine_config) 
with open(engine_path, "wb") as f:    
    f.write(engine_bytes)

# Step 5: post-compilation cleanup

if config.delete_onnx_models_after_compilation:
    shutil.rmtree(config.path_to_temp_onnx_models)