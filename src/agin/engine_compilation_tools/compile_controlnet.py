import torch
import onnx
from onnx.external_data_helper import convert_model_to_external_data
import tensorrt as trt
import os
import shutil

from dependencies.app_config import AppConfig
from dependencies.controlnet_model import ControlNetModel

# Step 0: prepare

config = AppConfig.from_json("config.json")

engine_path = config.engine_save_path / "detail_controlnet.engine"
if os.path.isfile(engine_path):
    print("Engine", engine_path, "already exists.")
    exit()

# Step 1: load model as torch model

controlnet = ControlNetModel.from_pretrained(
    config.path_to_models / "lineart_controlnet",
    torch_dtype=config.torch_dtype
).to("cuda", config.torch_dtype)

# Step 2: save it as onnx model (as many files)

path_to_onnx_files = config.path_to_temp_onnx_models / "controlnet_onnx_files"

os.makedirs(path_to_onnx_files, exist_ok=True)

latent_height = config.height // 8
latent_width = config.width // 8

def generate_onnx_sample_input(batch_size=1):
    inputs = (
        torch.randn(batch_size, 4, latent_height, latent_width, device="cuda", dtype=config.torch_dtype), 
        torch.randn(batch_size, device="cuda", dtype=config.torch_dtype),  
        torch.randn(batch_size, 77, 2048, device="cuda", dtype=config.torch_dtype), 
        torch.randn(batch_size, 3, config.height, config.width, device="cuda", dtype=config.torch_dtype),  
        torch.randn(batch_size, 1280, device="cuda", dtype=config.torch_dtype), 
        torch.randn(batch_size, 6, device="cuda", dtype=config.torch_dtype), 
    )
    
    input_names = [
        "sample",
        "timestep",
        "encoder_hidden_states",
        "controlnet_cond",
        "text_embeds",
        "time_ids"
    ]

    return inputs, input_names

inputs, input_names = generate_onnx_sample_input()

dynamic_axes = {name: {0: "batch"} for name in input_names}
output_names = ["output_" + str(i) for i in range(0, 10)]

for i in output_names:
    dynamic_axes[i] = {0: "batch"}

torch.onnx.export(
    controlnet,
    inputs,
    str(path_to_onnx_files / "controlnet.onnx"),
    export_params=True,
    do_constant_folding=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    opset_version=17,
    external_data = False,
)

# Step 3: merge many onnx files into one

onnx_model = onnx.load(path_to_onnx_files / "controlnet.onnx")

convert_model_to_external_data(
    onnx_model,
    all_tensors_to_one_file=True,  
    location="controlnet.data",
    size_threshold=0,              
    convert_attribute=False        
)

onnx.save_model(
    onnx_model,
    "controlnet.onnx",
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="controlnet.data",
    size_threshold=0,
)

del onnx_model
del controlnet

# Step 4: configure engine

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) 
builder = trt.Builder(TRT_LOGGER)
engine_config = builder.create_builder_config()
cache = engine_config.create_timing_cache(b"")
engine_config.set_timing_cache(cache, ignore_mismatch=False)


flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(flag)
parser = trt.OnnxParser(network, TRT_LOGGER)

path_onnx_model = "controlnet.onnx"
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


max_batch_size = 4
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

# Step 5: compile engine

engine_bytes = builder.build_serialized_network(network, engine_config) 

with open(engine_path, "wb") as f:    
    f.write(engine_bytes)


# Step 6: post-compilation cleanup

os.remove("controlnet.onnx")
os.remove("controlnet.data")

if config.delete_onnx_models_after_compilation:
    shutil.rmtree(config.path_to_temp_onnx_models)
