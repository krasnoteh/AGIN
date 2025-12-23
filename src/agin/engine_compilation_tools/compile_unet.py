import torch
from safetensors.torch import load_file
import onnx
from onnx.external_data_helper import convert_model_to_external_data
import tensorrt as trt
import os
import shutil


from dependencies.app_config import AppConfig
from dependencies.unet_model import UNet2DConditionModel

# Step 0: prepare

use_force_attention = False

config = AppConfig.from_json("config.json")

engine_path = config.engine_save_path / "unet.engine"
if os.path.isfile(engine_path):
    print("Engine", engine_path, "already exists.")
    exit()

# Step 1: load model as torch model

unet_config = UNet2DConditionModel.load_config(config.path_to_models / "unet_turbo/config.json")
unet = UNet2DConditionModel.from_config(unet_config).to("cuda", torch.float16)
unet.load_state_dict(load_file(config.path_to_models / "unet_turbo/diffusion_pytorch_model.fp16.safetensors", device="cuda"))

# Step 2: save it as onnx model (as many files)

path_to_onnx_files = config.path_to_temp_onnx_models / "unet_onnx_files"

os.makedirs(path_to_onnx_files, exist_ok=True)

latent_height = config.height // 8
latent_width = config.width // 8


def generate_onnx_sample_input(batch_size=1):
    num_attention_tokens_in_lowest_block = (latent_height // 4) * (latent_width // 4)
    num_attention_tokens_in_middle_block = (latent_height // 2) * (latent_width // 2)

    inputs = (
        torch.randn(batch_size, 4, latent_height, latent_width, device="cuda", dtype=torch.float16),  # sample
        torch.randn(batch_size, device="cuda", dtype=torch.float16),               # timestep
        torch.randn(batch_size, 77, 2048, device="cuda", dtype=torch.float16),      # encoder_hidden_states
        
        *[
            torch.randn(batch_size, 320, latent_height, latent_width, device="cuda", dtype=torch.float16),   # down_block_0
            torch.randn(batch_size, 320, latent_height, latent_width, device="cuda", dtype=torch.float16),   # down_block_1
            torch.randn(batch_size, 320, latent_height, latent_width, device="cuda", dtype=torch.float16),   # down_block_2
            torch.randn(batch_size, 320, latent_height // 2, latent_width // 2, device="cuda", dtype=torch.float16),     # down_block_3
            torch.randn(batch_size, 640, latent_height // 2, latent_width // 2, device="cuda", dtype=torch.float16),     # down_block_4
            torch.randn(batch_size, 640, latent_height // 2, latent_width // 2, device="cuda", dtype=torch.float16),     # down_block_5
            torch.randn(batch_size, 640, latent_height // 4, latent_width // 4, device="cuda", dtype=torch.float16),     # down_block_6
            torch.randn(batch_size, 1280, latent_height // 4, latent_width // 4, device="cuda", dtype=torch.float16),   # down_block_7
            torch.randn(batch_size, 1280, latent_height // 4, latent_width // 4, device="cuda", dtype=torch.float16),   # down_block_8
        ],
        
        torch.randn(batch_size, 1280, latent_height // 4, latent_width // 4, device="cuda", dtype=torch.float16),       # mid_block
        
        torch.randn(batch_size, 1280, device="cuda", dtype=torch.float16),               # text_embeds
        torch.randn(batch_size, 6, device="cuda", dtype=torch.float16),                 # time_ids

        torch.randn(batch_size, 30, 20, num_attention_tokens_in_lowest_block, 64, device="cuda", dtype=torch.float16),  # extended_attention_1_key
        torch.randn(batch_size, 30, 20, num_attention_tokens_in_lowest_block, 64, device="cuda", dtype=torch.float16),  # extended_attention_1_value

        torch.randn(batch_size, 6, 10, num_attention_tokens_in_middle_block, 64, device="cuda", dtype=torch.float16),  # extended_attention_2_key
        torch.randn(batch_size, 6, 10, num_attention_tokens_in_middle_block, 64, device="cuda", dtype=torch.float16),  # extended_attention_2_value
    )

    input_names = [
        "sample",
        "timestep",
        "encoder_hidden_states",
        *[f"down_block_{i}" for i in range(9)],
        "mid_block",
        "text_embeds",
        "time_ids",
        "extended_attention_1_key",
        "extended_attention_1_value",
        "extended_attention_2_key",
        "extended_attention_2_value"
    ]

    return inputs, input_names




inputs, input_names = generate_onnx_sample_input()
output_names = [
        "output",     
        "out_extended_attention_1_key",
        "out_extended_attention_1_value",
        "out_extended_attention_2_key",
        "out_extended_attention_2_value"
        ]

dynamic_axes = {name: {0: "batch"} for name in input_names}

for i in output_names:
    dynamic_axes[i] = {0: "batch"}

torch.onnx.export(
    unet,
    inputs,
    str(path_to_onnx_files / "unet.onnx"),
    export_params=True,
    do_constant_folding=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    opset_version=17,
    external_data = False
)

# Step 3: merge many onnx files into one

onnx_model = onnx.load(path_to_onnx_files / "unet.onnx")

convert_model_to_external_data(
    onnx_model,
    all_tensors_to_one_file=True,  
    location="unet.data",
    size_threshold=0,              
    convert_attribute=False        
)

onnx.save_model(
    onnx_model,
    "unet.onnx",
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="unet.data",
    size_threshold=0,
)

del onnx_model
del unet

# Step 4: configure engine

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) 
builder = trt.Builder(TRT_LOGGER)
engine_config = builder.create_builder_config()
cache = engine_config.create_timing_cache(b"")
engine_config.set_timing_cache(cache, ignore_mismatch=False)


flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(flag)
parser = trt.OnnxParser(network, TRT_LOGGER)

path_onnx_model = "unet.onnx"
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

os.remove("unet.onnx")
os.remove("unet.data")

if config.delete_onnx_models_after_compilation:
    shutil.rmtree(config.path_to_temp_onnx_models)
