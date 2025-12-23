from diffusers import AutoencoderKL

from safetensors.torch import load_file
import torch
import tensorrt as trt
import shutil
import os
import torch.nn as nn

from dependencies.app_config import AppConfig

# Step 0: prepare

config = AppConfig.from_json("config.json")

engine_path = config.engine_save_path / "full_decoder.engine"
if os.path.isfile(engine_path):
    print("Engine", engine_path, "already exists.")
    exit()

# Step 1: load model as torch model

vae_config = AutoencoderKL.load_config(config.path_to_models / "vae/config.json")
vae = AutoencoderKL.from_config(vae_config).to("cuda", config.torch_dtype)
vae.load_state_dict(load_file(config.path_to_models / "vae/weights.safetensors", device="cuda"))

class PostQuantDecoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.decoder = vae.decoder
        self.post_quant_conv = vae.post_quant_conv
        self.scaling_factor = vae.config.scaling_factor

    def forward(self, latent):
        latent = latent / self.scaling_factor
        latent = self.post_quant_conv(latent)
        image = self.decoder(latent)

        return image

decoder = PostQuantDecoder(vae)
decoder = decoder.eval()

# Step 2: save it as onnx model

latent_height = config.height // 8
latent_width = config.width // 8

dummy_input = torch.randn(1, 4, latent_height, latent_width).to("cuda", config.torch_dtype)

torch.onnx.export(
    decoder,
    dummy_input,
    config.path_to_temp_onnx_models / "big_decoder.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},  # Dynamic batch size
        "output": {0: "batch_size"},
    },
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

path_onnx_model = config.path_to_temp_onnx_models / "big_decoder.onnx"
with open(path_onnx_model, "rb") as f:
    if not parser.parse(f.read()):
        print(f"ERROR: Failed to parse the ONNX file {path_onnx_model}")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
            
inputs = [network.get_input(i) for i in range(network.num_inputs)]
outputs = [network.get_output(i) for i in range(network.num_outputs)]


for input in inputs:
    print(f"Model {input.name} shape: {input.shape} {input.dtype}")
for output in outputs:
    print(f"Model {output.name} shape: {output.shape} {output.dtype}") 

max_batch_size = 1

profile = builder.create_optimization_profile()

for input in inputs:
    input_shape = input.shape 
    min_batch = 1
    opt_batch = max_batch_size 
    max_batch = max_batch_size

    min_shape = [min_batch] + [4, latent_height, latent_width]
    opt_shape = [opt_batch] + [4, latent_height, latent_width]
    max_shape = [max_batch] + [4, latent_height, latent_width]

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