# Not so optimized, just for the tests.
# Loads whole StableDiffusionXLControlNetPipeline with all models to apply lora to it.
# But this is compatible with probably all loras for SDXL.

# Note that even in tensorrt engine unet with lora is usually a bit slower than default unet.

# Note that if your lora depents on text encoders and has weights for them, they would be ignored.
# In this case you would probably need to load lora to the text encoders too, see agin/stream_processor/subprocesses/model_inference_subprocess.py.

import torch
from safetensors.torch import load_file
import onnx
from onnx.external_data_helper import convert_model_to_external_data
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from diffusers import UNet2DConditionModel,  AutoencoderTiny, ControlNetModel
from diffusers import  UNet2DConditionModel, StableDiffusionXLControlNetPipeline
from diffusers import EulerAncestralDiscreteScheduler
import tensorrt as trt
import os
import shutil


from agin.engine_compilation_tools.config import Config
from agin.engine_compilation_tools.unet_model import UNet2DConditionModel


def compile_lora_unet():
    # Step 0: prepare

    lora_name = "your_lora.safetensors"

    config = Config.from_json("configs/engine_compiler_config.json")
    torch_dtype = torch.float16
    onnx_model_filename = "unet.onnx"
    onnx_model_data_filename = "unet.data"
    path_to_onnx_files = "unet_onnx_files"
    engine_path = config.engine_save_path / "your_lora_unet.engine"

    if os.path.isfile(engine_path):
        print("Engine", engine_path, "already exists.")
        return

    # Step 1: load model as torch model

    unet = UNet2DConditionModel.from_config(os.path.join(config.path_to_models, "unet/config.json"), device="cpu")

    state_dict = load_file(
        os.path.join(config.path_to_models, "unet/diffusion_pytorch_model.fp16.safetensors"),
        device="cpu"
    )

    unet.load_state_dict(state_dict)

    unet = unet.to(torch.float16)
    unet = unet.to("cuda")

    taesdxl_config = AutoencoderTiny.load_config(config.path_to_models / "taesdxl/config.json")
    taesdxl = AutoencoderTiny.from_config(taesdxl_config).to("cuda", torch.float16)
    taesdxl.load_state_dict(load_file(config.path_to_models / "taesdxl/weights.safetensors", device="cuda"))

    text_encoder = CLIPTextModel.from_pretrained(config.path_to_models / "text_encoder").to("cuda", torch.float16)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(config.path_to_models / "text_encoder_2").to("cuda", torch.float16)

    tokenizer = CLIPTokenizer.from_pretrained(config.path_to_models / "tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(config.path_to_models / "tokenizer_2")

    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(config.path_to_models / "scheduler")

    controlnet = ControlNetModel.from_pretrained(
        config.path_to_models / "controlnet",
        torch_dtype=torch.float16
    ).to("cuda", torch.float16)

    pipe = StableDiffusionXLControlNetPipeline(
        vae=taesdxl,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=scheduler,
        feature_extractor=None,
        controlnet = controlnet
    )

    pipe.load_lora_weights(config.path_to_models / "loras" / lora_name)

    # Step 2: save it as onnx model (as many files)

    os.makedirs(path_to_onnx_files, exist_ok=True)

    latent_height = config.height // 8
    latent_width = config.width // 8

    def generate_onnx_sample_input(batch_size=1):
        num_attention_tokens_in_lowest_block = (latent_height // 4) * (latent_width // 4)
        num_attention_tokens_in_middle_block = (latent_height // 2) * (latent_width // 2)

        inputs = (
            torch.randn(batch_size, 4, latent_height, latent_width, device="cuda", dtype=torch_dtype),  # sample
            torch.randn(batch_size, device="cuda", dtype=torch_dtype),               # timestep
            torch.randn(batch_size, 77, 2048, device="cuda", dtype=torch_dtype),      # encoder_hidden_states
            
            *[
                torch.randn(batch_size, 320, latent_height, latent_width, device="cuda", dtype=torch_dtype),   # down_block_0
                torch.randn(batch_size, 320, latent_height, latent_width, device="cuda", dtype=torch_dtype),   # down_block_1
                torch.randn(batch_size, 320, latent_height, latent_width, device="cuda", dtype=torch_dtype),   # down_block_2
                torch.randn(batch_size, 320, latent_height // 2, latent_width // 2, device="cuda", dtype=torch_dtype),     # down_block_3
                torch.randn(batch_size, 640, latent_height // 2, latent_width // 2, device="cuda", dtype=torch_dtype),     # down_block_4
                torch.randn(batch_size, 640, latent_height // 2, latent_width // 2, device="cuda", dtype=torch_dtype),     # down_block_5
                torch.randn(batch_size, 640, latent_height // 4, latent_width // 4, device="cuda", dtype=torch_dtype),     # down_block_6
                torch.randn(batch_size, 1280, latent_height // 4, latent_width // 4, device="cuda", dtype=torch_dtype),   # down_block_7
                torch.randn(batch_size, 1280, latent_height // 4, latent_width // 4, device="cuda", dtype=torch_dtype),   # down_block_8
            ],
            
            torch.randn(batch_size, 1280, latent_height // 4, latent_width // 4, device="cuda", dtype=torch_dtype),       # mid_block
            
            torch.randn(batch_size, 1280, device="cuda", dtype=torch_dtype),               # text_embeds
            torch.randn(batch_size, 6, device="cuda", dtype=torch_dtype),                 # time_ids

            torch.randn(batch_size, 30, 20, num_attention_tokens_in_lowest_block, 64, device="cuda", dtype=torch_dtype),  # extended_attention_1_key
            torch.randn(batch_size, 30, 20, num_attention_tokens_in_lowest_block, 64, device="cuda", dtype=torch_dtype),  # extended_attention_1_value

            torch.randn(batch_size, 6, 10, num_attention_tokens_in_middle_block, 64, device="cuda", dtype=torch_dtype),  # extended_attention_2_key
            torch.randn(batch_size, 6, 10, num_attention_tokens_in_middle_block, 64, device="cuda", dtype=torch_dtype),  # extended_attention_2_value
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
        os.path.join(path_to_onnx_files, onnx_model_filename),
        export_params=True,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        external_data = False
    )

    # Step 3: merge many onnx files into one

    onnx_model = onnx.load(os.path.join(path_to_onnx_files, onnx_model_filename))

    convert_model_to_external_data(
        onnx_model,
        all_tensors_to_one_file=True,  
        location=onnx_model_data_filename,
        size_threshold=0,              
        convert_attribute=False        
    )

    onnx.save_model(
        onnx_model,
        onnx_model_filename,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=onnx_model_data_filename,
        size_threshold=0,
    )

    del taesdxl
    del text_encoder
    del text_encoder_2
    del pipe
    del onnx_model
    del unet
    torch.cuda.empty_cache()

    # Step 4: configure engine

    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) 
    builder = trt.Builder(TRT_LOGGER)
    engine_config = builder.create_builder_config()
    cache = engine_config.create_timing_cache(b"")
    engine_config.set_timing_cache(cache, ignore_mismatch=False)


    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_model_filename, "rb") as f:
        if not parser.parse(f.read()):
            print(f"ERROR: Failed to parse the ONNX file {onnx_model_filename}")
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

    os.remove(onnx_model_filename)
    os.remove(onnx_model_data_filename)
    shutil.rmtree(path_to_onnx_files)

if __name__ == "__main__":
    compile_lora_unet()