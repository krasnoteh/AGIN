from diffusers import AutoencoderTiny

from safetensors.torch import load_file
import torch
import tensorrt as trt
import os

from agin.engine_compilation_tools.config import Config

def compile_taesdxl_decoder():
    # Step 0: prepare

    config = Config.from_json("configs/engine_compiler_config.json")
    torch_dtype = torch.float16
    onnx_model_filename = "decoder.onnx"
    engine_path = config.engine_save_path / "taesdxl_decoder.engine"


    if os.path.isfile(engine_path):
        print("Engine", engine_path, "already exists.")
        return

    # Step 1: load model as torch model

    taesdxl_config = AutoencoderTiny.load_config(config.path_to_models / "taesdxl/config.json")
    taesdxl = AutoencoderTiny.from_config(taesdxl_config).to("cuda", torch_dtype)
    taesdxl.load_state_dict(load_file(config.path_to_models / "taesdxl/weights.safetensors", device="cuda"))

    decoder = taesdxl.decoder
    decoder = decoder.eval()

    # Step 2: save it as onnx model

    latent_height = config.height // 8
    latent_width = config.width // 8

    dummy_input = torch.randn(1, 4, latent_height, latent_width).to("cuda", torch_dtype)

    torch.onnx.export(
        decoder,
        dummy_input,
        onnx_model_filename,
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

    del decoder
    torch.cuda.empty_cache()

    # Step 3: configure engine

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

    os.remove(onnx_model_filename)

if __name__ == "__main__":
    compile_taesdxl_decoder()