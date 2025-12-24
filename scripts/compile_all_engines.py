from agin.engine_compilation_tools.compile_encoder import compile_encoder
from agin.engine_compilation_tools.compile_controlnet import compile_controlnet
from agin.engine_compilation_tools.compile_unet import compile_unet
from agin.engine_compilation_tools.compile_vae_decoder import compile_vae_decoder
from agin.engine_compilation_tools.compile_taesdxl_decoder import compile_taesdxl_decoder
from agin.engine_compilation_tools.compile_interpolation_model import compile_interpolation_model


def compile_all_engines():
    print("Compiling encoder...")
    compile_encoder()
    print("Compiling controlnet...")
    compile_controlnet()
    print("Compiling unet...")
    compile_unet()
    print("Compiling vae decoder...")
    compile_vae_decoder()
    print("Compiling taesdxl decoder...")
    compile_taesdxl_decoder()
    print("Compiling interpolation model...")
    compile_interpolation_model()

    print("Each compilation script executed normally.")
    print("Done.")

if __name__ == "__main__":
    compile_all_engines()