from agin.stream_processor.engine_wrappers.decoder_engine import DecoderEngine
from polygraphy import cuda
import torch
import time


def main():
    resolution = {
        "height" : 1344,
        "width" : 768
    }
    engine_path = "engines/portrait_engines/full_decoder.engine"
    cuda_stream = cuda.Stream()
    decoder_engine = DecoderEngine(engine_path, cuda_stream, resolution)

    num_checks = 100

    #warmup
    for i in range(num_checks):
        input_tensor = torch.normal(0, 1, (1, 4, resolution["height"] // 8, resolution["width"] // 8),
                                 device="cuda", dtype=torch.float16)
        output_tensor = decoder_engine(input_tensor)
        torch.cuda.synchronize()


    start_time = time.time()
    for i in range(num_checks):
        input_tensor = torch.normal(0, 1, (1, 4, resolution["height"] // 8, resolution["width"] // 8),
                                 device="cuda", dtype=torch.float16)
        output_tensor = decoder_engine(input_tensor)
        torch.cuda.synchronize()

    print((time.time() - start_time) / num_checks)
    print(output_tensor.max())

if __name__ == "__main__":
    main()
