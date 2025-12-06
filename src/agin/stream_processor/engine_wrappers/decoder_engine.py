from agin.stream_processor.engine_wrappers.engine import Engine
import torch


class DecoderEngine:
    def __init__(self, engine_path, cuda_stream, resolution, dtype = torch.float16):
        self.cuda_stream = cuda_stream
        self.dtype = dtype

        self.engine = Engine(engine_path, dtype=dtype)
        self.engine.load()
        self.engine.activate()

        self.height = resolution["height"]
        self.width = resolution["width"]

        self.latent_height = self.height // 8
        self.latent_width = self.width // 8

        self.engine.allocate_buffers(
            shape_dict={
                "input": [1, 4, self.latent_height, self.latent_width],
                "output": [1, 3, self.height, self.width],
            },
            device="cuda",
        )
        
    def __call__(self, latent):
        images = self.engine.infer(
            {"input": latent.to("cuda", self.dtype)},
            self.cuda_stream
        )["output"]

        return images