from agin.stream_processor.engine_wrappers.engine import Engine
import torch


class InterpolationModelEngine:
    def __init__(self, engine_path, cuda_stream, resolution):
        self.cuda_stream = cuda_stream

        self.engine = Engine(engine_path)
        self.engine.load()
        self.engine.activate()

        self.height = resolution["height"]
        self.width = resolution["width"]

        self.engine.allocate_buffers(
            shape_dict={
                "images": [1, 6, self.height, self.width],
                "interpolated_image": [1, 3, self.height, self.width],
            },
            device="cuda",
        )
        
    def __call__(self, latent):
        images = self.engine.infer(
            {"images": latent.to("cuda", torch.float16)},
            self.cuda_stream
        )["interpolated_image"]

        return images