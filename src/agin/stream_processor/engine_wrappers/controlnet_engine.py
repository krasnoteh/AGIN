from agin.stream_processor.engine_wrappers.engine import Engine
import torch


class ControlnetEngine:
    def __init__(self, engine_path, cuda_stream, resolution):
        self.cuda_stream = cuda_stream

        self.engine = Engine(engine_path)
        self.engine.load()
        self.engine.activate()
        self.prev_batch_size = 1

        self.height = resolution["height"]
        self.width = resolution["width"]

        self.latent_height = self.height // 8
        self.latent_width = self.width // 8

        
        self.allocate_buffers(self.prev_batch_size)



    def allocate_buffers(self, batch_size):
        self.engine.allocate_buffers(
            shape_dict={
                "sample" : (batch_size, 4, self.latent_height, self.latent_width),
                "timestep" : (batch_size,),
                "encoder_hidden_states" : (batch_size, 77, 2048),
                "controlnet_cond" : (batch_size, 3, self.height, self.width),
                "text_embeds" : (batch_size, 1280),
                "time_ids" : (batch_size, 6),
                "output_0" : (batch_size, 320, self.latent_height, self.latent_width),
                "output_1" : (batch_size, 320, self.latent_height, self.latent_width),
                "output_2" : (batch_size, 320, self.latent_height, self.latent_width),
                "output_3" : (batch_size, 320, self.latent_height // 2, self.latent_width // 2),
                "output_4" : (batch_size, 640, self.latent_height // 2, self.latent_width // 2),
                "output_5" : (batch_size, 640, self.latent_height // 2, self.latent_width // 2),
                "output_6" : (batch_size, 640, self.latent_height // 4, self.latent_width // 4),
                "output_7" : (batch_size, 1280, self.latent_height // 4, self.latent_width // 4),
                "output_8" : (batch_size, 1280, self.latent_height // 4, self.latent_width // 4),
                "output_9" : (batch_size, 1280, self.latent_height // 4, self.latent_width // 4),
            },
            device="cuda",
        )
        
    def __call__(self,
                 sample,
                 timestep,
                 encoder_hidden_states,
                 controlnet_cond,
                 text_embeds,
                 time_ids):

        batch_size = sample.shape[0]

        if batch_size != self.prev_batch_size:
            self.allocate_buffers(batch_size)
            self.prev_batch_size = batch_size
        
        connections = self.engine.infer(
            {"sample": sample.to("cuda", torch.float16),
            "timestep": timestep.to("cuda", torch.float16),
            "encoder_hidden_states": encoder_hidden_states.to("cuda", torch.float16),
            "controlnet_cond": controlnet_cond.to("cuda", torch.float16),
            "text_embeds": text_embeds.to("cuda", torch.float16),
            "time_ids": time_ids.to("cuda", torch.float16)},
            self.cuda_stream
        )

        out_connections = [connections["output_" + str(i)] for i in range(0, 10)]

        return out_connections