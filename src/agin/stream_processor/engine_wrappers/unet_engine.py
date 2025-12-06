from agin.stream_processor.engine_wrappers.engine import Engine
import torch

class UnetEngine:
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

        self.num_attention_tokens_in_lowest_block = (self.latent_height // 4) * (self.latent_width // 4)
        self.num_attention_tokens_in_middle_block = (self.latent_height // 2) * (self.latent_width // 2)


        self.allocate_buffers(self.prev_batch_size)



    def allocate_buffers(self, batch_size):
        self.engine.allocate_buffers(
            shape_dict={
                "sample" : (batch_size, 4, self.latent_height, self.latent_width),
                "timestep" : (batch_size,),
                "encoder_hidden_states" : (batch_size, 77, 2048),
                "down_block_0" : (batch_size, 320, self.latent_height, self.latent_width),
                "down_block_1" : (batch_size, 320, self.latent_height, self.latent_width),
                "down_block_2" : (batch_size, 320, self.latent_height, self.latent_width),
                "down_block_3" : (batch_size, 320, self.latent_height // 2, self.latent_width // 2),
                "down_block_4" : (batch_size, 640, self.latent_height // 2, self.latent_width // 2),
                "down_block_5" : (batch_size, 640, self.latent_height // 2, self.latent_width // 2),
                "down_block_6" : (batch_size, 640, self.latent_height // 4, self.latent_width // 4),
                "down_block_7" : (batch_size, 1280, self.latent_height // 4, self.latent_width // 4),
                "down_block_8" : (batch_size, 1280, self.latent_height // 4, self.latent_width // 4),
                "mid_block" : (batch_size, 1280, self.latent_height // 4, self.latent_width // 4),
                "text_embeds" : (batch_size, 1280),
                "time_ids" : (batch_size, 6),
                "extended_attention_1_key" : (batch_size, 30, 20, self.num_attention_tokens_in_lowest_block, 64),
                "extended_attention_1_value" : (batch_size, 30, 20, self.num_attention_tokens_in_lowest_block, 64),
                "extended_attention_2_key" : (batch_size, 6, 10, self.num_attention_tokens_in_middle_block, 64),
                "extended_attention_2_value" : (batch_size, 6, 10, self.num_attention_tokens_in_middle_block, 64),
                
                "output" : (batch_size, 4, self.latent_height, self.latent_width),
                "out_extended_attention_1_key" : (batch_size, 30, 20, self.num_attention_tokens_in_lowest_block, 64),
                "out_extended_attention_1_value" : (batch_size, 30, 20, self.num_attention_tokens_in_lowest_block, 64),
                "out_extended_attention_2_key" : (batch_size, 6, 10, self.num_attention_tokens_in_middle_block, 64),
                "out_extended_attention_2_value" : (batch_size, 6, 10, self.num_attention_tokens_in_middle_block, 64)
            },
            device="cuda",
        )
        
    def __call__(self,
                 sample,
                 timestep,
                 encoder_hidden_states,
                 down_block_0,
                 down_block_1,
                 down_block_2,
                 down_block_3,
                 down_block_4,
                 down_block_5,
                 down_block_6,
                 down_block_7,
                 down_block_8,
                 mid_block,
                 text_embeds,
                 time_ids,
                 extended_attention_1_key,
                 extended_attention_1_value,
                 extended_attention_2_key,
                 extended_attention_2_value):

        batch_size = sample.shape[0]

        if batch_size != self.prev_batch_size:
            self.allocate_buffers(batch_size)
            self.prev_batch_size = batch_size

        infer_outputs = self.engine.infer(
            {"sample": sample.to("cuda", torch.float16),
            "timestep": timestep.to("cuda", torch.float16),
            "encoder_hidden_states": encoder_hidden_states.to("cuda", torch.float16),
            "down_block_0": down_block_0.to("cuda", torch.float16),
            "down_block_1": down_block_1.to("cuda", torch.float16),
            "down_block_2": down_block_2.to("cuda", torch.float16),
            "down_block_3": down_block_3.to("cuda", torch.float16),
            "down_block_4": down_block_4.to("cuda", torch.float16),
            "down_block_5": down_block_5.to("cuda", torch.float16),
            "down_block_6": down_block_6.to("cuda", torch.float16),
            "down_block_7": down_block_7.to("cuda", torch.float16),
            "down_block_8": down_block_8.to("cuda", torch.float16),
            "mid_block": mid_block.to("cuda", torch.float16),
            "text_embeds": text_embeds.to("cuda", torch.float16),
            "time_ids": time_ids.to("cuda", torch.float16),
            "extended_attention_1_key": extended_attention_1_key.to("cuda", torch.float16),
            "extended_attention_1_value": extended_attention_1_value.to("cuda", torch.float16),
            "extended_attention_2_key": extended_attention_2_key.to("cuda", torch.float16),
            "extended_attention_2_value": extended_attention_2_value.to("cuda", torch.float16)},
            self.cuda_stream
        )

        noise = infer_outputs["output"]
        extended_attention_1_key = infer_outputs["out_extended_attention_1_key"]
        extended_attention_1_value = infer_outputs["out_extended_attention_1_value"]
        extended_attention_2_key = infer_outputs["out_extended_attention_2_key"]
        extended_attention_2_value = infer_outputs["out_extended_attention_2_value"]

        return noise, extended_attention_1_key, extended_attention_1_value, extended_attention_2_key, extended_attention_2_value