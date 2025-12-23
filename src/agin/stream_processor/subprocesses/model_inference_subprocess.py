from agin.stream_processor.pipelines.stable_diffusion_xl_controlnet_pipeline import StableDiffusionXLControlNetPipeline
from agin.stream_processor.engine_wrappers.decoder_engine import DecoderEngine
from agin.stream_processor.engine_wrappers.encoder_engine import EncoderEngine
from agin.stream_processor.engine_wrappers.unet_engine import UnetEngine
from agin.stream_processor.engine_wrappers.controlnet_engine import ControlnetEngine
from agin.stream_processor.engine_wrappers.interpolation_model_engine import InterpolationModelEngine

from diffusers import EulerAncestralDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from polygraphy import cuda
import numpy as np
import torch
import time
import os

from multiprocessing import Process, Value, Manager
from copy import deepcopy
import cv2
from queue import Empty

from agin.utils.shared_tensor import SharedTensor

class PipeCache:
    def __init__(self):
        self.prompt_cache = {}

class ExtendedAttentionHandler:
    def __init__(self, height, width):
        self.chains = {}

        self.latent_height = height // 8
        self.latent_width = width // 8

        self.num_attention_tokens_in_lowest_block = (self.latent_height // 4) * (self.latent_width // 4)
        self.num_attention_tokens_in_middle_block = (self.latent_height // 2) * (self.latent_width // 2)

    def make_zero_extention(self, batch_size: int):
        extended_attention_1_key = torch.zeros(batch_size, 30, 20, self.num_attention_tokens_in_lowest_block, 64, device="cuda:0", dtype=torch.float16)
        extended_attention_1_value = torch.zeros(batch_size, 30, 20, self.num_attention_tokens_in_lowest_block, 64, device="cuda:0", dtype=torch.float16)
        extended_attention_2_key = torch.zeros(batch_size, 6, 10, self.num_attention_tokens_in_middle_block, 64, device="cuda:0", dtype=torch.float16)
        extended_attention_2_value = torch.zeros(batch_size, 6, 10, self.num_attention_tokens_in_middle_block, 64, device="cuda:0", dtype=torch.float16)

        extention = extended_attention_1_key, extended_attention_1_value, extended_attention_2_key, extended_attention_2_value

        return extention

    def update_chain(self, timestep : float, extention : tuple):
            self.chains[timestep] = extention

    def get_chain_extension(self, timestep : float) -> tuple:

        if timestep in self.chains:
            return self.chains[timestep]
        else:
            return self.make_zero_extention(1)
        
    def make_extensions_for_tasks(self, tasks):

        ext_att1_key_list, ext_att1_val_list = [], []
        ext_att2_key_list, ext_att2_val_list = [], []
        
        for task in tasks:
            timestep = float(task["timestep"].item())
            ext_att = self.get_chain_extension(timestep)
            ext_att1_key_list.append(ext_att[0])
            ext_att1_val_list.append(ext_att[1])
            ext_att2_key_list.append(ext_att[2])
            ext_att2_val_list.append(ext_att[3])

        extended_attention_1_key = torch.cat(ext_att1_key_list, dim=0)
        extended_attention_1_value = torch.cat(ext_att1_val_list, dim=0)
        extended_attention_2_key = torch.cat(ext_att2_key_list, dim=0)
        extended_attention_2_value = torch.cat(ext_att2_val_list, dim=0)

        extentions = extended_attention_1_key, extended_attention_1_value, extended_attention_2_key, extended_attention_2_value

        return extentions
    
    def update_chains_from_batch(self, extentions, tasks):
        
        batch_size = len(tasks)

        new_ea1_key, new_ea1_val, new_ea2_key, new_ea2_val = extentions

        new_ea1_key_chunks = new_ea1_key.chunk(batch_size, dim=0)
        new_ea1_val_chunks = new_ea1_val.chunk(batch_size, dim=0)
        new_ea2_key_chunks = new_ea2_key.chunk(batch_size, dim=0)
        new_ea2_val_chunks = new_ea2_val.chunk(batch_size, dim=0)
        
        for i, task in enumerate(tasks):
            timestep = float(task["timestep"].item())
            new_ext_att = (
                new_ea1_key_chunks[i],
                new_ea1_val_chunks[i],
                new_ea2_key_chunks[i],
                new_ea2_val_chunks[i]
            )
            self.update_chain(timestep, new_ext_att)


class ModelInferenceSubprocess:
    def __init__(self, config: dict, input_shared_tensor_name: str, output_batch_shared_tensor_name: str, pack_is_ready, last_processing_time):
        self.running = Value('b', False)
        self.process = None
        self.config = config
        self.height = self.config["resolution"]["height"]
        self.width = self.config["resolution"]["width"]
        self.resolution = self.config["resolution"]
        self.prompt = self.config["default_prompt"]
        self.input_shared_tensor_name = input_shared_tensor_name
        self.output_batch_shared_tensor_name = output_batch_shared_tensor_name
        self.pack_is_ready = pack_is_ready
        self.last_processing_time = last_processing_time

        manager = Manager()
        self.command_queue = manager.Queue()
        self.shared_state = manager.dict()

    def process_init(self):

        self.process_state = {
                "prompt" : self.config["default_prompt"],
                "steps" : self.config["default_steps"],
                "inversion_strength" : self.config["default_inversion_strength"],
                "controlnet_conditioning_scale" : self.config["default_controlnet_conditioning_scale"],
                "canny_low_threshold" : self.config["default_canny_low_threshold"],
                "canny_high_threshold" : self.config["default_canny_high_threshold"],
                "seed" : self.config["default_seed"]
            }

        self.input_shared_tensor = SharedTensor((self.resolution["height"], self.resolution["width"], 3), name=self.input_shared_tensor_name)
        self.output_batch_shared_tensor = SharedTensor((2, self.resolution["height"], self.resolution["width"], 3), name=self.output_batch_shared_tensor_name)

        self.pipelines = []

        models_path = self.config["models_path"]
        text_encoder_path = os.path.join(models_path, self.config["models_directories"]["text_encoder"])
        text_encoder_2_path = os.path.join(models_path, self.config["models_directories"]["text_encoder_2"])
        tokenizer_path = os.path.join(models_path, self.config["models_directories"]["tokenizer"])
        tokenizer_2_path = os.path.join(models_path, self.config["models_directories"]["tokenizer_2"])
        scheduler_path = os.path.join(models_path,  self.config["models_directories"]["scheduler"])

        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_path).to("cuda", torch.float16)
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(text_encoder_2_path).to("cuda", torch.float16)
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(tokenizer_2_path)
        self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(scheduler_path, timestep_spacing = "trailing")

        engines_path = self.config["engines_path"]
        decoder_engine_path = os.path.join(engines_path, self.config["engines_filenames"]["decoder"])
        encoder_engine_path = os.path.join(engines_path, self.config["engines_filenames"]["encoder"])
        unet_engine_path = os.path.join(engines_path, self.config["engines_filenames"]["unet"])
        controlnet_engine_path = os.path.join(engines_path, self.config["engines_filenames"]["controlnet"])
        interpolation_model_engine_path = os.path.join(engines_path, self.config["engines_filenames"]["interpolation_model"])

        cuda_stream = cuda.Stream()
        self.decoder_engine = DecoderEngine(decoder_engine_path, cuda_stream, self.resolution)
        self.encoder_engine = EncoderEngine(encoder_engine_path, cuda_stream, self.resolution)
        self.unet_engine = UnetEngine(unet_engine_path, cuda_stream, self.resolution)
        self.controlnet_engine = ControlnetEngine(controlnet_engine_path, cuda_stream, self.resolution)
        self.interpolation_model_engine = InterpolationModelEngine(interpolation_model_engine_path, cuda_stream, self.resolution)
        
        self.pipe_cache = PipeCache()
        self.ext_attention_handler = ExtendedAttentionHandler(self.height, self.width)

        self.previous_frame = None

    def start(self):
        self.running.value = True
        self.process = Process(target=self.process_main)
        self.process.start()

    def stop(self):
        self.running.value = False
        if self.process:
            self.process.join()

    def register_new_pipeline(self, image_input):
        new_sheduler = deepcopy(self.scheduler)

        pipe = StableDiffusionXLControlNetPipeline(
                encoder=self.encoder_engine,
                decoder=self.decoder_engine,
                text_encoder=self.text_encoder,
                text_encoder_2=self.text_encoder_2,
                tokenizer=self.tokenizer,
                tokenizer_2=self.tokenizer_2,
                scheduler=new_sheduler,
                seed=self.process_state["seed"]
        )

        color_conditioning_image = image_input
        color_conditioning_image = cv2.cvtColor(color_conditioning_image, cv2.COLOR_RGB2BGR)

        canny = cv2.Canny(image_input, self.process_state["canny_low_threshold"], self.process_state["canny_high_threshold"])
        canny = canny[:, :, None]
        canny_image = np.concatenate([canny, canny, canny], axis=2)

        pipe.execute_preparations(self.process_state['prompt'],
                                  num_inference_steps=self.process_state["steps"],
                                  controlnet_image = canny_image,
                                  color_conditioning_image = color_conditioning_image,
                                  inversion_strength = self.process_state["inversion_strength"],
                                  controlnet_conditioning_scale= self.process_state["controlnet_conditioning_scale"],
                                  pipe_cache = self.pipe_cache)

        self.pipelines.append(pipe)

    @torch.no_grad()
    def step_all_pipelines(self):
        if not self.pipelines:
            return

        active_pipes = [pipe for pipe in self.pipelines if not pipe.latent_is_ready]
        tasks = [pipe.get_diffusion_task() for pipe in active_pipes]
        
        if not tasks:
            return

        def batch_tensors(key_fn):
            return torch.cat([key_fn(t) for t in tasks], dim=0)
        
        control_inputs = (
            batch_tensors(lambda t: t["sample"]),
            torch.stack([t["timestep"] for t in tasks]),
            batch_tensors(lambda t: t["encoder_hidden_states"]),
            batch_tensors(lambda t: t["controlnet_image"]),
            batch_tensors(lambda t: t["added_cond_kwargs"]['text_embeds']),
            batch_tensors(lambda t: t["added_cond_kwargs"]['time_ids'])
        )
        
        controlnet_connections = self.controlnet_engine(*control_inputs)

        conditioning_scales = torch.tensor(
            [t["controlnet_conditioning_scale"] for t in tasks],
            dtype=torch.float16,
            device="cuda"
        ).view(-1, 1, 1, 1)

        down_block = [
            (conn * conditioning_scales)
            for conn in controlnet_connections[:9]
        ]
        mid_block = (controlnet_connections[-1] * conditioning_scales)

        batch_size = len(tasks)
        
        extentions = self.ext_attention_handler.make_extensions_for_tasks(tasks)

        unet_inputs = (
            batch_tensors(lambda t: t["sample"]),
            torch.stack([t["timestep"] for t in tasks]),
            batch_tensors(lambda t: t["encoder_hidden_states"]),
            *down_block,
            mid_block,
            batch_tensors(lambda t: t["added_cond_kwargs"]['text_embeds']),
            batch_tensors(lambda t: t["added_cond_kwargs"]['time_ids']),
            *extentions
        )
        
        batched_output = self.unet_engine(*unet_inputs)
        noise, new_ea1_key, new_ea1_val, new_ea2_key, new_ea2_val = batched_output

        extention = new_ea1_key, new_ea1_val, new_ea2_key, new_ea2_val

        self.ext_attention_handler.update_chains_from_batch(extention, tasks)

        for pipe, result in zip(active_pipes, noise.chunk(batch_size, dim=0)):
            pipe.return_diffusion_result((result,))


    def get_image_from_ready_pipeline(self):

        for i in range(len(self.pipelines)):
            if self.pipelines[i].latent_is_ready:
                image = self.pipelines[i].decode_latent()
                self.pipelines.pop(i)
                return image
            
        return None
    
    def set_param(self, name: str, value) -> None:
        self.command_queue.put(("set_param", (name, value)))
    
    def update_process_state(self) -> None:
        """
        Called by the internal process
        """
        try:
            while True:
                cmd, payload = self.command_queue.get_nowait()
                if cmd == "set_param":
                    name, value = payload
                    self.process_state[name] = value
        except Empty:
            pass

    def process_main(self):
        self.process_init()

        prev_time = time.time()
        while self.running.value:

            self.update_process_state()

            frame = self.input_shared_tensor.array

            self.register_new_pipeline(frame)
            
            self.step_all_pipelines()
            
            image = self.get_image_from_ready_pipeline()

            if image is None:
                continue
        
            if self.previous_frame is None:
                self.previous_frame = image

            interpolated_image = self.interpolation_model_engine(torch.cat([self.previous_frame, image], dim=1))
            self.previous_frame = image

            frames_gpu = torch.cat([interpolated_image, image], dim=0)
            frames_cpu = (frames_gpu * 255).to(torch.uint8).permute(0, 2, 3, 1).to("cpu").numpy()
            frames_cpu_bgr = frames_cpu[..., ::-1]
            self.output_batch_shared_tensor.copy_from(frames_cpu_bgr)
            self.pack_is_ready.value = True

            processing_time = time.time() - prev_time
            prev_time = time.time()
            self.last_processing_time.value = processing_time
            print("fps", 1 / processing_time)
            