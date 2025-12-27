from agin.stream_processor.pipelines.sdxl_controlnet_pipeline import StableDiffusionXLControlNetPipeline
from agin.stream_processor.engine_wrappers.decoder_engine import DecoderEngine
from agin.stream_processor.engine_wrappers.encoder_engine import EncoderEngine
from agin.stream_processor.engine_wrappers.unet_engine import UnetEngine
from agin.stream_processor.engine_wrappers.controlnet_engine import ControlnetEngine
from agin.stream_processor.engine_wrappers.interpolation_model_engine import InterpolationModelEngine
from agin.stream_processor.extended_attention_handler import ExtendedAttentionHandler

from diffusers import EulerAncestralDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from polygraphy import cuda
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


    def init_process_state(self):
        self.process_state = {
            "prompt": self.config["default_prompt"],
            "steps": self.config["default_steps"],
            "inversion_strength": self.config["default_inversion_strength"],
            "controlnet_conditioning_scale": self.config["default_controlnet_conditioning_scale"],
            "canny_low_threshold": self.config["default_canny_low_threshold"],
            "canny_high_threshold": self.config["default_canny_high_threshold"],
            "seed": self.config["default_seed"],
        }

    def load_models(self):
        models_path = self.config["models_path"]
        dirs = self.config["models_directories"]

        self.text_encoder = CLIPTextModel.from_pretrained(
            os.path.join(models_path, dirs["text_encoder"])
        ).to("cuda", torch.float16)

        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            os.path.join(models_path, dirs["text_encoder_2"])
        ).to("cuda", torch.float16)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            os.path.join(models_path, dirs["tokenizer"])
        )

        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            os.path.join(models_path, dirs["tokenizer_2"])
        )

        self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            os.path.join(models_path, dirs["scheduler"]),
            timestep_spacing="trailing",
        )

    def load_engines(self):
        engines_path = self.config["engines_path"]
        names = self.config["engines_filenames"]

        cuda_stream = cuda.Stream()

        self.decoder_engine = DecoderEngine(
            os.path.join(engines_path, names["decoder"]),
            cuda_stream,
            self.resolution,
        )

        self.encoder_engine = EncoderEngine(
            os.path.join(engines_path, names["encoder"]),
            cuda_stream,
            self.resolution,
        )

        self.unet_engine = UnetEngine(
            os.path.join(engines_path, names["unet"]),
            cuda_stream,
            self.resolution,
        )

        self.controlnet_engine = ControlnetEngine(
            os.path.join(engines_path, names["controlnet"]),
            cuda_stream,
            self.resolution,
        )

        self.interpolation_model_engine = InterpolationModelEngine(
            os.path.join(engines_path, names["interpolation_model"]),
            cuda_stream,
            self.resolution,
        )

    def init_shared_tensors(self):
        h, w = self.resolution["height"], self.resolution["width"]

        self.input_shared_tensor = SharedTensor(
            (h, w, 3),
            name=self.input_shared_tensor_name,
        )

        # Two frames: interpolated + original
        self.output_batch_shared_tensor = SharedTensor(
            (2, h, w, 3),
            name=self.output_batch_shared_tensor_name,
        )

    def process_init(self):
        """
        Initializes all resources required by the inference subprocess.
        """
        self.init_process_state()
        self.init_shared_tensors()

        self.load_models()
        self.load_engines()

        self.pipelines = []
        self.previous_frame = None
        self.pipe_cache = PipeCache()
        self.ext_attention_handler = ExtendedAttentionHandler(self.height, self.width)

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

        color_conditioning_image = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)

        # For some reason we can simply pass grayscale image instead of canny edges to this controlnet
        # And it works even better
        # Very experimental, maybe I will return canny here
        controlnet_image = cv2.cvtColor(image_input, cv2.COLOR_RGB2GRAY)
        controlnet_image = cv2.cvtColor(controlnet_image, cv2.COLOR_GRAY2BGR)

        pipe.execute_preparations(self.process_state['prompt'],
                                  num_inference_steps=self.process_state["steps"],
                                  controlnet_image = controlnet_image,
                                  color_conditioning_image = color_conditioning_image,
                                  inversion_strength = self.process_state["inversion_strength"],
                                  controlnet_conditioning_scale= self.process_state["controlnet_conditioning_scale"],
                                  pipe_cache = self.pipe_cache)

        self.pipelines.append(pipe)

    def collect_active_tasks(self):
        """
        Returns pipelines that are still diffusing and their corresponding tasks.
        """
        active_pipes = [p for p in self.pipelines if not p.latent_is_ready]
        tasks = [p.get_diffusion_task() for p in active_pipes]
        return active_pipes, tasks
    
    def batch_tasks(self, tasks: list):
        """
        Batch DiffusionTask fields into tensors suitable for ControlNet + UNet.
        """
        samples = torch.cat([t.sample for t in tasks], dim=0)
        timesteps = torch.stack([t.timestep for t in tasks])
        encoder_hidden_states = torch.cat([t.encoder_hidden_states for t in tasks], dim=0)
        controlnet_images = torch.cat([t.controlnet_image for t in tasks], dim=0)
        text_embeds = torch.cat([t.text_embeds for t in tasks], dim=0)
        time_ids = torch.cat([t.time_ids for t in tasks], dim=0)

        conditioning_scales = torch.tensor(
            [t.conditioning_scale for t in tasks],
            device=samples.device,
            dtype=samples.dtype
        ).view(-1, 1, 1, 1)

        return (
            samples,
            timesteps,
            encoder_hidden_states,
            controlnet_images,
            text_embeds,
            time_ids,
            conditioning_scales
        )
    
    def run_controlnet(
        self,
        samples,
        timesteps,
        encoder_hidden_states,
        controlnet_images,
        text_embeds,
        time_ids,
        conditioning_scales,
    ):
        """
        Executes ControlNet and applies per-task conditioning scales.
        """
        controlnet_outputs = self.controlnet_engine(
            samples,
            timesteps,
            encoder_hidden_states,
            controlnet_images,
            text_embeds,
            time_ids,
        )

        down_block_residuals = [
            residual * conditioning_scales
            for residual in controlnet_outputs[:-1]
        ]

        mid_block_residual = controlnet_outputs[-1] * conditioning_scales

        return down_block_residuals, mid_block_residual
    

    def run_unet(
        self,
        samples,
        timesteps,
        encoder_hidden_states,
        down_block_residuals,
        mid_block_residual,
        text_embeds,
        time_ids,
        tasks,
    ):
        """
        Executes UNet with extended attention state.
        """
        extended_attention = self.ext_attention_handler.make_extensions_for_tasks(tasks)

        unet_outputs = self.unet_engine(
            samples,
            timesteps,
            encoder_hidden_states,
            *down_block_residuals,
            mid_block_residual,
            text_embeds,
            time_ids,
            *extended_attention,
        )

        noise, ea1_key, ea1_val, ea2_key, ea2_val = unet_outputs

        self.ext_attention_handler.update_chains_from_batch(
            (ea1_key, ea1_val, ea2_key, ea2_val),
            tasks,
        )

        return noise

    @torch.no_grad()
    def step_all_pipelines(self) -> None:
        """
        Steps all active diffusion pipelines by one diffusion step.
        """
        if not self.pipelines:
            return

        active_pipes, tasks = self.collect_active_tasks()

        if not tasks:
            return

        (
            samples,
            timesteps,
            encoder_hidden_states,
            controlnet_images,
            text_embeds,
            time_ids,
            conditioning_scales,
        ) = self.batch_tasks(tasks)

        down_blocks, mid_block = self.run_controlnet(
            samples,
            timesteps,
            encoder_hidden_states,
            controlnet_images,
            text_embeds,
            time_ids,
            conditioning_scales,
        )

        noise = self.run_unet(
            samples,
            timesteps,
            encoder_hidden_states,
            down_blocks,
            mid_block,
            text_embeds,
            time_ids,
            tasks,
        )

        batch_size = len(tasks)
        for pipe, noise_chunk in zip(active_pipes, noise.chunk(batch_size, dim=0)):
            pipe.return_diffusion_result((noise_chunk,))

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

    def interpolate_with_previous(self, frame):
        """
        Interpolates the current frame with the previous one.
        """
        if self.previous_frame is None:
            self.previous_frame = frame

        interpolated = self.interpolation_model_engine(
            torch.cat([self.previous_frame, frame], dim=1)
        )
        self.previous_frame = frame
        return interpolated
    
    def send_frames(self, interpolated_image, image):
        """
        Converts GPU tensors to BGR uint8 frames and writes them to shared memory.
        """
        frames_gpu = torch.cat([interpolated_image, image], dim=0)
        frames_cpu = (
            frames_gpu
            .mul(255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        frames_cpu_bgr = frames_cpu[..., ::-1]

        self.output_batch_shared_tensor.copy_from(frames_cpu_bgr)
        self.pack_is_ready.value = True

    def update_time(self, prev_time):
        now = time.time()
        processing_time = now - prev_time
        self.last_processing_time.value = processing_time
        print("fps", 1 / processing_time)
        return now

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
            interpolated_image = self.interpolate_with_previous(image)
            self.send_frames(interpolated_image, image)
            prev_time = self.update_time(prev_time)
            