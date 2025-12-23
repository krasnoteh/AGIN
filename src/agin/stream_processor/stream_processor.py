from multiprocessing import Process, Value, Manager, Event
from agin.utils import SharedTensor
from agin.stream_processor.subprocesses.model_inference_subprocess import ModelInferenceSubprocess
from agin.stream_processor.subprocesses.output_scheduler_subprocess import OutputSchedulerSubprocess
from queue import Empty
import json
import time


class StreamProcessor:
    def __init__(self, config_path: str):
        self.config = self.parse_config(config_path)

        self.resolution = self.config["resolution"]

    def parse_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as file:
            return json.load(file)

    def start(self) -> None:
        self.input_shared_tensor = SharedTensor((self.resolution["height"], self.resolution["width"], 3), create=True)
        self.output_shared_tensor = SharedTensor((self.resolution["height"], self.resolution["width"], 3), create=True)
        self.output_batch_shared_tensor = SharedTensor((2, self.resolution["height"], self.resolution["width"], 3), create=True)
        
        self.pack_is_ready = Value('b', False)
        self.last_processing_time = Value("f", 0.0)
        
        self.model_inference_subprocess = ModelInferenceSubprocess(self.config, self.input_shared_tensor.name, self.output_batch_shared_tensor.name, self.pack_is_ready, self.last_processing_time)
        self.model_inference_subprocess.start()

        self.output_scheduler_subprocess = OutputSchedulerSubprocess(self.config, self.output_batch_shared_tensor.name, self.output_shared_tensor.name, self.pack_is_ready, self.last_processing_time)
        self.output_scheduler_subprocess.start()

    def stop(self) -> None:
        self.model_inference_subprocess.stop()
        self.output_scheduler_subprocess.stop()

    def set_prompt(self, prompt: str) -> None:
        self.model_inference_subprocess.set_param(name="prompt", value=prompt)

    def set_steps(self, steps: int) -> None:
        self.model_inference_subprocess.set_param(name="steps", value=steps)

    def set_inversion_strength(self, inversion_strength: float) -> None:
        self.model_inference_subprocess.set_param(name="inversion_strength", value=inversion_strength)

    def set_controlnet_conditioning_scale(self, controlnet_conditioning_scale: float) -> None:
        self.model_inference_subprocess.set_param(name="controlnet_conditioning_scale", value=controlnet_conditioning_scale)

    def set_canny_low_threshold(self, canny_low_threshold: int) -> None:
        self.model_inference_subprocess.set_param(name="canny_low_threshold", value=canny_low_threshold)

    def set_canny_high_threshold(self, canny_high_threshold: int) -> None:
        self.model_inference_subprocess.set_param(name="canny_high_threshold", value=canny_high_threshold)

    def set_seed(self, seed: int) -> None:
        self.model_inference_subprocess.set_param(name="seed", value=seed)

    def set_param(self, name: str, value) -> None:
        self.model_inference_subprocess.set_param(name=name, value=value)

    def get_resolution(self) -> dict:
        return self.resolution

    def get_input_shared_tensor_name(self) -> str:
        return self.input_shared_tensor.name

    def get_output_shared_tensor_name(self) -> str:
        return self.output_shared_tensor.name