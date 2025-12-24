from dataclasses import dataclass
import torch


@dataclass
class DiffusionTask:
    """
    A single diffusion step request consumed by ControlNet + UNet.
    """
    sample: torch.Tensor
    timestep: torch.Tensor
    encoder_hidden_states: torch.Tensor
    text_embeds: torch.Tensor
    time_ids: torch.Tensor
    controlnet_image: torch.Tensor
    conditioning_scale: float
