import torch
from agin.stream_processor.diffusion_task import DiffusionTask

from typing import Tuple
AttentionExtension = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

class ExtendedAttentionHandler:
    """
    Manages per-timestep extended attention ("attention chains") for stream diffusion.

    This implements a lightweight, no-fine-tuning variant of TokenFlow-style
    extended attention adapted to real-time stream diffusion inference.

    Core idea:
    - During UNet inference, native attention key/value tensors are extracted.
    - These tensors are cached per diffusion timestep.
    - On the next frame, for the *same diffusion timestep*, the cached
      key/value tensors are concatenated into the attention computation.

    Differences from TokenFlow:
    - TokenFlow uses symmetric reshaping during batched inference.
    - This implementation is asymmetric and sequential, operating on a stream.
    - Because of this asymmetry, we refer to the mechanism as "attention chains"
      rather than strict TokenFlow extended attention.

    Notes:
    - Zero tensors are used when no chain exists for a given timestep.
    """
    def __init__(self, height: int, width: int):
        self.chains: dict[int, AttentionExtension] = {}

        latent_height = height // 8
        latent_width = width // 8

        self.num_low_tokens = (latent_height // 4) * (latent_width // 4)
        self.num_mid_tokens = (latent_height // 2) * (latent_width // 2)

        # Attention tensor shapes for different UNet blocks
        self.low_block_shape = (30, 20, self.num_low_tokens, 64)
        self.mid_block_shape = (6, 10, self.num_mid_tokens, 64)

    def _task_timestep(self, task: DiffusionTask) -> int:
        """
        Extracts integer timestep key from DiffusionTask.
        """
        return int(task.timestep.item())

    def make_zero_extension(self, batch_size: int) -> AttentionExtension:
        """
        Returns zero-filled attention tensors used when no chain exists.
        """
        device = "cuda:0"
        dtype = torch.float16

        ea1_key = torch.zeros(batch_size, *self.low_block_shape, device=device, dtype=dtype)
        ea1_val = torch.zeros(batch_size, *self.low_block_shape, device=device, dtype=dtype)
        ea2_key = torch.zeros(batch_size, *self.mid_block_shape, device=device, dtype=dtype)
        ea2_val = torch.zeros(batch_size, *self.mid_block_shape, device=device, dtype=dtype)

        return ea1_key, ea1_val, ea2_key, ea2_val

    def update_chain(self, timestep: int, extension: AttentionExtension) -> None:
        """
        Stores attention extension for a specific diffusion timestep.
        """
        self.chains[timestep] = extension

    def get_chain_extension(self, timestep: int) -> AttentionExtension:
        """
        Retrieves attention extension for timestep or returns zeros if missing.
        """
        return self.chains.get(timestep, self.make_zero_extension(1))

    def make_extensions_for_tasks(self, tasks: list[DiffusionTask]) -> AttentionExtension:
        """
        Collects and batches attention extensions for a list of diffusion tasks.
        """
        ea1_keys, ea1_vals, ea2_keys, ea2_vals = [], [], [], []

        for task in tasks:
            ext = self.get_chain_extension(self._task_timestep(task))
            ea1_keys.append(ext[0])
            ea1_vals.append(ext[1])
            ea2_keys.append(ext[2])
            ea2_vals.append(ext[3])

        return (
            torch.cat(ea1_keys, dim=0),
            torch.cat(ea1_vals, dim=0),
            torch.cat(ea2_keys, dim=0),
            torch.cat(ea2_vals, dim=0),
        )

    def update_chains_from_batch(
        self,
        extensions: AttentionExtension,
        tasks: list[DiffusionTask],
    ) -> None:
        """
        Splits batched attention tensors and updates per-timestep chains.
        """
        batch_size = len(tasks)
        ea1_key, ea1_val, ea2_key, ea2_val = extensions

        ea1_key_chunks = ea1_key.chunk(batch_size, dim=0)
        ea1_val_chunks = ea1_val.chunk(batch_size, dim=0)
        ea2_key_chunks = ea2_key.chunk(batch_size, dim=0)
        ea2_val_chunks = ea2_val.chunk(batch_size, dim=0)

        for i, task in enumerate(tasks):
            self.update_chain(
                self._task_timestep(task),
                (
                    ea1_key_chunks[i],
                    ea1_val_chunks[i],
                    ea2_key_chunks[i],
                    ea2_val_chunks[i],
                ),
            )