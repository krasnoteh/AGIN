import torch
from typing import Tuple
from agin.stream_processor.diffusion_task import DiffusionTask


class StableDiffusionXLControlNetPipeline:
    """
    Lightweight SDXL ControlNet pipeline wrapper for batched stream diffusion.

    Responsibilities:
    - Prepare initial latents and conditioning
    - Track diffusion timestep progression
    - Emit DiffusionTask objects for batched execution
    - Consume noise predictions and update latents
    - Decode final latent into image

    UNet and ControlNet execution are handled externally.
    """

    def __init__(
        self,
        encoder,
        decoder,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        scheduler,
        seed,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.scheduler = scheduler

        self.device = device
        self.dtype = dtype
        self.generator = torch.Generator(device=device).manual_seed(seed)

        self.latents = None
        self.timesteps = None
        self.current_timestep_index = None
        self.latent_is_ready = False

    def retrieve_timesteps(
        self,
        num_inference_steps: int,
        inversion_strength: float = 1.0,
    ) -> Tuple[torch.Tensor, int]:
        """
        Computes the diffusion timesteps based on inversion strength.
        """
        init_step = min(
            max(int(num_inference_steps * inversion_strength), 1),
            num_inference_steps,
        )
        start_step = max(num_inference_steps - init_step, 0)

        timesteps = self.scheduler.timesteps[start_step * self.scheduler.order :]
        return timesteps, num_inference_steps - start_step

    @torch.no_grad()
    def encode_prompt(self, prompt: str):
        """
        Encodes the text prompt using both SDXL text encoders.
        """
        inputs_1 = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.device)

        inputs_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_2.model_max_length,
            return_tensors="pt",
        ).to(self.device)

        enc_1 = self.text_encoder(
            inputs_1.input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        enc_2 = self.text_encoder_2(
            inputs_2.input_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        text_embeds = enc_2[0]
        tokens_1 = enc_1.hidden_states[-2]
        tokens_2 = enc_2.hidden_states[-2]

        encoder_hidden_states = torch.cat([tokens_1, tokens_2], dim=-1)
        return text_embeds, encoder_hidden_states

    def _encode_and_add_noise(self, timestep, conditioning_image):
        latents = self.encoder(conditioning_image)
        noise = torch.normal(
            mean=0,
            std=1,
            size=latents.shape,
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        )
        return self.scheduler.add_noise(latents, noise, timestep)

    @torch.no_grad()
    def execute_preparations(
        self,
        prompt: str,
        controlnet_image,
        inversion_strength: float,
        controlnet_conditioning_scale: float,
        color_conditioning_image,
        num_inference_steps: int,
        pipe_cache,
    ):
        """
        Initializes latent state, text conditioning, and scheduler.
        Must be called before any diffusion steps.
        """
        # Image preprocessing
        color_conditioning_image = torch.tensor(
            color_conditioning_image,
            device=self.device,
            dtype=self.dtype,
        ).permute(2, 0, 1).unsqueeze(0)
        color_conditioning_image = color_conditioning_image / 128.0 - 1.0

        controlnet_image = torch.tensor(
            controlnet_image,
            device=self.device,
            dtype=self.dtype,
        ).permute(2, 0, 1).unsqueeze(0)
        controlnet_image = controlnet_image / 255.0

        height, width = controlnet_image.shape[-2:]
        latent_height, latent_width = height // 8, width // 8

        # Prompt encoding (cached)
        if prompt in pipe_cache.prompt_cache:
            text_embeds, encoder_hidden_states = pipe_cache.prompt_cache[prompt]
        else:
            text_embeds, encoder_hidden_states = self.encode_prompt(prompt)
            pipe_cache.prompt_cache[prompt] = (text_embeds, encoder_hidden_states)

        # Scheduler setup
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        self.timesteps, _ = self.retrieve_timesteps(
            num_inference_steps,
            inversion_strength,
        )

        self.current_timestep_index = -1

        # Latent initialization
        num_latent_channels = 4
        first_timestep = self.timesteps[:1]

        if inversion_strength == 1.0:
            latents = torch.normal(
                mean=0,
                std=1,
                size=(1, num_latent_channels, latent_height, latent_width),
                generator=self.generator,
                device=self.device,
                dtype=self.dtype,
            )
            latents *= self.scheduler.init_noise_sigma
        else:
            latents = self._encode_and_add_noise(
                timestep=first_timestep,
                conditioning_image=color_conditioning_image,
            )

        # Persist state
        self.latents = latents
        self.text_embeds = text_embeds
        self.encoder_hidden_states = encoder_hidden_states
        self.controlnet_image = controlnet_image
        self.controlnet_conditioning_scale = controlnet_conditioning_scale

        # SDXL time embedding
        self.time_ids = torch.tensor(
            [[1024, 1024, 0, 0, 1024, 1024]],
            dtype=self.dtype,
            device=self.device,
        )

        self.latent_is_ready = False

    @torch.no_grad()
    def get_diffusion_task(self) -> DiffusionTask:
        """
        Advances the internal timestep index and returns a DiffusionTask.
        """
        self.current_timestep_index += 1
        timestep = self.timesteps[self.current_timestep_index]

        scaled_latents = self.scheduler.scale_model_input(
            self.latents,
            timestep,
        )

        return DiffusionTask(
            sample=scaled_latents,
            timestep=timestep,
            encoder_hidden_states=self.encoder_hidden_states,
            text_embeds=self.text_embeds,
            time_ids=self.time_ids,
            controlnet_image=self.controlnet_image,
            conditioning_scale=self.controlnet_conditioning_scale,
        )

    @torch.no_grad()
    def return_diffusion_result(self, result):
        """
        Consumes the predicted noise and updates the latent state.
        """
        noise_pred = result[0]
        timestep = self.timesteps[self.current_timestep_index]

        self.latents = self.scheduler.step(
            noise_pred,
            timestep,
            self.latents,
            generator=self.generator,
        )[0]

        if self.current_timestep_index == len(self.timesteps) - 1:
            self.latent_is_ready = True

    @torch.no_grad()
    def decode_latent(self) -> torch.Tensor:
        """
        Decodes the final latent into an RGB image tensor in [0, 1].
        """
        image = self.decoder(self.latents)
        image = (image + 1) / 2.0
        return torch.clamp(image, 0, 1)
