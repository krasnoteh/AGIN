import torch


class StableDiffusionXLControlNetPipeline:
    """
    Lightweight SDXL ControlNet pipeline wrapper for batched stream diffusion.
    UNet and ControlNet are handled externally.
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
        device="cuda",
        dtype=torch.float16,
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

    def retrieve_timesteps(self, num_inference_steps, inversion_strength: float = 1.0):
        init_step = min(
            max(int(num_inference_steps * inversion_strength), 1),
            num_inference_steps,
        )
        start_step = max(num_inference_steps - init_step, 0)

        timesteps = self.scheduler.timesteps[start_step * self.scheduler.order :]
        return timesteps, num_inference_steps - start_step
    
    @torch.no_grad()
    def encode_prompt(self, prompt: str):
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
        prompt,
        controlnet_image,
        inversion_strength,
        controlnet_conditioning_scale,
        color_conditioning_image,
        num_inference_steps,
        pipe_cache,
    ):
        color_conditioning_image = torch.tensor(color_conditioning_image, device=self.device, dtype=self.dtype)
        color_conditioning_image = color_conditioning_image.permute(2, 0, 1).unsqueeze(0)
        color_conditioning_image = color_conditioning_image / 128.0 - 1.0

        controlnet_image = torch.tensor(controlnet_image, device=self.device, dtype=self.dtype)
        controlnet_image = controlnet_image.permute(2, 0, 1).unsqueeze(0)
        controlnet_image = controlnet_image / 255.0

        height, width = controlnet_image.shape[-2:]
        latent_height, latent_width = height // 8, width // 8

        if prompt in pipe_cache.prompt_cache:
            text_embeds, encoder_hidden_states = pipe_cache.prompt_cache[prompt]
        else:
            text_embeds, encoder_hidden_states = self.encode_prompt(prompt)
            pipe_cache.prompt_cache[prompt] = (text_embeds, encoder_hidden_states)

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps, _ = self.retrieve_timesteps(num_inference_steps, inversion_strength)

        self.timesteps = timesteps
        self.current_timestep_index = -1

        timestep = timesteps[:1]
        num_latent_channels = 4

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
                timestep=timestep,
                conditioning_image=color_conditioning_image,
            )

        self.latents = latents
        self.text_embeds = text_embeds
        self.encoder_hidden_states = encoder_hidden_states

        self.time_ids = torch.tensor(
            [[1024, 1024, 0, 0, 1024, 1024]],
            dtype=self.dtype,
            device=self.device,
        )

        self.controlnet_image = controlnet_image
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.latent_is_ready = False

    @torch.no_grad()
    def get_diffusion_task(self):
        self.current_timestep_index += 1
        t = self.timesteps[self.current_timestep_index]

        latent_input = self.scheduler.scale_model_input(self.latents, t)

        return {
            "sample": latent_input,
            "timestep": t,
            "encoder_hidden_states": self.encoder_hidden_states,
            "added_cond_kwargs": {
                "text_embeds": self.text_embeds,
                "time_ids": self.time_ids,
            },
            "controlnet_image": self.controlnet_image,
            "controlnet_conditioning_scale": self.controlnet_conditioning_scale,
        }

    @torch.no_grad()
    def return_diffusion_result(self, result):
        noise_pred = result[0]
        t = self.timesteps[self.current_timestep_index]

        self.latents = self.scheduler.step(
            noise_pred,
            t,
            self.latents,
            generator=self.generator,
        )[0]

        if self.current_timestep_index == len(self.timesteps) - 1:
            self.latent_is_ready = True

    @torch.no_grad()
    def decode_latent(self):
        image = self.decoder(self.latents)
        image = (image + 1) / 2.0
        image = torch.clamp(image, 0, 1)

        return image
