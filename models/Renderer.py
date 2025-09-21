import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Union


from transformers import CLIPTextModelWithProjection, T5EncoderModel, CLIPTokenizer, T5TokenizerFast
from diffusers.models import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.loaders import SD3LoraLoaderMixin

from .backbone import NewSD3Transformer2DModel

from utils.dist import is_main_process

import inspect

import copy

from tqdm import tqdm

import numpy as np

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def main_print(str, is_main):
    if is_main:
        print(str)

class RenderNet(nn.Module):
    def __init__(
        self, args
    ):
        super().__init__()
        self.args = args
        
        self.is_main = is_main_process()
        
        self.transformer = NewSD3Transformer2DModel.from_pretrained(self.args.pretrained_model_path, subfolder="transformer")

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.args.pretrained_model_path, subfolder="scheduler")

        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(self.args.pretrained_model_path, subfolder="text_encoder")
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(self.args.pretrained_model_path, subfolder="text_encoder_2")
        self.text_encoder_3 = T5EncoderModel.from_pretrained(self.args.pretrained_model_path, subfolder="text_encoder_3")

        self.tokenizer = CLIPTokenizer.from_pretrained(self.args.pretrained_model_path, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(self.args.pretrained_model_path, subfolder="tokenizer_2")
        self.tokenizer_3 = T5TokenizerFast.from_pretrained(self.args.pretrained_model_path, subfolder="tokenizer_3")

        self.vae = AutoencoderKL.from_pretrained(self.args.pretrained_model_path, subfolder="vae")

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        # image processor is not required for inference path here

        self.scheduler_copy = copy.deepcopy(self.scheduler)

        self.tokenizer_max_length = self.tokenizer.model_max_length

        self.embed_dim = 1536

        self.point_embedder = nn.Linear(1024, self.embed_dim)

        # refer to SAM
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            torch.randn((2, 1024)),
        )

        self.positional_encoding_gaussian_matrix = self.positional_encoding_gaussian_matrix.to('cuda', torch.bfloat16)

        self.not_seg_embeddings = nn.Embedding(1, self.embed_dim)
        self.seg_embeddings = nn.Embedding(1, self.embed_dim)

        self.point_embedder = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, self.embed_dim)
        )
        
        
    def to(self, *args, **kwargs):
        model_converted = super().to(*args, **kwargs)
        self.device = next(self.parameters()).device
        self.dtype = next(self.transformer.parameters()).dtype
        return model_converted
        
    def set_inference_mode(self):
        """Set all components to inference mode (eval + no gradients)"""
        self.vae.eval()
        self.text_encoder.eval()
        self.text_encoder_2.eval()
        self.text_encoder_3.eval()
        self.transformer.eval()
        
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.text_encoder_3.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.point_embedder.requires_grad_(False)
        self.not_seg_embeddings.requires_grad_(False)
        self.seg_embeddings.requires_grad_(False)
        
        
    def get_positional_encoding(self, points, labels):
        points = 2 * points - 1.0
        points = points.to(torch.bfloat16)
        
        points = points @ self.positional_encoding_gaussian_matrix
        points = 2 * np.pi * points
        point_embedding = torch.cat([torch.sin(points), torch.cos(points)], dim=-1)

        point_embedding = self.point_embedder(point_embedding)

        labels = labels.squeeze(-1)
        point_embedding[labels == 0] += self.not_seg_embeddings.weight
        point_embedding[labels == 1] += self.seg_embeddings.weight

        return point_embedding

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.text_encoder_3 is None:
            return torch.zeros(
                (
                    batch_size * num_images_per_prompt,
                    self.tokenizer_max_length,
                    self.transformer.config.joint_attention_dim,
                ),
                device=device,
                dtype=dtype,
            )

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])

        prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]

        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline._get_clip_prompt_embeds
    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clip_skip: Optional[int] = None,
        clip_model_index: int = 0,
    ):
        device = device or self._execution_device

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])

        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds


    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        prompt_3: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
        max_sequence_length: int = 256,
        lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, SD3LoraLoaderMixin):
            self._lora_scale = lora_scale


        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=0,
            )
            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                prompt=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=1,
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

            t5_prompt_embed = self._get_t5_prompt_embeds(
                prompt=prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
            )

            # clip_prompt_embeds[:,0,2048:] = point_embedding

            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_3 = negative_prompt_3 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )
            negative_prompt_3 = (
                batch_size * [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embed, negative_pooled_prompt_embed = self._get_clip_prompt_embeds(
                negative_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=0,
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                negative_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=1,
            )
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

            t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                prompt=negative_prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1]),
            )

            negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
            )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")
    
    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.scheduler_copy.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.scheduler_copy.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
        
    def vae_encode(self, image):
        b, c, h, w = image.size()
        latents = self.vae.encode(image).latent_dist.sample()
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        latents = latents.to(dtype=self.dtype)
        return latents

    def vae_decode(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        dec = self.vae.decode(latents, return_dict=False)[0]
        image = (dec / 2 + 0.5).clamp(0, 1)
        return image
    

    def train(self, mode=True):
        """Override train method to keep model in inference mode"""
        if mode:
            # Force eval mode for inference-only model
            super().eval()
            self.set_inference_mode()
        return self

    def forward(self, inputs):
        outputs = dict()
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.args.seed)
        inputs['generator'] = generator
        outputs = self.eval_fn(inputs, outputs)
        removed_key = inputs.pop("generator", None)
        return outputs

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents


    @torch.no_grad()
    def eval_fn(self, inputs, outputs):

        input_image = inputs['input_images']

        prompt = inputs['prompt']

        b, c, h, w = input_image.size()
        do_classifier_free_guidance = self.args.guidance_scale > 1.0

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            device=self.device,
        )

        # point_embeds = self.point_embedder(point_embeds).unsqueeze(1)
        point_embeds = self.get_positional_encoding(inputs['coor_point'], inputs['point_labels'])  # .unsqueeze(1) for single point

        add_hidden_states = self.vae_encode(input_image)
        add_hidden_states = add_hidden_states.to(dtype=self.dtype)

        if do_classifier_free_guidance:
            point_embeds = torch.cat([torch.zeros_like(point_embeds), point_embeds], dim=0)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            add_hidden_states = torch.concat([add_hidden_states, add_hidden_states], dim=0)

        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, self.args.num_inference_steps, self.device, timesteps=None)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        num_channels_latents = self.transformer.config.in_channels
        gen_height = h
        gen_width = w
        generator = inputs['generator']

        latents = self.prepare_latents(
            b ,
            num_channels_latents,
            gen_height,
            gen_width,
            self.dtype,
            self.device,
            generator,
            latents=None,
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                timestep = t.expand(latent_model_input.shape[0])
                # predict the noise residual
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    add_hidden_states=add_hidden_states,
                    point_embeds=point_embeds,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    # input_embedder=self.input_embedder,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.args.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # Post-processing
        gen_img = self.vae_decode(latents)

        outputs['images'] = gen_img
        return outputs


