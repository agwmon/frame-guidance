# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import html
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL
from PIL import Image
import regex as re
import torch
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
import numpy as np
import torch.nn.functional as F
import gc

import torchvision
from .utils.models import setup_csd
import random

clip_preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
    )
])

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy

EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers.utils import export_to_video
        >>> from diffusers import AutoencoderKLWan, WanPipeline
        >>> from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        >>> # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
        >>> model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        >>> vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        >>> pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
        >>> flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
        >>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
        >>> pipe.to("cuda")

        >>> prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
        >>> negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

        >>> output = pipe(
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     height=720,
        ...     width=1280,
        ...     num_frames=81,
        ...     guidance_scale=5.0,
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=16)
        ```
"""


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class WanPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Pipeline for text-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer ([`T5Tokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        transformer ([`WanTransformer3DModel`]):
            Conditional Transformer to denoise the input latents.
        scheduler ([`UniPCMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        transformer: WanTransformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae.requires_grad_(False),
            text_encoder=text_encoder.requires_grad_(False),
            tokenizer=tokenizer,
            transformer=transformer.requires_grad_(False),
            scheduler=scheduler,
        )

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline.encode_prompt
    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

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

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        # # no scaling in wan
        # frames = self.vae.decode(latents).sample
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        frames = self.vae.decode(latents, return_dict=False)[0]

        return frames
    
    @torch.no_grad()
    def encode_video(self, video: list[Image.Image], generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # video = self.video_processor.preprocess_video(video, height=height, width=width).to(self.device, dtype=self.vae.dtype)
        latents = [retrieve_latents(self.vae.encode(v), generator) for v in video]
        latents = torch.cat(latents, dim=0).permute(0, 2, 1, 3, 4) # [batch_size, num_frames, num_channels, height, width]

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    # ------------------------------------------------------------------ #
    #  Helper: validate guidance args & normalize to per-step lists
    # ------------------------------------------------------------------ #
    def _validate_guidance_args(self, guidance_step, guidance_lr, num_inference_steps, loss_fn):
        VALID_LOSS_FNS = {"style", "loop"}
        if loss_fn not in VALID_LOSS_FNS:
            raise ValueError(f"loss_fn must be one of {sorted(VALID_LOSS_FNS)}")

        if isinstance(guidance_step, int):
            guidance_step = [guidance_step] * num_inference_steps
        else:
            assert len(guidance_step) == num_inference_steps, \
                "guidance_step must be a list of length num_inference_steps"

        if isinstance(guidance_lr, float):
            guidance_lr = [guidance_lr] * num_inference_steps
        else:
            assert len(guidance_lr) == num_inference_steps, \
                "guidance_lr must be a list of length num_inference_steps"

        return guidance_step, guidance_lr

    # ------------------------------------------------------------------ #
    #  Helper: prepare per-loss-fn context (models, embeddings, etc.)
    # ------------------------------------------------------------------ #
    def _prepare_loss_context(self, loss_fn, video, additional_inputs, device, dtype, latent_downscale_factor=1):
        """Return a dict of pre-computed objects needed by the chosen loss."""
        ctx: Dict[str, Any] = {}

        if loss_fn == "style":
            assert additional_inputs is not None and "style_image" in additional_inputs, \
                "additional_inputs with 'style_image' required for style loss"
            if not hasattr(self, "csd"):
                self.csd, self.preprocess = setup_csd(device=device)
            ctx["cosine_loss"] = torch.nn.CosineSimilarity(dim=1)
            ctx["style_image"] = additional_inputs["style_image"]
            ctx["style_weight"] = additional_inputs.get("style_weight", 1.0)
            ctx["content_weight"] = additional_inputs.get("content_weight", 0.0)

            if "content_weight" in additional_inputs:
                content_embeddings = []
                for content_image in video:
                    _, content_emb_ref, _ = self.csd(
                        self.preprocess(content_image).unsqueeze(0).to(device))
                    content_embeddings.append(content_emb_ref.to(device))
                ctx["content_embeddings"] = content_embeddings

        elif loss_fn == "loop":
            pass  # No special context needed

        return ctx

    # ------------------------------------------------------------------ #
    #  Helper: single transformer forward pass (with or without grad)
    # ------------------------------------------------------------------ #
    def _predict_noise(
        self, latent_model_input, encoder_hidden_states,
        timestep, attention_kwargs, *, enable_grad=False,
    ):
        ctx_mgr = torch.enable_grad() if enable_grad else torch.no_grad()
        with ctx_mgr:
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
        return noise_pred

    # ------------------------------------------------------------------ #
    #  Helper: compute guidance loss for one fixed frame
    # ------------------------------------------------------------------ #
    def _compute_single_frame_loss(
        self, loss_fn, pred_original_sample, fixed_frame,
        original_video, loss_ctx, additional_inputs, device, step_idx, rep_idx,
        latent_downscale_factor=1,
    ):
        """Decode the predicted latent around *fixed_frame* and return a scalar loss."""
        if fixed_frame == 0:
            print("fixed_frame == 0, pass")
            return None

        latent_frame = (fixed_frame - 1) // 4 + 1
        latent_pair = pred_original_sample[:, :, latent_frame - 1 : latent_frame + 1]  # [1, C, 2, H, W]
        if latent_downscale_factor > 1:
            latent_pair = F.interpolate(
                latent_pair.squeeze(0), scale_factor=1/latent_downscale_factor,
                mode='bilinear', align_corners=False
            ).unsqueeze(0)
        decoded_frames = self.decode_latents(latent_pair)  # [1, 3, 5, H, W]
        del latent_pair

        relative_frame_idx = (fixed_frame - 1) % 4 + 1
        pred_frame = decoded_frames[:, :, relative_frame_idx : relative_frame_idx + 1]  # [1, 3, 1, H, W]

        # ---- per-loss-fn computation ----
        if loss_fn == "style":
            pred_frame_sq = pred_frame.squeeze(2)
            _, content_emb, style_emb = self.csd(clip_preprocess(pred_frame_sq.float()))
            with torch.no_grad():
                if "style_embed" not in additional_inputs:
                    _, _, style_emb_ref = self.csd(
                        self.preprocess(loss_ctx["style_image"]).unsqueeze(0).to(device))
                    additional_inputs["style_embed"] = style_emb_ref
                style_emb_ref = additional_inputs["style_embed"].to(device)
            cos = loss_ctx["cosine_loss"]
            style_sim = cos(style_emb, style_emb_ref)
            print(f"style_similarity({step_idx}/{rep_idx}): {style_sim.item():.3f}")
            loss = loss_ctx["style_weight"] * (1 - style_sim)

            if "content_embeddings" in loss_ctx:
                content_sim = cos(content_emb, loss_ctx["content_embeddings"][fixed_frame])
                print(f"content_similarity({step_idx}/{rep_idx}): {content_sim.item():.3f}")
                loss = loss + loss_ctx["content_weight"] * (1 - content_sim)

        del decoded_frames, pred_frame
        return loss

    # ------------------------------------------------------------------ #
    #  Helper: compute loop loss (MSE between first and last frame)
    # ------------------------------------------------------------------ #
    def _compute_loop_loss(self, pred_original_sample, num_frames, latent_downscale_factor=1):
        """Decode first and last frames and compute MSE between them."""
        fixed_frames = [0, num_frames - 1]
        predicted_images = []
        for fixed_frame in fixed_frames:
            if fixed_frame == 0:
                latent_pair = pred_original_sample[:, :, 0:1]
            else:
                latent_frame = (fixed_frame - 1) // 4 + 1
                latent_pair = pred_original_sample[:, :, latent_frame - 1 : latent_frame + 1]

            if latent_downscale_factor > 1:
                latent_pair = F.interpolate(
                    latent_pair.squeeze(0), scale_factor=1/latent_downscale_factor,
                    mode='bilinear', align_corners=False
                ).unsqueeze(0)

            if fixed_frame == 0:
                # with torch.no_grad():
                decoded = self.decode_latents(latent_pair)
                relative_idx = 0
            else:
                decoded = self.decode_latents(latent_pair)
                relative_idx = (fixed_frame - 1) % 4 + 1

            pred_frame = decoded[:, :, relative_idx : relative_idx + 1]
            predicted_images.append(pred_frame)
            del decoded, latent_pair

        loss = F.mse_loss(predicted_images[0], predicted_images[1])
        # loss = F.mse_loss(predicted_images[0].detach(), predicted_images[1])
        del predicted_images
        torch.cuda.empty_cache()
        return loss

    # ------------------------------------------------------------------ #
    #  Helper: apply gradient-based guidance update to latents
    # ------------------------------------------------------------------ #
    def _apply_guidance_update(
        self, latents, pred_original_sample, grad,
        guidance_lr_i, sigma, generator, device, in_travel_time,
    ):
        grad_norm = grad.norm(2)
        rho = 1.0 / grad_norm

        with torch.no_grad():
            if in_travel_time:
                noise = randn_tensor(
                    pred_original_sample.shape, generator=generator,
                    device=device, dtype=pred_original_sample.dtype)
                latents = sigma * noise + (1 - sigma) * pred_original_sample
                latents = latents - guidance_lr_i * rho * grad
            else:
                latents = latents - guidance_lr_i * rho * grad

        return latents

    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        video: Optional[list[Image.Image]] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        fixed_frames: Optional[Union[int, List[int]]] = None,
        guidance_step: Union[list[int], int] = 0,
        guidance_lr: Union[float, list[float]] = 1e-2,
        loss_fn: str = "frame",
        additional_inputs: Optional[Dict[str, Any]] = None,
        travel_time: Tuple[int, int] = (0, 50),
        latent_downscale_factor: int = 1,
        offload: bool = True,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, defaults to `480`):
                The height in pixels of the generated image.
            width (`int`, defaults to `832`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            autocast_dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
                The dtype to use for the torch.amp.autocast.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """
        # --- Validate & normalize guidance arguments ---
        guidance_step, guidance_lr = self._validate_guidance_args(
            guidance_step, guidance_lr, num_inference_steps, loss_fn)

        if latent_downscale_factor not in (1, 2, 4):
            raise ValueError("latent_downscale_factor must be one of [1, 2, 4]")

        # --- Prepare loss-specific context (models, conditioning videos, etc.) ---
        loss_ctx = self._prepare_loss_context(
            loss_fn, video, additional_inputs,
            device=self._execution_device, dtype=self.transformer.dtype,
            latent_downscale_factor=latent_downscale_factor)

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        if offload:
            self.text_encoder.to(device)
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if offload:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        # Prepare target video for guidance
        if loss_fn == "loop":
            fixed_frames = [0, num_frames - 1]
            original_video = None
        elif video is not None:
            target_h = height // latent_downscale_factor if latent_downscale_factor > 1 else height
            target_w = width // latent_downscale_factor if latent_downscale_factor > 1 else width
            original_video = self.video_processor.preprocess_video(video, height=target_h, width=target_w)
            original_video = original_video.to(device=device, dtype=self.vae.dtype)  # [1, 3, num_frames, H, W]
        else:
            original_video = None

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        scheduler = self.scheduler

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                in_guidance_range = (guidance_step[i] != 0)
                n_repeats = guidance_step[i]

                self._current_timestep = t
                timestep = t.expand(latents.shape[0])

                # Scheduler coefficients for this step
                if scheduler.step_index is None:
                    scheduler._init_step_index(timestep)
                sigma = scheduler.sigmas[scheduler.step_index]
                sigma_next = scheduler.sigmas[scheduler.step_index + 1]

                for rep in range(n_repeats + 1):
                    latents = latents.detach().clone().requires_grad_(in_guidance_range)
                    latent_model_input = latents.to(transformer_dtype)

                    need_grad = in_guidance_range and rep < n_repeats

                    # Predict noise (positive and negative passes)
                    noise_pred = self._predict_noise(
                        latent_model_input, prompt_embeds,
                        timestep, attention_kwargs, enable_grad=need_grad,
                    )
                    noise_uncond = self._predict_noise(
                        latent_model_input, negative_prompt_embeds,
                        timestep, attention_kwargs, enable_grad=need_grad,
                    )
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                    # Predicted x0 (flow matching)
                    pred_original_sample = latents - sigma * noise_pred

                    # ---- Guidance gradient step (skip on last rep) ----
                    if need_grad:
                        if loss_fn == "loop":
                            total_loss = self._compute_loop_loss(
                                pred_original_sample, num_frames,
                                latent_downscale_factor=latent_downscale_factor,
                            )
                        else:
                            fixed_frames_ = fixed_frames if fixed_frames is not None \
                                else random.sample(range(5, num_frames), 4)
                            total_loss = 0.0
                            for ff in fixed_frames_:
                                loss = self._compute_single_frame_loss(
                                    loss_fn, pred_original_sample, ff,
                                    original_video, loss_ctx, additional_inputs,
                                    device, step_idx=i, rep_idx=rep,
                                    latent_downscale_factor=latent_downscale_factor,
                                )
                                if loss is not None:
                                    total_loss = total_loss + loss

                        total_loss.backward()
                        print(f"total_loss({i}/{rep}): {total_loss.item():.3f}")
                        grad = latents.grad.clone()
                        latents.grad = None

                        in_travel = i >= travel_time[0] and i <= travel_time[1]
                        latents = self._apply_guidance_update(
                            latents, pred_original_sample, grad,
                            guidance_lr[i], sigma, generator, device, in_travel,
                        )
                        torch.cuda.empty_cache()

                # ---- Final denoising step ----
                with torch.no_grad():
                    latents = latents + (sigma_next - sigma) * noise_pred

                latents = latents.to(noise_pred.dtype)
                scheduler._step_index += 1

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
