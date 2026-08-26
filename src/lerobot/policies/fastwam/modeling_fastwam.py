# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import torch
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.import_utils import require_package

from .configuration_fastwam import FastWAMConfig
from .fastwam_lora import FASTWAM_VIDEO_LORA_TARGETS, inject_lora
from .text_context import (
    TextContextProvenance,
    format_task_prompts,
    load_text_context_artifact,
)
from .wan import (
    ActionDiT,
    FastWAM,
    MoT,
    WanVideoDiT,
    build_wan_tokenizer,
    load_pretrained_wan_text_encoder,
    load_pretrained_wan_vae,
)


def _lora_checkpoint_key_aliases(model_state_dict: dict[str, Tensor], model: Any) -> dict[str, str]:
    """Map legacy unwrapped video projection keys to LoRA base-layer keys."""
    config = getattr(model, "config", None)
    if getattr(config, "lora_rank", 0) <= 0:
        return {}

    aliases = {}
    for target in FASTWAM_VIDEO_LORA_TARGETS:
        for parameter_name in ("weight", "bias"):
            source_key = f"model.{target}.{parameter_name}"
            target_key = f"model.{target}.base_layer.{parameter_name}"
            if target_key in model_state_dict:
                aliases[source_key] = target_key
    return aliases


class FastWAMPolicy(PreTrainedPolicy):
    """LeRobot policy wrapper for FastWAM.

    Attention backend: FastWAM's DiT uses ``torch.nn.functional.scaled_dot_product_attention``
    (SDPA) for all attention. It does not use FlashAttention, because MoT routing requires
    arbitrary boolean ``[query, key]`` masks that the FlashAttention varlen API cannot express;
    installing ``flash-attn`` has no effect on the FastWAM path. (SDPA may still dispatch to
    PyTorch's own flash/mem-efficient/math kernel internally, unrelated to the ``flash-attn`` package.)

    Args:
        config (FastWAMConfig): FastWAM policy configuration.
        dataset_stats (dict[str, dict[str, Tensor]] | None): Optional LeRobot
            dataset statistics passed by the training/evaluation stack.
    """

    config_class = FastWAMConfig
    name = "fastwam"
    # FSDP2 wrap units: MoTLayer is the single FSDP owner of each layer's expert blocks
    # (the blocks are re-parented onto it precisely so sharding has one boundary to hook).
    _fsdp_wrap_modules = ["MoTLayer"]

    def __init__(
        self,
        config: FastWAMConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        **kwargs: Any,
    ):
        # FastWAM's Wan2.2 backbone needs transformers (UMT5 text encoder/tokenizer) and
        # diffusers (Wan VAE), both behind the `fastwam` extra. Fail fast with an actionable
        # message in base installs rather than deep in Wan component construction.
        require_package("transformers", extra="fastwam")
        require_package("diffusers", extra="fastwam")
        # `make_policy`/`from_pretrained` forward extra kwargs (e.g. `dataset_meta`); the
        # dataset feature metadata is already applied to `config` by make_policy upstream,
        # so we accept and ignore them, matching the other LeRobot policies.
        super().__init__(config, dataset_stats)
        config.validate_features()
        self.config = config
        self.dataset_stats = dataset_stats
        preloaded_text_context = None
        if config.text_context_path:
            preloaded_text_context = load_text_context_artifact(
                config.text_context_path,
                expected_provenance=_expected_text_context_provenance(config),
            )
            dataset_meta = kwargs.get("dataset_meta")
            if dataset_meta is not None:
                tasks = _dataset_task_strings(dataset_meta)
                if tasks:
                    preloaded_text_context.validate_prompts(
                        format_task_prompts(tasks, config.prompt_template)
                    )
        self._preloaded_text_context = preloaded_text_context
        try:
            self.model = self._build_core_model(config)
        finally:
            self._preloaded_text_context = None
        self.lora_target_modules: tuple[str, ...] = ()
        if config.lora_rank > 0:
            # Freeze the video base before wrapping its projections. Injection then
            # restores requires_grad only on the new fp32 adapter tensors.
            self._freeze_video_expert()
            self.lora_target_modules = inject_lora(self.model, rank=config.lora_rank, alpha=config.lora_alpha)
        elif config.freeze_video_expert:
            self._freeze_video_expert()
        self.reset()

    @classmethod
    def _load_as_safetensor(cls, model, model_file: str, map_location: str, strict: bool):
        """Shape-aware load that supports cross-embodiment fine-tuning.

        `safetensors.load_model(strict=False)` ignores missing/unexpected keys but
        still raises on a shape mismatch for a shared key. When fine-tuning from a
        checkpoint trained on a different embodiment (e.g. the LIBERO 7-DoF / 8-dim
        checkpoint adapted to a 6-DoF / 6-dim arm), the action encoder/head and
        proprio encoder legitimately differ in shape. With `strict=False` we drop
        only those shape-mismatched tensors — leaving them at their freshly
        initialized values — and load every compatible tensor. With `strict=True`
        the standard exact-match loader is used.
        """
        from safetensors import safe_open

        model_state_dict = model.state_dict()
        lora_aliases = _lora_checkpoint_key_aliases(model_state_dict, model)
        mismatched = []
        with safe_open(model_file, framework="pt") as f:
            checkpoint_keys = list(f.keys())
            for key in checkpoint_keys:
                target_key = lora_aliases.get(key, key)
                if target_key in model_state_dict and tuple(model_state_dict[target_key].shape) != tuple(
                    f.get_slice(key).get_shape()
                ):
                    mismatched.append(target_key)

        if mismatched and strict:
            raise RuntimeError(
                f"FastWAM: {len(mismatched)} checkpoint tensors have a shape mismatch under "
                f"strict=True: {mismatched}"
            )

        from safetensors.torch import load_file

        if mismatched:
            logging.warning(
                "FastWAM cross-embodiment load: reinitializing %d shape-mismatched tensor(s), keeping "
                "every compatible weight: %s",
                len(mismatched),
                mismatched,
            )

        # Always materialize the checkpoint on CPU. Loading the ~6B-parameter state dict directly
        # on CUDA duplicates the already assembled model and exhausts a 24 GB card before the first
        # training step. The core is constructed in bf16 on CPU, populated here, and moved once.
        state_dict = load_file(model_file, device="cpu")
        for source_key, target_key in lora_aliases.items():
            if source_key in state_dict:
                if target_key not in state_dict:
                    state_dict[target_key] = state_dict[source_key]
                del state_dict[source_key]
        if mismatched:
            for key in mismatched:
                state_dict.pop(key, None)
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict, strict=not bool(mismatched) and strict
        )
        del state_dict
        if missing_keys:
            logging.info("Missing key(s) when loading FastWAM model: %s", missing_keys)
        if unexpected_keys:
            logging.info("Unexpected key(s) when loading FastWAM model: %s", unexpected_keys)

        target_device = torch.device(map_location)
        if target_device.type != "cpu":
            model.to(target_device)
        return model

    def get_optim_params(self) -> list[Tensor]:
        # Return the trainable tensors directly (a single param group). The optimizer
        # builder wraps these in a param group; returning a bare {"params": [...]} dict
        # instead would make `list(...)` yield the key string "params".
        params = (
            list(self.model.dit.parameters()) if hasattr(self.model, "dit") else list(self.model.parameters())
        )
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            params.extend(list(proprio_encoder.parameters()))
        return [p for p in params if p.requires_grad]

    def _freeze_video_expert(self) -> None:
        """Freeze the video expert, including blocks re-parented into MoT layers."""
        video_expert = getattr(self.model, "video_expert", None)
        if video_expert is None:
            raise RuntimeError("FastWAM LoRA requires a video expert")
        video_expert.requires_grad_(False)
        mot = getattr(self.model, "mot", None)
        if mot is not None and getattr(mot, "layers", None) is not None:
            for layer in mot.layers:
                if "video" in layer.blocks:
                    layer.blocks["video"].requires_grad_(False)

    def reset(self) -> None:
        self._action_queue: deque[Tensor] = deque([], maxlen=self.config.n_action_steps)

    def _batch_to_training_sample(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Adapt a standard LeRobot batch to the FastWAM-native sample that
        `FastWAM.build_inputs` consumes (`video`, `action`, `context`/`context_mask`,
        per-frame `proprio`).

        The LeRobot training loop passes raw `observation.images.*`, a single-step
        `observation.state` `[B, D]`, `action`, and a language `task` string. We do
        only the translation `build_inputs` can't: stack the camera frames into a
        video, encode the prompt with the (frozen) text encoder (mirroring inference,
        so language-conditioned datasets need no precomputed context), and give proprio
        the per-frame axis `build_inputs` indexes. All shape/presence validation is
        left to `build_inputs`, the single authority on the contract.
        """
        sample = dict(batch)
        if "video" not in sample:
            sample["video"] = _stack_video_from_images(batch, self.config)
        has_context = "context" in sample
        has_context_mask = "context_mask" in sample
        if has_context != has_context_mask:
            raise KeyError("FastWAM batches must provide both `context` and `context_mask`.")
        if not has_context:
            prompt = _prompt_from_batch(batch=batch, config=self.config)
            if prompt is None:
                raise KeyError("FastWAM training requires a `task`/`prompt` or cached context tensors.")
            if getattr(self.model, "text_context_artifact", None) is not None:
                sample["context"], sample["context_mask"] = self.model.encode_cached_prompt(prompt)
            else:
                sample["context"], sample["context_mask"] = self.model.encode_prompt(prompt)
        if self.config.proprio_dim is not None and "proprio" not in sample:
            state = sample.get(OBS_STATE)
            if state is not None:
                # LeRobot gives a single-step state [B, D]; build_inputs expects
                # per-frame [B, T, D] and uses frame 0, so add a T=1 axis.
                sample["proprio"] = state.unsqueeze(1) if state.ndim == 2 else state
        return sample

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Any]]:
        """Compute FastWAM training loss for a LeRobot batch.

        Args:
            batch (dict[str, Tensor]): Batch containing FastWAM-ready keys
                (`video`, `action`, `context`, `context_mask`) or LeRobot keys
                that can be adapted (`observation.images.*`, `observation.state`,
                `action`, `action_is_pad`).

        Returns:
            tuple[Tensor, dict[str, Any]]: The scalar loss to backprop, and a dict of
            logging metrics (e.g. `loss_video`, `loss_action`) — the `(loss, output_dict)`
            contract the LeRobot training loop expects.
        """

        sample = self._batch_to_training_sample(batch)
        loss, metrics = self.model.training_loss(sample)
        return loss, dict(metrics or {})

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **_: Any) -> Tensor:
        """Predict a chunk of actions from the current FastWAM observation.

        Args:
            batch (dict[str, Tensor]): Inference batch with `input_image` or
                image observation keys, plus `context/context_mask` or `prompt`.

        Returns:
            Tensor: Action chunk with shape `[B, action_horizon, action_dim]`.
        """

        self.eval()
        infer_kwargs = _batch_to_infer_kwargs(batch=batch, config=self.config)
        if (
            getattr(self.model, "text_context_artifact", None) is not None
            and infer_kwargs["prompt"] is not None
        ):
            infer_kwargs["context"], infer_kwargs["context_mask"] = self.model.encode_cached_prompt(
                infer_kwargs["prompt"]
            )
            infer_kwargs["prompt"] = None
        batch_size = _infer_kwargs_batch_size(infer_kwargs)
        if batch_size == 1:
            action = _action_from_model_output(self.model.infer_action(**infer_kwargs))
        else:
            action = torch.cat(
                [
                    _action_from_model_output(
                        self.model.infer_action(
                            **_slice_infer_kwargs(infer_kwargs, index=i, batch_size=batch_size)
                        )
                    )
                    for i in range(batch_size)
                ],
                dim=0,
            )
        return action.to(device=batch_device(batch), dtype=torch.float32)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def _build_core_model(self, config: FastWAMConfig) -> FastWAM:
        """Build the FastWAM core for training / inference.

        Only the trainable parts (the MoT DiT and the proprio encoder) are
        materialized empty here and then filled from the policy's
        `model.safetensors` by the base `from_pretrained`. The *frozen* Wan2.2 VAE
        and UMT5 text encoder are loaded with their real weights from the
        `Wan-AI/Wan2.2-TI2V-5B-Diffusers` repo (cached in the HF cache, shared
        across checkpoints) and are intentionally excluded from `model.safetensors`
        — see `FastWAM.__init__`. The tokenizer comes from `google/umt5-xxl`.
        """
        dtype = _dtype_from_name(config.torch_dtype)
        # Build the complete core on CPU so neither model construction nor checkpoint loading
        # needs a second copy of the 6B-parameter state on the 24 GB training GPU.
        construction_device = "cpu"
        text_context_artifact = getattr(self, "_preloaded_text_context", None)
        if text_context_artifact is None and config.text_context_path:
            text_context_artifact = load_text_context_artifact(
                config.text_context_path,
                expected_provenance=_expected_text_context_provenance(config),
            )
        video_expert = WanVideoDiT(**config.video_dit_config).to(device=construction_device, dtype=dtype)
        action_expert = ActionDiT(**config.action_dit_config).to(device=construction_device, dtype=dtype)
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=config.mot_checkpoint_mixed_attn,
        )
        text_encoder = (
            load_pretrained_wan_text_encoder(
                model_id=config.text_encoder_model_id, torch_dtype=dtype, device=construction_device
            )
            if config.load_text_encoder
            else None
        )
        return FastWAM(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=load_pretrained_wan_vae(torch_dtype=dtype, device=construction_device),
            text_encoder=text_encoder,
            tokenizer=(
                build_wan_tokenizer(
                    model_id=config.tokenizer_model_id, tokenizer_max_len=config.tokenizer_max_len
                )
                if config.load_text_encoder
                else None
            ),
            text_context_artifact=text_context_artifact,
            text_dim=int(config.video_dit_config["text_dim"]),
            proprio_dim=config.proprio_dim,
            device=construction_device,
            torch_dtype=dtype,
            video_train_shift=float(config.video_scheduler["train_shift"]),
            video_infer_shift=float(config.video_scheduler["infer_shift"]),
            video_num_train_timesteps=int(config.video_scheduler["num_train_timesteps"]),
            action_train_shift=float(config.action_scheduler["train_shift"]),
            action_infer_shift=float(config.action_scheduler["infer_shift"]),
            action_num_train_timesteps=int(config.action_scheduler["num_train_timesteps"]),
            loss_lambda_video=float(config.loss["lambda_video"]),
            loss_lambda_action=float(config.loss["lambda_action"]),
        )


def _expected_text_context_provenance(config: FastWAMConfig) -> TextContextProvenance:
    return TextContextProvenance(
        model_id=config.model_id,
        tokenizer_model_id=config.tokenizer_model_id,
        text_encoder_model_id=config.text_encoder_model_id,
        dtype=config.torch_dtype,
        prompt_template=config.prompt_template,
        tokenizer_max_len=config.tokenizer_max_len,
        context_len=config.context_len,
        context_dim=int(config.video_dit_config["text_dim"]),
    )


def _dataset_task_strings(dataset_meta: Any) -> list[str]:
    tasks = getattr(dataset_meta, "tasks", None)
    if tasks is None:
        return []
    try:
        return [str(task) for task in tasks.index.tolist()]
    except AttributeError:
        return [str(task) for task in tasks]


def _scalar(value: Any) -> Any:
    """Unwrap a 0-/1-element tensor (e.g. from DataLoader collation) to a Python scalar."""
    return value.item() if isinstance(value, Tensor) else value


def _batch_to_infer_kwargs(batch: dict[str, Tensor], config: FastWAMConfig) -> dict[str, Any]:
    return {
        "prompt": _prompt_from_batch(batch=batch, config=config),
        "input_image": _input_image_from_batch(batch, config),
        "action_horizon": config.action_horizon,
        "proprio": batch.get("proprio", batch.get(OBS_STATE)),
        "context": batch.get("context"),
        "context_mask": batch.get("context_mask"),
        "negative_prompt": batch.get("negative_prompt", config.negative_prompt),
        "text_cfg_scale": float(_scalar(batch.get("text_cfg_scale", config.text_cfg_scale))),
        "num_inference_steps": int(_scalar(batch.get("num_inference_steps", config.num_inference_steps))),
        "sigma_shift": batch.get("sigma_shift", config.sigma_shift),
        "seed": batch.get("seed", config.inference_seed),
        "rand_device": batch.get("rand_device", config.rand_device),
        "tiled": bool(batch.get("tiled", config.tiled)),
    }


def _prompt_from_batch(batch: dict[str, Tensor], config: FastWAMConfig) -> Any:
    prompt = batch.get("prompt")
    if prompt is not None:
        return prompt

    task = batch.get("task")
    if task is None:
        return None
    if isinstance(task, str):
        return config.prompt_template.format(task=task)
    if isinstance(task, (list, tuple)):
        return [config.prompt_template.format(task=str(item)) for item in task]
    return config.prompt_template.format(task=str(task))


def _action_from_model_output(output: Any) -> Tensor:
    action = output["action"] if isinstance(output, dict) else output
    if action.ndim == 2:
        action = action.unsqueeze(0)
    return action


def _infer_kwargs_batch_size(infer_kwargs: dict[str, Any]) -> int:
    image = infer_kwargs["input_image"]
    if not isinstance(image, Tensor):
        raise TypeError(f"`input_image` must be a tensor, got {type(image).__name__}.")
    if image.ndim == 3:
        return 1
    if image.ndim == 4:
        return int(image.shape[0])
    raise ValueError(f"`input_image` must be [B,C,H,W] or [C,H,W], got {tuple(image.shape)}.")


def _slice_infer_kwargs(infer_kwargs: dict[str, Any], *, index: int, batch_size: int) -> dict[str, Any]:
    return {
        key: _slice_infer_value(value, index=index, batch_size=batch_size)
        for key, value in infer_kwargs.items()
    }


def _slice_infer_value(value: Any, *, index: int, batch_size: int) -> Any:
    if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
        return value[index : index + 1]
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        return value[index]
    return value


def _dtype_from_name(name: str) -> torch.dtype:
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if name not in dtype_map:
        raise ValueError(f"Unsupported torch dtype `{name}`.")
    return dtype_map[name]


def batch_device(batch: dict[str, Any]) -> torch.device:
    for value in batch.values():
        if isinstance(value, Tensor):
            return value.device
    return torch.device("cpu")


def _resize_frames(frames: Tensor, size: tuple[int, int]) -> Tensor:
    """Resize a frame tensor to `size` (H, W), tolerating a leading temporal/batch stack.

    `interpolate` only accepts a single leading batch dim (`[N, C, H, W]`), but FastWAM camera
    tensors arrive as `[B, C, H, W]` (live eval) or `[B, T, C, H, W]` (temporal stack), so flatten
    any leading dims into the batch, resize, then restore. A no-op when already at `size`.
    """
    if tuple(frames.shape[-2:]) == size:
        return frames
    lead = frames.shape[:-3]
    flat = frames.reshape(-1, *frames.shape[-3:])
    flat = torch.nn.functional.interpolate(
        flat, size=size, mode="bilinear", align_corners=False, antialias=True
    )
    return flat.reshape(*lead, *flat.shape[-3:])


def _stack_video_from_images(batch: dict[str, Tensor], config: FastWAMConfig) -> Tensor:
    # Exclude the `*_is_pad` companion tensors that delta-timestamp loading adds alongside
    # each camera (shape [B, T]); they share the `observation.images.` prefix but are not frames.
    image_keys = sorted(k for k in batch if k.startswith("observation.images.") and not k.endswith("_is_pad"))
    if not image_keys:
        raise KeyError("FastWAM batch must contain `video` or `observation.images.*` keys.")
    per_cam = (int(config.image_size[0]), int(config.image_size[1]) // len(image_keys))
    images = [_resize_frames(batch[key], per_cam) for key in image_keys]
    # Cameras concatenate along width (last dim) in both the single-frame and temporal case.
    image = torch.cat(images, dim=-1) if len(images) > 1 else images[0]
    if image.ndim == 4:
        # [B, C, H, W]: a single frame (e.g. the live eval observation) -> repeat across time.
        image = image.unsqueeze(2).repeat(1, 1, config.model_video_frames, 1, 1)
    elif image.ndim == 5:
        # [B, T, C, H, W]: temporal stack from delta-timestamp loading -> [B, C, T, H, W].
        image = image.permute(0, 2, 1, 3, 4)
    else:
        raise ValueError(f"Expected image batch [B,C,H,W] or temporal [B,T,C,H,W], got {tuple(image.shape)}.")
    return image


def _input_image_from_batch(batch: dict[str, Tensor], config: FastWAMConfig) -> Tensor:
    if "input_image" in batch:
        return _prepare_infer_image(batch["input_image"], config)
    video = batch.get("video")
    if video is None:
        video = _stack_video_from_images(batch, config)
    if video.ndim == 5:
        return _prepare_infer_image(video[:, :, 0], config)
    if video.ndim == 4:
        return _prepare_infer_image(video, config)
    raise ValueError(f"Cannot build input image from tensor with shape {tuple(video.shape)}.")


def _prepare_infer_image(image: Tensor, config: FastWAMConfig) -> Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected image tensor [B,C,H,W] or [C,H,W], got {tuple(image.shape)}.")

    # Resize to the full configured resolution (no-op when the video path already produced it, but
    # also covers a directly-supplied `input_image`). The model owns its input resolution — see
    # `_stack_video_from_images` — so we resize rather than assert on a mismatch.
    target_h, target_w = int(config.image_size[0]), int(config.image_size[1])
    return _resize_frames(image, (target_h, target_w))
