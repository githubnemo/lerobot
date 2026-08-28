"""PreTrainedPolicy adapter for frozen Cosmos/LTX features and SmolExpert actions."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc import RTCProcessor
from lerobot.utils.constants import OBS_STATE

from .configuration_video_vam import ACTION_NAMES, ACTION_UNITS, VideoVAMConfig
from .context_transform import apply_context_transform
from .smol_expert import SmolExpertActionDecoder, SmolVLANormalizer

HISTORY_FRAME_INDEX_KEY = "observation.history_frame_index"


class VideoVAMSafetyError(RuntimeError):
    """Raised before a physically unsafe chunk can enter the rollout queue."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_normalizer(run_dir: Path, checkpoint_metadata: dict[str, Any]) -> SmolVLANormalizer:
    tensors = load_file(str(run_dir / "normalizer.safetensors"), device="cpu")
    required = {"state_mean", "state_std", "action_mean", "action_std"}
    if set(tensors) != required:
        raise ValueError(f"normalizer keys mismatch: {sorted(tensors)}")
    metadata = _read_json(run_dir / "normalizer.json")
    source = metadata.get("source")
    if metadata.get("source_split") != "train" or not isinstance(source, dict):
        raise ValueError("SmolExpert normalizer must declare source_split=train")
    if source.get("episodes") != list(range(32)):
        raise ValueError("SmolExpert normalizer must declare train episodes 0-31")
    if checkpoint_metadata.get("normalizer_source") != source:
        raise ValueError("checkpoint and normalizer provenance disagree")
    for key in required:
        expected = torch.tensor(metadata[key], dtype=torch.float32)
        if not torch.equal(tensors[key].float(), expected):
            raise ValueError(f"normalizer JSON and safetensors disagree for {key}")
    return SmolVLANormalizer(**{key: tensors[key].float() for key in required})


def restore_uint8_history(history: Tensor) -> Tensor:
    """Invert rollout uint8-to-float preparation without changing any pixel."""

    if history.ndim != 5 or tuple(history.shape[1:]) != (3, 5, 480, 640):
        raise ValueError(f"history must have shape [B, 3, 5, 480, 640], got {tuple(history.shape)}")
    if history.dtype is torch.uint8:
        return history
    if not torch.is_floating_point(history) or not torch.isfinite(history).all().item():
        raise TypeError("prepared image history must be finite floating point or uint8")
    if history.amin().item() < 0.0 or history.amax().item() > 1.0:
        raise ValueError("prepared image history must be in [0, 1]")
    scaled = history.float() * 255.0
    rounded = scaled.round()
    max_roundtrip_error = float((scaled - rounded).abs().max())
    if max_roundtrip_error > 1e-4:
        raise ValueError(
            "image history is not a lossless uint8/255 preparation "
            f"(max reconstruction error={max_roundtrip_error})"
        )
    return rounded.to(dtype=torch.uint8)


class VideoVAMPolicy(PreTrainedPolicy):
    """Persistent frozen video extractor plus physical-unit SmolExpert decoder."""

    config_class = VideoVAMConfig
    name = "video_vam"

    def __init__(
        self,
        config: VideoVAMConfig,
        *,
        decoder: nn.Module | None = None,
        extractor: Any | None = None,
        prompt_embedding: Tensor | None = None,
    ) -> None:
        super().__init__(config)
        self._action_queue: deque[Tensor] = deque()
        self.rtc_processor: RTCProcessor | None = None
        self._closed = False
        self._ltx_context_transform: str | None = None

        injected = (decoder is not None, extractor is not None, prompt_embedding is not None)
        if any(injected) and not all(injected):
            raise ValueError("decoder, extractor, and prompt_embedding must be injected together")
        if all(injected):
            assert decoder is not None and extractor is not None and prompt_embedding is not None
            self.decoder = decoder
            self.extractor = extractor
            self.prompt_embedding = prompt_embedding
            return

        run_dir = Path(config.pretrained_path) if config.pretrained_path is not None else None
        if run_dir is None:
            raise ValueError("VideoVAMPolicy requires a pretrained expert run directory")
        metadata = _read_json(run_dir / "best.json")
        self._validate_checkpoint_contract(metadata)
        normalizer = _load_normalizer(run_dir, metadata)
        input_channels = 2048 if config.backend == "cosmos" else 4096
        self.decoder = SmolExpertActionDecoder.from_training_checkpoint(
            run_dir / "best.safetensors",
            normalizer=normalizer,
            expert_checkpoint=str(metadata["expert_checkpoint"]),
            vlm_config_name=str(metadata["hyperparameters"]["vlm_config"]),
            device=config.device,
            num_steps=int(metadata["action_semantics"]["euler_steps"]),
            input_channels=input_channels,
        ).eval()
        self.extractor, self.prompt_embedding = self._build_extractor()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: VideoVAMConfig | None = None,
        **kwargs: Any,
    ) -> VideoVAMPolicy:
        # The run best.safetensors is the decoder custom artifact, not a wrapper
        # model.safetensors. Build the wrapper and let the decoder load it strictly.
        if config is None:
            config = VideoVAMConfig.from_pretrained(pretrained_name_or_path)
        config.pretrained_path = Path(pretrained_name_or_path)
        for ignored in ("revision", "strict", "local_files_only"):
            kwargs.pop(ignored, None)
        if kwargs:
            raise TypeError(f"unexpected VideoVAMPolicy.from_pretrained arguments: {sorted(kwargs)}")
        return cls(config).eval()

    @property
    def type(self) -> str:
        return self.name

    def _validate_checkpoint_contract(self, metadata: dict[str, Any]) -> None:
        semantics = metadata.get("action_semantics", {})
        expected = {"action_dim": 6, "horizon": 30, "internal_action_dim": 32, "euler_steps": 10}
        for key, value in expected.items():
            if semantics.get(key) != value:
                raise ValueError(f"checkpoint action_semantics.{key} must be {value}")
        injection = metadata.get("injection", {})
        if self.config.backend == "cosmos":
            allowed_artifacts = {
                "smolexpert_on_cosmos_training_checkpoint",
                "smolexpert_on_cached_context_training_checkpoint",
            }
            if metadata.get("artifact") not in allowed_artifacts:
                raise ValueError("Cosmos wrapper requires a Cosmos SmolExpert checkpoint")
            if not any(
                injection.get(key) == "pool2"
                for key in ("cosmos_context_transform", "stored_context_transform")
            ):
                raise ValueError("Cosmos checkpoint must declare pool2 context")
            if not any(
                injection.get(key) == ["B", 4800, 2048] for key in ("cosmos_input_shape", "input_shape")
            ):
                raise ValueError("Cosmos checkpoint input shape must be [B, 4800, 2048]")
        else:
            transform = injection.get("stored_context_transform")
            expected_shapes = {
                "pool2": ["B", 640, 4096],
                "none": ["B", 2400, 4096],
            }
            if transform not in expected_shapes:
                raise ValueError("LTX checkpoint stored_context_transform must be one of 'pool2' or 'none'")
            expected_shape = expected_shapes[transform]
            if injection.get("input_shape") != expected_shape:
                raise ValueError(f"LTX checkpoint input shape must be {expected_shape}")
            self._ltx_context_transform = transform

    def _build_extractor(self) -> tuple[Any, Tensor]:
        if self.config.backend == "cosmos":
            from .cosmos_predict2_extractor import CosmosPredict2Extractor, CosmosPredict2ExtractorConfig
            from .cosmos_prompt_embedding import load_prompt_embedding

            prompt = load_prompt_embedding(self.config.cosmos_prompt).embedding
            extractor = CosmosPredict2Extractor(
                CosmosPredict2ExtractorConfig(
                    checkpoint_path=self.config.cosmos_checkpoint,
                    tokenizer_path=self.config.cosmos_tokenizer,
                    device=self.config.device,
                    dtype="bfloat16",
                    hidden_layer=20,
                    stop_after_step=0,
                    high_noise_sigma=80.0,
                    seed=0,
                    attention_backend=self.config.cosmos_attention_backend,
                    compile_friendly=self.config.cosmos_compile_friendly,
                    use_cuda_graphs=self.config.cosmos_use_cuda_graphs,
                    vae_input_mode="observed_prefix",
                )
            )
            return extractor, prompt

        from .ltx_extractor import LTXExtractor, LTXExtractorConfig
        from .ltx_prompt_embedding import load_ltx_prompt_artifact

        prompt = load_ltx_prompt_artifact(self.config.ltx_prompt).embedding
        extractor = LTXExtractor(
            LTXExtractorConfig(
                checkpoint_path=self.config.ltx_transformer,
                video_vae_path=self.config.ltx_vae,
                device=self.config.device,
                dtype="bfloat16",
                hidden_layer=34,
                high_noise_sigma=1.0,
                seed=0,
                input_frames=5,
                padded_input_frames=9,
                offload_mode=self.config.ltx_offload_mode,
                quantization=self.config.ltx_quantization,
                persistent_transformer=True,
                explicit_prefix_execution=True,
                vae_compile_mode=self.config.ltx_vae_compile_mode,
            )
        )
        extractor.open_transformer()
        return extractor, prompt

    def _feature_seed(self, batch: dict[str, Any], explicit_seed: int | None) -> int:
        if explicit_seed is not None:
            if explicit_seed < 0:
                raise ValueError("feature_noise_seed must be non-negative")
            return explicit_seed
        if self.config.feature_seed_episode_index is None:
            raise ValueError("feature_seed_episode_index is required when feature_noise_seed is not supplied")
        frame = batch.get(HISTORY_FRAME_INDEX_KEY)
        if not isinstance(frame, Tensor) or frame.numel() != 1:
            raise ValueError(f"batch must contain scalar {HISTORY_FRAME_INDEX_KEY}")
        frame_index = self.config.feature_seed_frame_offset + int(frame.item())
        if self.config.backend == "cosmos":
            from .cosmos_feature_cache import derive_window_seed
        else:
            from .ltx_extractor import derive_window_seed
        return derive_window_seed(
            self.config.dataset_revision,
            self.config.feature_seed_episode_index,
            frame_index,
            self.config.feature_seed_global,
        )

    def _extract_context(self, images: Tensor, feature_seed: int) -> Tensor:
        if images.shape[0] != 1:
            raise ValueError("Video-VAM hardware rollout supports batch size one only")
        prompt = self.prompt_embedding
        if prompt.shape[0] != 1:
            raise ValueError("frozen prompt artifact must have batch size one")
        extraction = self.extractor.extract(images, prompt, noise_seed=feature_seed)
        if self.config.backend == "cosmos":
            context = apply_context_transform(extraction.tokens, "pool2")
            expected = (1, 4800, 2048)
        else:
            from .ltx_action import pool2_ltx_context

            transform = self._ltx_context_transform
            if transform == "pool2":
                context = pool2_ltx_context(extraction.hidden_grid)
                expected = (1, 640, 4096)
            elif transform == "none":
                hidden_grid = extraction.hidden_grid
                if hidden_grid.ndim != 5 or tuple(hidden_grid.shape[1:]) != (8, 15, 20, 4096):
                    raise ValueError(
                        "LTX hidden grid must have shape [B, 8, 15, 20, 4096] for unpooled context"
                    )
                context = hidden_grid.reshape(hidden_grid.shape[0], 2400, 4096).contiguous()
                expected = (1, 2400, 4096)
            else:
                raise ValueError("LTX context transform was not resolved from checkpoint metadata")
        if tuple(context.shape) != expected:
            raise ValueError(
                f"{self.config.backend} {self._ltx_context_transform} context has shape "
                f"{tuple(context.shape)}, expected {expected}"
            )
        return context.to(device=self.config.device, dtype=torch.bfloat16)

    def _apply_action_safety(self, actions: Tensor) -> Tensor:
        if tuple(actions.shape) != (1, 30, 6):
            raise VideoVAMSafetyError(f"expert returned {tuple(actions.shape)}, expected [1, 30, 6]")
        if not torch.isfinite(actions).all().item():
            raise VideoVAMSafetyError("expert returned non-finite actions")
        self.config._validate_joint_limits(allow_missing=False)
        assert self.config.joint_limits_min is not None and self.config.joint_limits_max is not None
        lower = torch.tensor(self.config.joint_limits_min, device=actions.device, dtype=actions.dtype)
        upper = torch.tensor(self.config.joint_limits_max, device=actions.device, dtype=actions.dtype)
        tolerance = self.config.joint_limit_tolerance
        violation = (actions < lower - tolerance) | (actions > upper + tolerance)
        if violation.any().item():
            step, joint = violation[0].nonzero()[0].tolist()
            value = float(actions[0, step, joint])
            raise VideoVAMSafetyError(
                f"out-of-range action at horizon {step}, {ACTION_NAMES[joint]}={value} {ACTION_UNITS[joint]}; "
                f"configured limits=[{lower[joint].item()}, {upper[joint].item()}]"
            )
        # Clamp only tolerance-sized floating-point excursions. Material violations
        # raise above and never enter either the raw or processed RTC queue.
        return actions.clamp(min=lower.view(1, 1, 6), max=upper.view(1, 1, 6))

    def supports_rtc(self) -> bool:
        return True

    def init_rtc_processor(self) -> None:
        if self.config.rtc_config is None:
            raise ValueError("init_rtc_processor requires config.rtc_config")
        self.rtc_processor = RTCProcessor(self.config.rtc_config)

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        *,
        noise: Tensor | None = None,
        action_seed: int | None = None,
        feature_noise_seed: int | None = None,
        inference_delay: int | None = None,
        prev_chunk_left_over: Tensor | None = None,
    ) -> Tensor:
        if self._closed:
            raise RuntimeError("VideoVAMPolicy is closed")
        history_key = f"{self.config.camera_key}.history"
        history = batch.get(history_key)
        state = batch.get(OBS_STATE)
        if not isinstance(history, Tensor):
            raise ValueError(f"batch is missing {history_key}; use inference.observation_history_size=5")
        if not isinstance(state, Tensor) or tuple(state.shape) != (1, 6):
            raise ValueError(f"{OBS_STATE} must have shape [1, 6]")
        images = restore_uint8_history(history)
        feature_seed = self._feature_seed(batch, feature_noise_seed)
        context = self._extract_context(images, feature_seed)
        rtc = self.rtc_processor if prev_chunk_left_over is not None else None
        if prev_chunk_left_over is not None and rtc is None:
            raise ValueError("prev_chunk_left_over requires init_rtc_processor()")
        execution_horizon = None
        if rtc is not None:
            rtc_config = self.config.rtc_config
            if rtc_config is None:
                raise ValueError("prev_chunk_left_over requires config.rtc_config")
            execution_horizon = rtc_config.execution_horizon
        actions = self.decoder.sample_actions(
            state.to(device=self.config.device, dtype=torch.float32),
            context,
            seed=self.config.action_seed if action_seed is None else action_seed,
            noise=noise,
            rtc_processor=rtc,
            inference_delay=inference_delay,
            prev_chunk_left_over=prev_chunk_left_over,
            execution_horizon=execution_horizon,
        )
        return self._apply_action_safety(actions)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch, **kwargs)
            self._action_queue.extend(chunk[:, index] for index in range(chunk.shape[1]))
        return self._action_queue.popleft()

    def reset(self) -> None:
        self._action_queue.clear()
        if self.rtc_processor is not None:
            self.rtc_processor.reset_tracker()

    def close(self) -> None:
        if self._closed:
            return
        closer = getattr(self.extractor, "close", None)
        if closer is not None:
            closer()
        self._closed = True

    def get_optim_params(self) -> dict:
        return {}

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        del batch
        raise NotImplementedError("VideoVAMPolicy is an inference-only rollout adapter")
