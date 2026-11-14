"""PreTrainedPolicy adapter for frozen Cosmos/LTX features and SmolExpert actions."""

from __future__ import annotations

import json
import os
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


def _resolve_cosmos3_checkpoint_path(configured: str | Path, *, backend: str = "cosmos3_edge") -> Path:
    """Resolve Cosmos 3 Edge/Nano checkpoint path across machines and HuggingFace caches."""
    configured_path = Path(configured)
    if configured_path.exists():
        return configured_path

    is_nano = backend.replace("-", "_").lower() == "cosmos3_nano"
    cache_subdir = "cosmos3-nano" if is_nano else "cosmos3-edge"
    hub_repo_dir = "models--nvidia--Cosmos3-Nano" if is_nano else "models--nvidia--Cosmos3-Edge"

    default_cache = Path.home() / ".cache/video-vam" / cache_subdir
    if (default_cache / "transformer").is_dir():
        return default_cache

    hub_dirs: list[Path] = []
    if hf_home := os.environ.get("HF_HOME"):
        hub_dirs.append(Path(hf_home) / "hub")
    if hf_hub_cache := os.environ.get("HF_HUB_CACHE"):
        hub_dirs.append(Path(hf_hub_cache))
    hub_dirs.append(Path.home() / ".cache/huggingface/hub")
    hub_dirs.append(Path.home() / ".cache/pysandbox-lerobot/huggingface/hub")

    for hub_dir in hub_dirs:
        snapshots_dir = hub_dir / hub_repo_dir / "snapshots"
        if not snapshots_dir.is_dir():
            continue
        for snap in sorted(snapshots_dir.iterdir()):
            if snap.is_dir() and (snap / "transformer").is_dir():
                return snap

    return configured_path


def _load_normalizer(run_dir: Path, checkpoint_metadata: dict[str, Any]) -> SmolVLANormalizer:
    norm_path = run_dir / "normalizer.safetensors"
    if norm_path.is_file():
        tensors = load_file(str(norm_path), device="cpu")
    else:
        norm_json = json.loads((run_dir / "normalizer.json").read_text())
        tensors = {
            k: torch.tensor(v, dtype=torch.float32) for k, v in norm_json.items() if isinstance(v, list)
        }
    required = {"state_mean", "state_std", "action_mean", "action_std"}
    if not required <= set(tensors.keys()):
        raise ValueError(f"normalizer keys mismatch: {sorted(tensors)}")
    return SmolVLANormalizer(**{key: tensors[key].float() for key in required})


def restore_uint8_history(history: Tensor) -> Tensor:
    """Invert rollout uint8-to-float preparation and ensure [B, 3, 5, 480, 640] uint8."""

    if history.ndim == 4:
        # [B, 3, H, W] -> repeat 5 frames along temporal dimension
        history = history.unsqueeze(2).expand(-1, -1, 5, -1, -1).contiguous()
    if history.ndim != 5 or history.shape[1] != 3 or history.shape[2] != 5:
        raise ValueError(f"history must have shape [B, 3, 5, H, W], got {tuple(history.shape)}")
    if tuple(history.shape[3:]) != (480, 640):
        b, c, t, h, w = history.shape
        flat = history.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
        resized = torch.nn.functional.interpolate(flat, size=(480, 640), mode="bilinear", align_corners=False)
        history = resized.reshape(b, t, c, 480, 640).permute(0, 2, 1, 3, 4).contiguous()
        if history.amax().item() <= 1.0:
            return (history * 255.0).round().clamp(0, 255).to(dtype=torch.uint8)
        return history.round().clamp(0, 255).to(dtype=torch.uint8)
    if history.dtype is torch.uint8:
        return history
    if not torch.is_floating_point(history) or not torch.isfinite(history).all().item():
        raise TypeError("prepared image history must be finite floating point or uint8")
    if history.amin().item() < -1e-3 or history.amax().item() > 1.0 + 1e-3:
        if history.amin().item() >= 0.0 and history.amax().item() <= 255.0:
            return history.round().clamp(0, 255).to(dtype=torch.uint8)
        raise ValueError("prepared image history must be in [0, 1]")
    scaled = history.float().clamp(0.0, 1.0) * 255.0
    return scaled.round().clamp(0, 255).to(dtype=torch.uint8)


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
        self._frame_history: deque[Tensor] = deque(maxlen=5)
        self._auto_frame_index: int = 0
        self.rtc_processor: RTCProcessor | None = None
        self._closed = False
        self._ltx_context_transform: str | None = None
        self.jepa_world_model: Any | None = None

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
        if (run_dir / "best.json").is_file():
            metadata = _read_json(run_dir / "best.json")
            self._validate_checkpoint_contract(metadata)
        else:
            manifest_steps = 10
            if (run_dir / "run_manifest.json").is_file():
                manifest = _read_json(run_dir / "run_manifest.json")
                manifest_steps = int(manifest.get("policy", {}).get("num_steps", 10))
                manifest_prompt = manifest.get("arguments", {}).get("backbone_prompt")
                if isinstance(manifest_prompt, str) and manifest_prompt:
                    config.cosmos3_prompt = manifest_prompt
            metadata = {
                "expert_checkpoint": "lerobot/smolvla_base",
                "hyperparameters": {"vlm_config": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"},
                "action_semantics": {"euler_steps": manifest_steps},
            }
        normalizer = _load_normalizer(run_dir, metadata)
        input_channels = 2048 if config.backend in ("cosmos", "cosmos3_edge") else 4096
        self.decoder = SmolExpertActionDecoder.from_training_checkpoint(
            run_dir / "best.safetensors",
            normalizer=normalizer,
            expert_checkpoint=str(metadata.get("expert_checkpoint", "lerobot/smolvla_base")),
            vlm_config_name=str(
                metadata.get("hyperparameters", {}).get(
                    "vlm_config", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
                )
            ),
            device=config.device,
            num_steps=(
                config.action_euler_steps
                if config.action_euler_steps is not None
                else int(metadata.get("action_semantics", {}).get("euler_steps", 10))
            ),
            input_channels=input_channels,
        ).eval()
        should_compile = getattr(config, "cosmos_torch_compile", False)
        if should_compile and str(config.device).startswith("cuda") and hasattr(torch, "compile"):
            torch.set_float32_matmul_precision("high")
            self.decoder._vector_field = torch.compile(self.decoder._vector_field, mode="default")
        self.extractor, self.prompt_embedding = self._build_extractor()
        if config.jepa_world_model_checkpoint is not None and config.jepa_planning_mode != "none":
            from safetensors.torch import load_file

            from .jepa_world_model import ControlAwareJEPAWorldModel, JEPAWorldModelConfig

            wm = ControlAwareJEPAWorldModel(JEPAWorldModelConfig(vis_dim=input_channels)).to(config.device)
            wm.load_state_dict(load_file(str(config.jepa_world_model_checkpoint), device=str(config.device)))
            self.jepa_world_model = wm.eval()

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
        local_path = Path(pretrained_name_or_path)
        if not local_path.exists():
            from huggingface_hub import snapshot_download

            local_path = Path(snapshot_download(str(pretrained_name_or_path)))
        if config is None:
            config = VideoVAMConfig.from_pretrained(local_path)
        config.pretrained_path = local_path
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
                injection.get(key) in (["B", 4800, 2048], ["B", 600, 2048])
                for key in ("cosmos_input_shape", "input_shape")
            ):
                raise ValueError("Cosmos checkpoint input shape must be [B, 4800, 2048] or [B, 600, 2048]")
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
            from .cosmos_lora import merge_lora_file_into_base
            from .cosmos_predict2_extractor import CosmosPredict2Extractor, CosmosPredict2ExtractorConfig
            from .cosmos_prompt_embedding import load_prompt_embedding

            prompt = load_prompt_embedding(self.config.cosmos_prompt).embedding
            should_compile = getattr(self.config, "cosmos_torch_compile", False)
            compile_mode = getattr(self.config, "cosmos_compile_mode", "default")
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
                    torch_compile=False,
                    compile_mode=compile_mode,
                    use_cuda_graphs=self.config.cosmos_use_cuda_graphs,
                    state_t=self.config.cosmos_state_t,
                    fp8_linear=self.config.cosmos_fp8_linear,
                    vae_input_mode="observed_prefix",
                )
            )
            if (
                self.config.cosmos_lora_weights is not None
                and Path(self.config.cosmos_lora_weights).is_file()
            ):
                merge_lora_file_into_base(extractor.backbone, self.config.cosmos_lora_weights)
            if should_compile and extractor.device.type == "cuda":
                extractor.backbone = torch.compile(
                    extractor.backbone,
                    mode=compile_mode,
                )
                if hasattr(extractor.tokenizer, "model") and hasattr(extractor.tokenizer.model, "model"):
                    inner_vae = extractor.tokenizer.model.model
                    if hasattr(inner_vae, "encoder") and not isinstance(
                        inner_vae.encoder, torch._dynamo.eval_frame.OptimizedModule
                    ):
                        inner_vae.encoder = torch.compile(inner_vae.encoder, mode="default")
            return extractor, prompt

        elif self.config.backend in ("cosmos3_edge", "cosmos3_nano"):
            from .cosmos3_features import Cosmos3ExtractorConfig, Cosmos3FeatureExtractor

            is_nano = self.config.backend == "cosmos3_nano"
            run_dir = Path(self.config.pretrained_path) if self.config.pretrained_path is not None else None
            lora_ckpt = self.config.cosmos3_lora_weights
            if lora_ckpt is not None and not Path(lora_ckpt).exists():
                repo_root = Path(__file__).resolve().parents[4]
                if (repo_root / lora_ckpt).exists():
                    lora_ckpt = repo_root / lora_ckpt
                elif run_dir is not None and (run_dir / Path(lora_ckpt).name).exists():
                    lora_ckpt = run_dir / Path(lora_ckpt).name
            elif lora_ckpt is None and run_dir is not None and (run_dir / "best_backbone_lora.safetensors").is_file():
                lora_ckpt = run_dir / "best_backbone_lora.safetensors"

            default_layer = 18 if is_nano else 20
            tap_layer = int(getattr(self.config, "cosmos3_layer", None) or default_layer)
            extractor = Cosmos3FeatureExtractor(
                Cosmos3ExtractorConfig(
                    backbone_name="cosmos3-nano" if is_nano else "cosmos3-edge",
                    checkpoint_path=_resolve_cosmos3_checkpoint_path(
                        self.config.cosmos3_checkpoint, backend=self.config.backend
                    ),
                    hidden_layers=(tap_layer,),
                    device=str(self.config.device),
                    dtype="bfloat16",
                    fps=float(getattr(self.config, "cosmos3_fps", 10.0)),
                    prompt=getattr(self.config, "cosmos3_prompt", "take cube out of box"),
                    lora_checkpoint=lora_ckpt,
                    lora_rank=getattr(self.config, "cosmos3_lora_rank", None) or 16,
                    lora_alpha=getattr(self.config, "cosmos3_lora_alpha", None) or 32.0,
                    lora_use_dora=bool(getattr(self.config, "cosmos3_lora_use_dora", False)),
                    compile_dit=bool(getattr(self.config, "cosmos_torch_compile", False)),
                    compile_vae=bool(getattr(self.config, "cosmos_torch_compile", False)),
                )
            )
            return extractor, torch.zeros(1, 1, device=self.config.device)

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
            return getattr(self.config, "feature_seed_global", 0)
        frame = batch.get(HISTORY_FRAME_INDEX_KEY)
        if not isinstance(frame, Tensor) or frame.numel() != 1:
            if self.config.backend in ("cosmos3_edge", "cosmos3_nano"):
                return getattr(self.config, "feature_seed_global", 0)
            frame_val = self._auto_frame_index
            self._auto_frame_index += 1
        else:
            frame_val = int(frame.item())
        frame_index = self.config.feature_seed_frame_offset + frame_val
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
        if self.config.backend in ("cosmos3_edge", "cosmos3_nano"):
            extraction = self.extractor.extract(rgb_frames=images)
            context = extraction.features
            expected = (1, 600, 4096 if self.config.backend == "cosmos3_nano" else 2048)
        elif self.config.backend == "cosmos":
            prompt = self.prompt_embedding
            if prompt.shape[0] != 1:
                raise ValueError("frozen prompt artifact must have batch size one")
            extraction = self.extractor.extract(images, prompt, noise_seed=feature_seed)
            if (
                getattr(self.config, "cosmos_context_transform", "pool2") == "none"
                or self.config.cosmos_state_t == 2
            ):
                context = extraction.tokens
                expected = (1, 2400, 2048)
            else:
                context = apply_context_transform(extraction.tokens, "pool2")
                expected = (1, 600 if self.config.cosmos_state_t == 2 else 4800, 2048)
        else:
            from .ltx_action import pool2_ltx_context

            prompt = self.prompt_embedding
            if prompt.shape[0] != 1:
                raise ValueError("frozen prompt artifact must have batch size one")
            extraction = self.extractor.extract(images, prompt, noise_seed=feature_seed)
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
        if self.config.joint_limits_min is None or self.config.joint_limits_max is None:
            return actions
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

    def _update_frame_history(self, batch: dict[str, Tensor]) -> Tensor | None:
        single_frame = batch.get(self.config.camera_key)
        if not isinstance(single_frame, Tensor):
            return None
        if single_frame.ndim == 3 and single_frame.shape[-1] == 3:
            single_frame = single_frame.permute(2, 0, 1).unsqueeze(0)
        elif single_frame.ndim == 3 and single_frame.shape[0] == 3:
            single_frame = single_frame.unsqueeze(0)
        if not self._frame_history:
            for _ in range(5):
                self._frame_history.append(single_frame)
        else:
            self._frame_history.append(single_frame)
        return torch.stack(list(self._frame_history), dim=2)

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
        _history_already_updated: bool = False,
    ) -> Tensor:
        if self._closed:
            raise RuntimeError("VideoVAMPolicy is closed")
        history_key = f"{self.config.camera_key}.history"
        history = batch.get(history_key)
        if not isinstance(history, Tensor):
            if _history_already_updated and self._frame_history:
                history = torch.stack(list(self._frame_history), dim=2)
            else:
                history = self._update_frame_history(batch)
            if not isinstance(history, Tensor):
                raise ValueError(f"batch is missing {history_key}; use inference.observation_history_size=5")
        state = batch.get(OBS_STATE)
        if isinstance(state, Tensor) and state.ndim == 1 and state.shape[0] == 6:
            state = state.unsqueeze(0)
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
            if prev_chunk_left_over is not None and inference_delay is not None:
                prev_len = prev_chunk_left_over.shape[-2]
                if prev_len > 0:
                    execution_horizon = min(prev_len, max(execution_horizon, int(inference_delay) + 5))
        state_dev = state.to(device=self.config.device, dtype=torch.float32)
        base_seed = self.config.action_seed if action_seed is None else action_seed
        if self.jepa_world_model is not None and self.config.jepa_planning_mode != "none" and noise is None:
            n_cand = max(2, int(self.config.jepa_num_candidates))
            noises = []
            base_n: Tensor | None = None
            for k in range(n_cand):
                gen = torch.Generator(device=self.config.device).manual_seed(base_seed + k * 997)
                n_k = torch.randn((1, 30, 32), generator=gen, device=self.config.device, dtype=torch.float32)
                if base_n is not None:
                    n_k = 0.85 * base_n + 0.35 * n_k
                else:
                    base_n = n_k.clone()
                noises.append(n_k[0])
            noise_batch = torch.stack(noises, dim=0)
            cands_raw = self.decoder.sample_actions(
                state_dev.expand(n_cand, -1),
                context.float().expand(n_cand, -1, -1),
                noise=noise_batch,
                rtc_processor=rtc,
                inference_delay=inference_delay,
                prev_chunk_left_over=prev_chunk_left_over,
                execution_horizon=execution_horizon,
                use_cuda_graph=False,
            )
            norm = self.decoder.normalizer
            s_norm = norm.normalize_state(state_dev)
            cands_norm = norm.normalize_action(cands_raw)
            h_t = self.jepa_world_model.encode(context, s_norm, use_target=False)
            if self.config.jepa_planning_mode == "mppi":
                chosen_norm, _ = self.jepa_world_model.plan_mppi(h_t, s_norm, cands_norm)
                actions = norm.denormalize_action(chosen_norm).unsqueeze(0)
            else:
                scored = self.jepa_world_model.score_action_candidates(h_t, s_norm, cands_norm)
                best_idx = int(scored["score"].argmax().item())
                actions = cands_raw[best_idx : best_idx + 1]
        else:
            actions = self.decoder.sample_actions(
                state_dev,
                context,
                seed=base_seed,
                noise=noise,
                rtc_processor=rtc,
                inference_delay=inference_delay,
                prev_chunk_left_over=prev_chunk_left_over,
                execution_horizon=execution_horizon,
            )
        return self._apply_action_safety(actions)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        history_key = f"{self.config.camera_key}.history"
        history_updated = False
        if not isinstance(batch.get(history_key), Tensor):
            self._update_frame_history(batch)
            history_updated = True
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch, _history_already_updated=history_updated, **kwargs)
            n_steps = min(chunk.shape[1], max(1, int(self.config.n_action_steps)))
            self._action_queue.extend(chunk[:, index] for index in range(n_steps))
        return self._action_queue.popleft()

    def reset(self) -> None:
        self._action_queue.clear()
        self._frame_history.clear()
        self._auto_frame_index = 0
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
