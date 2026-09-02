#!/usr/bin/env python3
"""Open-loop, no-motor rollout rehearsal for cube-out-of-box policies.

The replay reads a held-out episode as a live stream: one RGB frame and one
proprioceptive state arrive per 10 Hz tick, while a five-frame causal camera
buffer is maintained for Cosmos/LTX. Inference runs in one background thread.
The currently active 30-step chunk is drained until the next prediction is
ready, at which point the new chunk replaces it. No robot is imported,
connected, or commanded.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from safetensors.torch import load_file
from torch import Tensor

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT, validate_metadata
from lerobot.policies.vam.action_rmse import EvaluationBatch, SmolVLABackend
from lerobot.policies.vam.context_transform import apply_context_transform
from lerobot.policies.vam.smol_expert import SmolExpertActionDecoder, SmolVLANormalizer
from lerobot.utils.robot_utils import precise_sleep

POLICIES = ("smolvla", "cosmos-smolexpert", "ltx-smolexpert")
JOINT_NAMES = CUBE_OUT_OF_BOX_CONTRACT.action_names
JOINT_UNITS = ("degrees", "degrees", "degrees", "degrees", "degrees", "range_0_100")
HORIZON = CUBE_OUT_OF_BOX_CONTRACT.action_chunk_size
FPS = CUBE_OUT_OF_BOX_CONTRACT.fps
CHUNK_SECONDS = HORIZON / FPS
TRAIN_EPISODES = tuple(range(32))
VAL_EPISODES = tuple(range(32, 40))

DEFAULT_DATASET = Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset")
DEFAULT_SMOLVLA = Path(
    "/home/anton/lerobot-video-vam/outputs/train/"
    "cube_out_of_box_il_smolvla_heldout_0_31_20260819_1hr/"
    "checkpoints/029200/pretrained_model"
)
DEFAULT_COSMOS_RUN = Path(
    "/home/anton/.cache/video-vam/runs/cosmos-scaling-20260825/smolexpert-prefix-retrain"
)
DEFAULT_LTX_RUN = Path("/home/anton/.cache/video-vam/runs/ltx25-smolexpert-20260826/run-a-pool2")
DEFAULT_COSMOS_CHECKPOINT = Path(
    "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt"
)
DEFAULT_COSMOS_TOKENIZER = Path(
    "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth"
)
DEFAULT_COSMOS_PROMPT = Path(
    "/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-t5-11b.safetensors"
)
DEFAULT_LTX_TRANSFORMER = Path(
    "/home/anton/.cache/video-vam/ltx-2.5-models/diffusion_models/"
    "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
)
DEFAULT_LTX_VAE = Path(
    "/home/anton/.cache/video-vam/ltx-2.5-models/vae/ltx-2.5-video-vae-conv-bf16.safetensors"
)
DEFAULT_LTX_PROMPT = Path(
    "/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-ltx25-gemma4.safetensors"
)


@dataclass(frozen=True, slots=True)
class ReplayObservation:
    """One policy input and its anchor-aligned recorded target."""

    episode_index: int
    local_frame_index: int
    absolute_frame_index: int
    rgb_history: Tensor  # [1, 3, 5, 480, 640], uint8
    state: Tensor  # [6], float32
    target_actions: Tensor  # [30, 6], float32
    action_is_pad: Tensor  # [30], bool
    task: str


@dataclass(slots=True)
class TimedPrediction:
    observation: ReplayObservation
    seed: int
    actions: Tensor
    latency_s: float


@dataclass(slots=True)
class ChunkReport:
    prediction: TimedPrediction
    installed_at_local_frame: int | None = None
    executed_actions: list[list[float]] = field(default_factory=list)
    recorded_executed_actions: list[list[float]] = field(default_factory=list)
    execution_local_frames: list[int] = field(default_factory=list)
    starved_after_actions: bool = False


class RolloutBackend(Protocol):
    """Minimal raw-action interface shared by all dry-run policies."""

    name: str
    normalization_audit: dict[str, Any]

    def predict_chunk(self, observation: ReplayObservation, seed: int) -> Tensor:
        """Return one finite [30, 6] chunk in stored physical units."""

    def close(self) -> None:
        """Release accelerator-backed resources."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite_chunk(actions: Tensor, backend: str) -> Tensor:
    actions = actions.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if tuple(actions.shape) != (HORIZON, len(JOINT_NAMES)):
        raise ValueError(f"{backend} returned {tuple(actions.shape)}, expected [30, 6]")
    if not torch.isfinite(actions).all().item():
        raise FloatingPointError(f"{backend} returned non-finite actions")
    return actions


def _load_expert_normalizer(run_dir: Path) -> tuple[SmolVLANormalizer, dict[str, Any]]:
    tensor_path = run_dir / "normalizer.safetensors"
    json_path = run_dir / "normalizer.json"
    tensors = load_file(str(tensor_path), device="cpu")
    required = {"state_mean", "state_std", "action_mean", "action_std"}
    if set(tensors) != required:
        raise ValueError(f"normalizer keys mismatch: {sorted(tensors)}")
    metadata = _json(json_path)
    if metadata.get("source_split") != "train":
        raise ValueError("SmolExpert normalizer is not marked source_split=train")
    source = metadata.get("source")
    if not isinstance(source, dict) or source.get("episodes") != list(TRAIN_EPISODES):
        raise ValueError("SmolExpert normalizer does not declare train episodes 0-31")
    max_json_delta = 0.0
    for key in required:
        expected = torch.tensor(metadata[key], dtype=torch.float32)
        max_json_delta = max(max_json_delta, float((tensors[key].float() - expected).abs().max()))
    if max_json_delta != 0.0:
        raise ValueError(f"normalizer JSON/tensor values differ by {max_json_delta}")
    normalizer = SmolVLANormalizer(**{key: tensors[key].float() for key in required})
    probe = torch.stack((normalizer.state_mean, normalizer.action_mean))
    roundtrip = normalizer.denormalize_action(normalizer.normalize_action(probe))
    audit = {
        "applied_by": "SmolExpertActionDecoder.sample_actions normalize_state + denormalize_action",
        "source_split": metadata["source_split"],
        "source_episodes": source["episodes"],
        "anchor_count": source.get("anchor_count"),
        "json_tensor_max_abs_delta": max_json_delta,
        "normalize_denormalize_roundtrip_max_abs_delta": float((roundtrip - probe).abs().max()),
        "state_mean": metadata["state_mean"],
        "state_std": metadata["state_std"],
        "action_mean": metadata["action_mean"],
        "action_std": metadata["action_std"],
        "status": "pass",
    }
    return normalizer, audit


def _train_stats(dataset_root: Path) -> dict[str, Tensor | int]:
    dataset = LeRobotDataset(
        CUBE_OUT_OF_BOX_CONTRACT.repo_id,
        root=dataset_root,
        episodes=list(TRAIN_EPISODES),
        revision=CUBE_OUT_OF_BOX_CONTRACT.revision,
        download_videos=False,
    )
    raw = dataset.reader.hf_dataset
    if raw is None:
        raise RuntimeError("training dataset reader did not activate")
    actions = torch.stack([raw[index]["action"].float() for index in range(len(raw))])
    states = torch.stack([raw[index]["observation.state"].float() for index in range(len(raw))])
    episodes = torch.tensor([int(raw[index]["episode_index"]) for index in range(len(raw))])
    adjacent = episodes[1:] == episodes[:-1]
    commanded_delta = actions[:-1] - states[:-1]
    observed_motion = states[1:] - states[:-1]
    sign_agreement: list[float] = []
    command_motion_correlation: list[float] = []
    for joint in range(len(JOINT_NAMES)):
        informative = (
            adjacent & (commanded_delta[:, joint].abs() > 0.5) & (observed_motion[:, joint].abs() > 0.05)
        )
        sign_agreement.append(
            float(
                (
                    torch.sign(commanded_delta[informative, joint])
                    == torch.sign(observed_motion[informative, joint])
                )
                .float()
                .mean()
            )
            if informative.any()
            else math.nan
        )
        paired = (
            adjacent & torch.isfinite(commanded_delta[:, joint]) & torch.isfinite(observed_motion[:, joint])
        )
        values = torch.stack((commanded_delta[paired, joint], observed_motion[paired, joint]))
        command_motion_correlation.append(float(torch.corrcoef(values)[0, 1]))
    return {
        "frame_count": len(raw),
        "action_min": actions.amin(dim=0),
        "action_max": actions.amax(dim=0),
        "action_mean": actions.mean(dim=0),
        "action_std": actions.std(dim=0, unbiased=False),
        "state_mean": states.mean(dim=0),
        "state_std": states.std(dim=0, unbiased=False),
        "command_motion_sign_agreement": sign_agreement,
        "command_motion_correlation": command_motion_correlation,
    }


def _smolvla_normalization_audit(checkpoint: Path, train_stats: dict[str, Tensor | int]) -> dict[str, Any]:
    pre_path = checkpoint / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    post_path = checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    pre = load_file(str(pre_path), device="cpu")
    post = load_file(str(post_path), device="cpu")
    keys = ("action.mean", "action.std", "observation.state.mean", "observation.state.std")
    processor_pair_delta = max(float((pre[key] - post[key]).abs().max()) for key in keys)
    train_action_mean = train_stats["action_mean"]
    train_state_mean = train_stats["state_mean"]
    assert isinstance(train_action_mean, Tensor) and isinstance(train_state_mean, Tensor)
    serialized_count = int(pre["action.count"].item())
    train_count = int(train_stats["frame_count"])
    source_is_train_only = serialized_count == train_count
    return {
        "applied_by": "checkpoint PolicyProcessorPipeline normalizer + unnormalizer",
        "processor_pair_max_abs_delta": processor_pair_delta,
        "serialized_action_count": serialized_count,
        "train_frame_count": train_count,
        "dataset_total_frames": CUBE_OUT_OF_BOX_CONTRACT.total_frames,
        "serialized_action_mean": pre["action.mean"].tolist(),
        "train_only_action_mean": train_action_mean.tolist(),
        "action_mean_max_abs_delta_from_train_only": float(
            (pre["action.mean"] - train_action_mean).abs().max()
        ),
        "state_mean_max_abs_delta_from_train_only": float(
            (pre["observation.state.mean"] - train_state_mean).abs().max()
        ),
        "source_is_train_only": source_is_train_only,
        "status": "pass" if source_is_train_only and processor_pair_delta == 0.0 else "fail",
        "finding": (
            "serialized stats are train-only"
            if source_is_train_only
            else "serialized checkpoint stats cover all 6,536 frames, including held-out episodes 32-39"
        ),
    }


class SmolVLARolloutBackend:
    name = "smolvla"

    def __init__(self, checkpoint: Path, train_stats: dict[str, Tensor | int], device: torch.device) -> None:
        self.backend = SmolVLABackend.from_pretrained(str(checkpoint), device=device)
        self.normalization_audit = _smolvla_normalization_audit(checkpoint, train_stats)

    def predict_chunk(self, observation: ReplayObservation, seed: int) -> Tensor:
        current_image = observation.rgb_history[0, :, -1].float().div(255.0)
        raw_observation = {
            CUBE_OUT_OF_BOX_CONTRACT.camera_key: current_image,
            CUBE_OUT_OF_BOX_CONTRACT.state_key: observation.state,
            "task": observation.task,
        }
        batch = EvaluationBatch(
            sample_ids=(f"episode-{observation.episode_index}-frame-{observation.absolute_frame_index}",),
            target_actions=observation.target_actions.unsqueeze(0),
            action_is_pad=observation.action_is_pad.unsqueeze(0),
            current_state=observation.state.unsqueeze(0),
            observations=(raw_observation,),
        )
        return _finite_chunk(self.backend.predict_actions(batch, seed)[0], self.name)

    def close(self) -> None:
        del self.backend
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SmolExpertRolloutBackend:
    """Common expert loader; subclasses supply one frozen video context."""

    name = "smolexpert"
    input_channels: int

    def __init__(self, run_dir: Path, device: torch.device) -> None:
        self.device = device
        self.run_dir = run_dir
        metadata = _json(run_dir / "best.json")
        normalizer, self.normalization_audit = _load_expert_normalizer(run_dir)
        if metadata.get("normalizer_source") != _json(run_dir / "normalizer.json").get("source"):
            raise ValueError("checkpoint and normalizer provenance disagree")
        self.decoder = SmolExpertActionDecoder.from_training_checkpoint(
            run_dir / "best.safetensors",
            normalizer=normalizer,
            expert_checkpoint=str(metadata["expert_checkpoint"]),
            vlm_config_name=str(metadata["hyperparameters"]["vlm_config"]),
            device=device,
            num_steps=int(metadata["action_semantics"]["euler_steps"]),
            input_channels=self.input_channels,
        ).eval()

    def extract_context(self, observation: ReplayObservation) -> Tensor:
        raise NotImplementedError

    @torch.inference_mode()
    def predict_chunk(self, observation: ReplayObservation, seed: int) -> Tensor:
        context = self.extract_context(observation)
        state = observation.state[None, None].to(device=self.device, dtype=torch.float32)
        actions = self.decoder.sample_actions(state, context, seed=seed)
        return _finite_chunk(actions[0], self.name)

    def close(self) -> None:
        del self.decoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class CosmosSmolExpertBackend(SmolExpertRolloutBackend):
    name = "cosmos-smolexpert"
    input_channels = 2048

    def __init__(
        self,
        run_dir: Path,
        device: torch.device,
        checkpoint: Path,
        tokenizer: Path,
        prompt: Path,
        state_t: int = 2,
        context_transform: str = "none",
        lora_weights: Path | None = None,
        merge_lora_weights: Path | None = None,
        compile: bool = True,
    ) -> None:
        from lerobot.policies.vam.cosmos_lora import (
            inject_lora,
            load_lora_state_dict,
            merge_lora_file_into_base,
        )
        from lerobot.policies.vam.cosmos_predict2_extractor import (
            CosmosPredict2Extractor,
            CosmosPredict2ExtractorConfig,
        )
        from lerobot.policies.vam.cosmos_prompt_embedding import load_prompt_embedding

        super().__init__(run_dir, device)
        self.state_t = state_t
        self.context_transform = context_transform
        self.compile = compile
        self.prompt = load_prompt_embedding(prompt).embedding
        config = CosmosPredict2ExtractorConfig(
            checkpoint_path=checkpoint,
            tokenizer_path=tokenizer,
            device=str(device),
            dtype="bfloat16",
            hidden_layer=20,
            stop_after_step=0,
            high_noise_sigma=80.0,
            seed=0,
            state_t=state_t,
            vae_input_mode="observed_prefix",
            torch_compile=compile,
            compile_friendly=compile,
            compile_mode="max-autotune" if compile else "default",
            use_cuda_graphs=compile,
        )
        self.extractor = CosmosPredict2Extractor(config)
        if merge_lora_weights is not None:
            merge_lora_file_into_base(self.extractor.backbone, merge_lora_weights)
        if lora_weights is not None:
            inject_lora(self.extractor.backbone, rank=16, alpha=16.0, block_indices=range(20))
            load_lora_state_dict(self.extractor.backbone, lora_weights)
            self.extractor.backbone.eval()

    def extract_context(self, observation: ReplayObservation) -> Tensor:
        from lerobot.policies.vam.cosmos_feature_cache import derive_window_seed

        feature_seed = derive_window_seed(
            CUBE_OUT_OF_BOX_CONTRACT.revision,
            observation.episode_index,
            observation.absolute_frame_index,
            0,
        )
        extraction = self.extractor.extract(observation.rgb_history, self.prompt, noise_seed=feature_seed)
        context = apply_context_transform(extraction.tokens, self.context_transform)
        return context.to(device=self.device, dtype=torch.bfloat16)

    def predict_chunk(self, observation: ReplayObservation, seed: int) -> Tensor:
        context = self.extract_context(observation)
        state = observation.state[None, None].to(device=self.device, dtype=torch.float32)
        actions = self.decoder.sample_actions(state, context, seed=seed, use_cuda_graph=self.compile)
        return _finite_chunk(actions[0], self.name)

    def close(self) -> None:
        del self.extractor
        super().close()


class LTXSmolExpertBackend(SmolExpertRolloutBackend):
    name = "ltx-smolexpert"
    input_channels = 4096

    def __init__(
        self,
        run_dir: Path,
        device: torch.device,
        transformer: Path,
        video_vae: Path,
        prompt: Path,
    ) -> None:
        from lerobot.policies.vam.ltx_extractor import LTXExtractor, LTXExtractorConfig
        from lerobot.policies.vam.ltx_prompt_embedding import load_ltx_prompt_artifact

        super().__init__(run_dir, device)
        self.prompt = load_ltx_prompt_artifact(prompt).embedding
        self.extractor = LTXExtractor(
            LTXExtractorConfig(
                checkpoint_path=transformer,
                video_vae_path=video_vae,
                device=str(device),
                dtype="bfloat16",
                hidden_layer=34,
                high_noise_sigma=1.0,
                seed=0,
                input_frames=5,
                padded_input_frames=9,
                offload_mode="cpu",
                quantization="fp8-cast",
                persistent_transformer=True,
                explicit_prefix_execution=True,
                vae_compile_mode=None,
            )
        )
        self.extractor.open_transformer()

    def extract_context(self, observation: ReplayObservation) -> Tensor:
        from lerobot.policies.vam.ltx_action import pool2_ltx_context
        from lerobot.policies.vam.ltx_extractor import derive_window_seed

        feature_seed = derive_window_seed(
            CUBE_OUT_OF_BOX_CONTRACT.revision,
            observation.episode_index,
            observation.absolute_frame_index,
            0,
        )
        extraction = self.extractor.extract(observation.rgb_history, self.prompt, noise_seed=feature_seed)
        context = pool2_ltx_context(extraction.hidden_grid)
        if tuple(context.shape) != (1, 640, 4096):
            raise ValueError(f"LTX pool2 context has unexpected shape {tuple(context.shape)}")
        return context.to(device=self.device, dtype=torch.bfloat16)

    def close(self) -> None:
        self.extractor.close()
        del self.extractor
        super().close()


class EpisodeStream:
    """Decode one held-out camera frame at a time and retain a causal history."""

    def __init__(self, dataset_root: Path, episode: int) -> None:
        self.episode = episode
        self.dataset = LeRobotDataset(
            CUBE_OUT_OF_BOX_CONTRACT.repo_id,
            root=dataset_root,
            episodes=[episode],
            revision=CUBE_OUT_OF_BOX_CONTRACT.revision,
            return_uint8=True,
            download_videos=False,
        )
        validate_metadata(self.dataset.meta, CUBE_OUT_OF_BOX_CONTRACT).raise_if_invalid()
        raw = self.dataset.reader.hf_dataset
        if raw is None:
            raise RuntimeError("episode dataset reader did not activate")
        self.raw = raw
        self.actions = torch.stack([raw[index]["action"].float() for index in range(len(raw))])
        self.states = torch.stack([raw[index]["observation.state"].float() for index in range(len(raw))])
        self.absolute_indices = [int(raw[index]["index"]) for index in range(len(raw))]
        self.images: deque[Tensor] = deque(maxlen=CUBE_OUT_OF_BOX_CONTRACT.history_length)

    def __len__(self) -> int:
        return len(self.raw)

    def read(self, local_index: int) -> ReplayObservation | None:
        sample = cast(dict[str, Any], self.dataset[local_index])
        image = sample[CUBE_OUT_OF_BOX_CONTRACT.camera_key]
        if image.dtype != torch.uint8:
            image = image.mul(255).round().clamp(0, 255).to(dtype=torch.uint8)
        if tuple(image.shape) != (3, 480, 640):
            raise ValueError(f"decoded camera frame has unexpected shape {tuple(image.shape)}")
        self.images.append(image.contiguous())
        if len(self.images) < CUBE_OUT_OF_BOX_CONTRACT.history_length:
            return None
        stop = min(local_index + HORIZON, len(self.actions))
        target = self.actions[local_index:stop]
        pad_count = HORIZON - len(target)
        if pad_count:
            target = torch.cat((target, target[-1:].expand(pad_count, -1)), dim=0)
        padding = torch.zeros(HORIZON, dtype=torch.bool)
        if pad_count:
            padding[-pad_count:] = True
        history = torch.stack(tuple(self.images), dim=1).unsqueeze(0)
        return ReplayObservation(
            episode_index=self.episode,
            local_frame_index=local_index,
            absolute_frame_index=self.absolute_indices[local_index],
            rgb_history=history,
            state=self.states[local_index],
            target_actions=target,
            action_is_pad=padding,
            task=CUBE_OUT_OF_BOX_CONTRACT.task,
        )


def _timed_predict(backend: RolloutBackend, observation: ReplayObservation, seed: int) -> TimedPrediction:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    actions = backend.predict_chunk(observation, seed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return TimedPrediction(observation, seed, actions, time.perf_counter() - started)


def _masked_rmse(prediction: Tensor, target: Tensor, padding: Tensor) -> dict[str, Any]:
    valid = ~padding
    if not valid.any().item():
        raise ValueError("RMSE mask contains no valid actions")
    error = prediction.float() - target.float()
    selected = error[valid]
    return {
        "aggregate": float(selected.square().mean().sqrt()),
        "per_joint": selected.square().mean(dim=0).sqrt().tolist(),
        "max_abs": float(selected.abs().max()),
    }


def _chunk_payload(report: ChunkReport) -> dict[str, Any]:
    pred = report.prediction
    observation = pred.observation
    full = _masked_rmse(pred.actions, observation.target_actions, observation.action_is_pad)
    h1 = _masked_rmse(pred.actions[:1], observation.target_actions[:1], observation.action_is_pad[:1])
    prefix_n = min(5, int((~observation.action_is_pad).sum()))
    prefix5 = _masked_rmse(
        pred.actions[:prefix_n], observation.target_actions[:prefix_n], observation.action_is_pad[:prefix_n]
    )
    executed = torch.tensor(report.executed_actions, dtype=torch.float32)
    recorded = torch.tensor(report.recorded_executed_actions, dtype=torch.float32)
    executed_metric = None
    if executed.numel():
        executed_metric = _masked_rmse(
            executed,
            recorded,
            torch.zeros(len(executed), dtype=torch.bool),
        )
    return {
        "anchor": {
            "episode_index": observation.episode_index,
            "local_frame_index": observation.local_frame_index,
            "absolute_frame_index": observation.absolute_frame_index,
        },
        "seed": pred.seed,
        "latency_s": pred.latency_s,
        "latency_fits_30_step_horizon": pred.latency_s < CHUNK_SECONDS,
        "latency_margin_s": CHUNK_SECONDS - pred.latency_s,
        "installed_at_local_frame": report.installed_at_local_frame,
        "anchor_aligned_rmse": {"full_horizon": full, "h1": h1, "first5": prefix5},
        "predicted_actions": pred.actions.tolist(),
        "recorded_anchor_actions": observation.target_actions.tolist(),
        "recorded_anchor_action_is_pad": observation.action_is_pad.tolist(),
        "executed_prefix_length": len(report.executed_actions),
        "execution_local_frames": report.execution_local_frames,
        "executed_actions": report.executed_actions,
        "recorded_time_aligned_actions": report.recorded_executed_actions,
        "executed_time_aligned_rmse": executed_metric,
        "starved_after_actions": report.starved_after_actions,
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _summarize(
    args: argparse.Namespace,
    backend: RolloutBackend,
    train_stats: dict[str, Tensor | int],
    reports: list[ChunkReport],
    seams: list[dict[str, Any]],
    determinism: dict[str, Any],
    elapsed_s: float,
    starved_ticks: int,
) -> dict[str, Any]:
    chunks = [_chunk_payload(report) for report in reports]
    latencies = [report.prediction.latency_s for report in reports]
    steady_latencies = latencies[1:]
    all_predictions = torch.cat([report.prediction.actions for report in reports])
    action_min = train_stats["action_min"]
    action_max = train_stats["action_max"]
    assert isinstance(action_min, Tensor) and isinstance(action_max, Tensor)
    below = all_predictions < action_min
    above = all_predictions > action_max
    excursion = below | above
    per_joint_excursions = excursion.sum(dim=0)
    installed = [report for report in reports if report.installed_at_local_frame is not None]
    seam_norms = [float(item["aggregate_rmse"]) for item in seams]
    aligned_seam_norms = [
        float(item["delay_aligned_aggregate_rmse"])
        for item in seams
        if item.get("delay_aligned_aggregate_rmse") is not None
    ]
    return {
        "schema_version": 2,
        "protocol": {
            "mode": "open-loop recorded-observation replay; no robot connection or motor commands",
            "task_success_measured": False,
            "policy": args.policy,
            "episode": args.episode,
            "held_out_validation_episode": args.episode in VAL_EPISODES,
            "fps": FPS,
            "history_frames": 5,
            "action_horizon": HORIZON,
            "chunk_duration_s": CHUNK_SECONDS,
            "scheduler": (
                "background replan after a fresh 10 Hz observation; active chunk drains until replacement is ready"
            ),
            "seed": args.seed,
            "max_control_frames": args.max_control_frames,
            "max_predictions": args.max_predictions,
            "partial_replay": args.max_control_frames is not None,
        },
        "units_and_order": {
            "action_names": list(JOINT_NAMES),
            "units": list(JOINT_UNITS),
            "metadata_names_match_state_names": CUBE_OUT_OF_BOX_CONTRACT.action_names
            == CUBE_OUT_OF_BOX_CONTRACT.state_names,
            "train_command_to_next_state_sign_agreement": train_stats["command_motion_sign_agreement"],
            "train_command_to_next_state_correlation": train_stats["command_motion_correlation"],
            "finding": (
                "SOFollower defaults to degrees for five arm joints; gripper uses calibrated range_0_100. "
                "meta/info.json records names and order but does not encode units."
            ),
        },
        "normalization": backend.normalization_audit,
        "latency": {
            "per_chunk_s": latencies,
            "cold_bootstrap_s": latencies[0],
            "mean_s": statistics.fmean(latencies),
            "steady_mean_s": statistics.fmean(steady_latencies) if steady_latencies else None,
            "steady_p95_s": _percentile(steady_latencies, 0.95) if steady_latencies else None,
            "steady_max_s": max(steady_latencies) if steady_latencies else None,
            "steady_all_fit_chunk_horizon": (
                all(value < CHUNK_SECONDS for value in steady_latencies) if steady_latencies else None
            ),
            "steady_minimum_margin_s": (CHUNK_SECONDS - max(steady_latencies) if steady_latencies else None),
            "steady_capacity_replan_hz": (
                1.0 / statistics.fmean(steady_latencies) if steady_latencies else None
            ),
            "p95_s": _percentile(latencies, 0.95),
            "max_s": max(latencies),
            "all_fit_chunk_horizon": all(value < CHUNK_SECONDS for value in latencies),
            "minimum_margin_s": CHUNK_SECONDS - max(latencies),
            "installed_chunks": len(installed),
            "effective_replan_hz": max(len(installed) - 1, 0) / elapsed_s if elapsed_s > 0 else 0.0,
            "execution_elapsed_s": elapsed_s,
            "starved_control_ticks": starved_ticks,
        },
        "range_safety": {
            "training_episodes": list(TRAIN_EPISODES),
            "observed_training_action_min": action_min.tolist(),
            "observed_training_action_max": action_max.tolist(),
            "predicted_min": all_predictions.amin(dim=0).tolist(),
            "predicted_max": all_predictions.amax(dim=0).tolist(),
            "excursion_count": int(excursion.sum()),
            "per_joint_excursion_count": per_joint_excursions.tolist(),
            "physical_joint_limits": None,
            "physical_limit_status": (
                "not verified: no SO-101 calibration file or explicit joint-limit config is present on abakus"
            ),
        },
        "determinism_fixed_noise": determinism,
        "seams": {
            "count": len(seams),
            "items": seams,
            "aggregate_rmse_mean": statistics.fmean(seam_norms) if seam_norms else None,
            "aggregate_rmse_max": max(seam_norms) if seam_norms else None,
            "delay_aligned_aggregate_rmse_mean": (
                statistics.fmean(aligned_seam_norms) if aligned_seam_norms else None
            ),
            "delay_aligned_aggregate_rmse_max": (max(aligned_seam_norms) if aligned_seam_norms else None),
            "finding": (
                "aggregate_rmse is the naive action-0 replacement metric; "
                "delay_aligned metrics skip chunk actions whose timestamps elapsed during inference, "
                "matching ActionQueue RTC replacement semantics"
            ),
        },
        "chunks": chunks,
    }


def _console_summary(summary: dict[str, Any]) -> None:
    latency = summary["latency"]
    ranges = summary["range_safety"]
    seams = summary["seams"]
    first = summary["chunks"][0]["anchor_aligned_rmse"]
    print("OPEN-LOOP DRY RUN ONLY — no task-success claim and no motor commands")
    print(f"policy={summary['protocol']['policy']} episode={summary['protocol']['episode']}")
    print(
        f"latency mean/p95/max={latency['mean_s']:.3f}/{latency['p95_s']:.3f}/"
        f"{latency['max_s']:.3f}s; horizon margin={latency['minimum_margin_s']:.3f}s; "
        f"replan={latency['effective_replan_hz']:.2f}Hz; starved_ticks={latency['starved_control_ticks']}"
    )
    print(
        f"first chunk RMSE h1/first5/full={first['h1']['aggregate']:.3f}/"
        f"{first['first5']['aggregate']:.3f}/{first['full_horizon']['aggregate']:.3f} stored units"
    )
    print(
        f"training-range excursions={ranges['excursion_count']}; "
        f"naive seam RMSE mean/max={seams['aggregate_rmse_mean']}/{seams['aggregate_rmse_max']}; "
        f"delay-aligned mean/max={seams['delay_aligned_aggregate_rmse_mean']}/"
        f"{seams['delay_aligned_aggregate_rmse_max']}"
    )
    print(
        f"normalization={summary['normalization']['status']}; "
        f"fixed-noise max spread={summary['determinism_fixed_noise']['max_abs_spread']:.6g}"
    )


def _make_backend(
    args: argparse.Namespace, train_stats: dict[str, Tensor | int], device: torch.device
) -> RolloutBackend:
    if args.policy == "smolvla":
        return SmolVLARolloutBackend(args.smolvla_checkpoint, train_stats, device)
    if args.policy == "cosmos-smolexpert":
        return CosmosSmolExpertBackend(
            args.cosmos_run,
            device,
            args.cosmos_checkpoint,
            args.cosmos_tokenizer,
            args.cosmos_prompt,
            state_t=args.cosmos_state_t,
            context_transform=args.cosmos_context_transform,
            lora_weights=args.cosmos_lora,
            merge_lora_weights=args.cosmos_merge_lora,
            compile=args.compile,
        )
    if args.policy == "ltx-smolexpert":
        return LTXSmolExpertBackend(
            args.ltx_run,
            device,
            args.ltx_transformer,
            args.ltx_vae,
            args.ltx_prompt,
        )
    raise AssertionError(args.policy)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.episode not in VAL_EPISODES:
        raise ValueError("--episode must be a held-out validation episode 32-39")
    if args.max_control_frames is not None and args.max_control_frames <= 0:
        raise ValueError("--max-control-frames must be positive")
    if args.max_predictions is not None and args.max_predictions <= 0:
        raise ValueError("--max-predictions must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train_stats = _train_stats(args.dataset_root)
    backend = _make_backend(args, train_stats, device)
    stream = EpisodeStream(args.dataset_root, args.episode)
    reports: list[ChunkReport] = []
    seams: list[dict[str, Any]] = []
    starved_ticks = 0
    pending: Future[TimedPrediction] | None = None
    active: ChunkReport | None = None
    previous_emitted: Tensor | None = None
    first_observation: ReplayObservation | None = None
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dry-run-inference")
    execution_started = 0.0

    try:
        for local_index in range(len(stream)):
            observation = stream.read(local_index)
            if observation is None:
                continue
            first_observation = observation
            initial = _timed_predict(backend, observation, args.seed)
            active = ChunkReport(initial, installed_at_local_frame=local_index)
            reports.append(active)
            break
        if active is None or first_observation is None:
            raise RuntimeError("episode has fewer than five frames")

        repeated = _timed_predict(backend, first_observation, args.seed)
        spread = (repeated.actions - active.prediction.actions).abs()
        determinism = {
            "runs": 2,
            "seed": args.seed,
            "max_abs_spread": float(spread.max()),
            "mean_abs_spread": float(spread.mean()),
            "deterministic_with_fixed_noise": bool(float(spread.max()) <= args.determinism_atol),
            "atol": args.determinism_atol,
        }

        execution_started = time.perf_counter()
        next_deadline = execution_started
        last_local = first_observation.local_frame_index
        for frames_executed, local_index in enumerate(
            range(first_observation.local_frame_index, len(stream)), start=1
        ):
            if local_index == first_observation.local_frame_index:
                observation = first_observation
            else:
                observation = stream.read(local_index)
                if observation is None:
                    raise AssertionError("history unexpectedly became incomplete")
            last_local = local_index

            if pending is not None and pending.done():
                prediction = pending.result()
                pending = None
                new_active = ChunkReport(prediction, installed_at_local_frame=local_index)
                reports.append(new_active)
                if previous_emitted is not None:
                    seam_delta = prediction.actions[0] - previous_emitted
                    recorded_delta = (
                        stream.actions[local_index]
                        - stream.actions[max(local_index - 1, first_observation.local_frame_index)]
                    )
                    elapsed_action_steps = local_index - prediction.observation.local_frame_index
                    delay_aligned_delta = None
                    if elapsed_action_steps < HORIZON:
                        delay_aligned_delta = prediction.actions[elapsed_action_steps] - previous_emitted
                    seams.append(
                        {
                            "from_chunk": len(reports) - 2,
                            "to_chunk": len(reports) - 1,
                            "local_frame_index": local_index,
                            "delta_per_joint": seam_delta.tolist(),
                            "aggregate_rmse": float(seam_delta.square().mean().sqrt()),
                            "max_abs": float(seam_delta.abs().max()),
                            "delay_aligned_action_index": elapsed_action_steps,
                            "delay_aligned_delta_per_joint": (
                                delay_aligned_delta.tolist() if delay_aligned_delta is not None else None
                            ),
                            "delay_aligned_aggregate_rmse": (
                                float(delay_aligned_delta.square().mean().sqrt())
                                if delay_aligned_delta is not None
                                else None
                            ),
                            "delay_aligned_max_abs": (
                                float(delay_aligned_delta.abs().max())
                                if delay_aligned_delta is not None
                                else None
                            ),
                            "recorded_delta_per_joint": recorded_delta.tolist(),
                            "recorded_aggregate_rmse": float(recorded_delta.square().mean().sqrt()),
                        }
                    )
                active = new_active

            assert active is not None
            offset = len(active.executed_actions)
            if offset < HORIZON:
                emitted = active.prediction.actions[offset]
                active.executed_actions.append(emitted.tolist())
                active.recorded_executed_actions.append(stream.actions[local_index].tolist())
                active.execution_local_frames.append(local_index)
                previous_emitted = emitted
            else:
                active.starved_after_actions = True
                starved_ticks += 1

            prediction_budget_available = args.max_predictions is None or len(reports) < args.max_predictions
            if (
                pending is None
                and prediction_budget_available
                and local_index > active.prediction.observation.local_frame_index
            ):
                seed = args.seed + len(reports)
                pending = executor.submit(_timed_predict, backend, observation, seed)

            if args.max_control_frames is not None and frames_executed >= args.max_control_frames:
                break
            next_deadline += 1.0 / FPS
            precise_sleep(max(next_deadline - time.perf_counter(), 0.0))

        elapsed_s = time.perf_counter() - execution_started
        if pending is not None:
            prediction = pending.result()
            reports.append(ChunkReport(prediction, installed_at_local_frame=None))
        if last_local < first_observation.local_frame_index:
            raise AssertionError("control loop did not execute")
        summary = _summarize(
            args,
            backend,
            train_stats,
            reports,
            seams,
            determinism,
            elapsed_s,
            starved_ticks,
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=False)
        backend.close()

    output = args.output or (
        Path("/home/anton/.cache/video-vam/dry-runs") / f"{args.policy}-episode-{args.episode}.json"
    )
    output = output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    _console_summary(summary)
    print(f"JSON_SUMMARY={output}")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--episode", type=int, default=32)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--determinism-atol", type=float, default=1e-5)
    parser.add_argument("--max-control-frames", type=int)
    parser.add_argument("--max-predictions", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smolvla-checkpoint", type=Path, default=DEFAULT_SMOLVLA)
    parser.add_argument("--cosmos-run", type=Path, default=DEFAULT_COSMOS_RUN)
    parser.add_argument("--cosmos-checkpoint", type=Path, default=DEFAULT_COSMOS_CHECKPOINT)
    parser.add_argument("--cosmos-tokenizer", type=Path, default=DEFAULT_COSMOS_TOKENIZER)
    parser.add_argument("--cosmos-prompt", type=Path, default=DEFAULT_COSMOS_PROMPT)
    parser.add_argument("--cosmos-state-t", type=int, choices=(2, 16), default=2)
    parser.add_argument("--cosmos-context-transform", choices=("none", "pool2"), default="none")
    parser.add_argument("--cosmos-lora", type=Path)
    parser.add_argument("--cosmos-merge-lora", type=Path)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile Cosmos DiT with max-autotune and use CUDA graph for SmolExpert decoder.",
    )
    parser.add_argument("--ltx-run", type=Path, default=DEFAULT_LTX_RUN)
    parser.add_argument("--ltx-transformer", type=Path, default=DEFAULT_LTX_TRANSFORMER)
    parser.add_argument("--ltx-vae", type=Path, default=DEFAULT_LTX_VAE)
    parser.add_argument("--ltx-prompt", type=Path, default=DEFAULT_LTX_PROMPT)
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if not math.isfinite(args.determinism_atol) or args.determinism_atol < 0:
        parser.error("--determinism-atol must be finite and non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
