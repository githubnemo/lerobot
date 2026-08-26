#!/usr/bin/env python3
"""Measure block-40 multi-tap extraction cost against the block-34 default."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT, validate_metadata
from lerobot.policies.vam.ltx_extractor import LTXExtractor, LTXExtractorConfig, derive_window_seed
from lerobot.policies.vam.ltx_layer_mix import LTX_LAYER_PROBE_DEPTHS
from lerobot.policies.vam.ltx_prompt_embedding import load_ltx_prompt_artifact
from scripts.video_vam.build_ltx_feature_cache import DEFAULT_TRANSFORMER, DEFAULT_VAE
from scripts.video_vam.smoke_test_cosmos_extractor import _relative_index, prepare_sample

DEFAULT_DATASET = Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset")
DEFAULT_PROMPT = Path(
    "/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-ltx25-gemma4.safetensors"
)
DEFAULT_OUTPUT = Path("/home/anton/.cache/video-vam/runs/ltx25-layer-probe-extraction-cost-20260826.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--transformer", type=Path, default=DEFAULT_TRANSFORMER)
    parser.add_argument("--video-vae", type=Path, default=DEFAULT_VAE)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--episode", type=int, default=32)
    parser.add_argument("--frame-offset", type=int, default=4)
    return parser.parse_args()


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def _measure(
    extractor: LTXExtractor,
    images: torch.Tensor,
    prompt: torch.Tensor,
    seed: int,
    *,
    multi: bool,
) -> tuple[dict[str, float], torch.Tensor, dict[str, Any]]:
    timings: dict[str, float] = {}
    torch.cuda.synchronize()
    started = time.perf_counter()
    if multi:
        multi_extraction = extractor.extract_layers(
            images,
            prompt,
            layer_indices=LTX_LAYER_PROBE_DEPTHS,
            noise_seed=seed,
            timings=timings,
        )
        output = multi_extraction.tokens_by_layer[34]
    else:
        extraction = extractor.extract(images, prompt, noise_seed=seed, timings=timings)
        output = extraction.tokens
    torch.cuda.synchronize()
    timings["total_extraction"] = time.perf_counter() - started
    return timings, output.detach().cpu(), dict(getattr(extractor.backbone, "last_profile", {}))


def main() -> int:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    dataset = LeRobotDataset(
        CUBE_OUT_OF_BOX_CONTRACT.repo_id,
        root=args.dataset_root.expanduser(),
        episodes=[args.episode],
        delta_timestamps=CUBE_OUT_OF_BOX_CONTRACT.delta_timestamps(),
        revision=CUBE_OUT_OF_BOX_CONTRACT.revision,
        return_uint8=True,
        download_videos=False,
    )
    validate_metadata(dataset.meta, CUBE_OUT_OF_BOX_CONTRACT).raise_if_invalid()
    rows = dataset.meta.episodes
    starts = {
        int(episode): int(start)
        for episode, start in zip(rows["episode_index"], rows["dataset_from_index"], strict=True)
    }
    frame = starts[args.episode] + args.frame_offset
    sample = dataset[_relative_index(dataset, frame)]
    prepared = prepare_sample(sample, frame_index=frame, config=CUBE_OUT_OF_BOX_CONTRACT)
    prompt = load_ltx_prompt_artifact(args.prompt.expanduser().resolve())
    seed = derive_window_seed(dataset.revision, args.episode, frame, 0)
    extractor = LTXExtractor(
        LTXExtractorConfig(
            checkpoint_path=args.transformer.expanduser().resolve(),
            video_vae_path=args.video_vae.expanduser().resolve(),
            device="cuda",
            dtype="bfloat16",
            hidden_layer=34,
            high_noise_sigma=1.0,
            offload_mode="cpu",
            quantization="fp8-cast",
            persistent_transformer=True,
            explicit_prefix_execution=True,
        )
    )
    extractor.open_transformer()
    single_samples: list[dict[str, float]] = []
    multi_samples: list[dict[str, float]] = []
    profiles: dict[str, Any] = {}
    try:
        _measure(extractor, prepared.rgb_history, prompt.embedding, seed, multi=False)
        _measure(extractor, prepared.rgb_history, prompt.embedding, seed, multi=True)
        single_output: torch.Tensor | None = None
        multi_output: torch.Tensor | None = None
        for _ in range(args.iterations):
            single, single_output, profiles["block34"] = _measure(
                extractor, prepared.rgb_history, prompt.embedding, seed, multi=False
            )
            multi, multi_output, profiles["multitap_to_block40"] = _measure(
                extractor, prepared.rgb_history, prompt.embedding, seed, multi=True
            )
            single_samples.append(single)
            multi_samples.append(multi)
    finally:
        extractor.close()
    if single_output is None or multi_output is None:
        raise RuntimeError("benchmark produced no outputs")
    reference = single_output.float().reshape(-1)
    candidate = multi_output.float().reshape(-1)
    cosine = float(torch.nn.functional.cosine_similarity(reference, candidate, dim=0))
    max_abs = float((reference - candidate).abs().max())
    keys = ("total_extraction", "transformer_compute_to_hidden")
    single_summary = {
        key: _summary([sample[key] for sample in single_samples if key in sample]) for key in keys
    }
    multi_summary = {
        "total_extraction": _summary([sample["total_extraction"] for sample in multi_samples]),
        "transformer_compute_to_deepest_hidden": _summary(
            [sample["transformer_compute_to_deepest_hidden"] for sample in multi_samples]
        ),
    }
    total_ratio = (
        multi_summary["total_extraction"]["median_seconds"]
        / single_summary["total_extraction"]["median_seconds"]
    )
    transformer_ratio = (
        multi_summary["transformer_compute_to_deepest_hidden"]["median_seconds"]
        / single_summary["transformer_compute_to_hidden"]["median_seconds"]
    )
    result = {
        "contract": {
            "single_layer": 34,
            "tapped_layers": list(LTX_LAYER_PROBE_DEPTHS),
            "deepest_layer": 40,
            "same_vae_prompt_noise_and_persistent_transformer": True,
            "iterations": args.iterations,
            "episode": args.episode,
            "frame": frame,
        },
        "single": single_summary,
        "multi": multi_summary,
        "extra_cost": {
            "total_ratio": total_ratio,
            "total_percent": (total_ratio - 1.0) * 100,
            "transformer_ratio": transformer_ratio,
            "transformer_percent": (transformer_ratio - 1.0) * 100,
        },
        "layer34_equivalence": {"cosine": cosine, "max_abs": max_abs},
        "profiles": profiles,
    }
    if cosine < 0.99999:
        raise RuntimeError(f"layer-34 equivalence failed: cosine={cosine}")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("LTX_LAYER_PROBE_COST=" + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
