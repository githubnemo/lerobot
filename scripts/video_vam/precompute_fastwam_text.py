#!/usr/bin/env python3
"""Precompute canonical UMT5 context rows for FastWAM before policy construction.

The script intentionally loads only the tokenizer and UMT5 encoder. It never builds
FastWAM, the Wan video expert, or the VAE, so the process can exit and release the
text encoder before a 24 GB training process starts.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT
from lerobot.policies.fastwam.text_context import (
    TextContextProvenance,
    encode_wan_text_context,
    format_task_prompts,
    load_text_context_artifact,
    save_text_context_artifact,
)
from lerobot.policies.fastwam.wan import build_wan_tokenizer, load_pretrained_wan_text_encoder

DEFAULT_DATASET_ROOT = Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset")
DEFAULT_OUTPUT = Path("/home/anton/.cache/video-vam/fastwam-text/cube-out-of-box.safetensors")


def _parse_episodes(value: str) -> list[int]:
    episodes: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start, stop = (int(piece) for piece in part.split(":", maxsplit=1))
            episodes.extend(range(start, stop))
        else:
            episodes.append(int(part))
    if not episodes or min(episodes) < 0:
        raise ValueError(f"Invalid episode selection: {value!r}")
    return sorted(set(episodes))


def _dtype(name: str) -> torch.dtype:
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        raise ValueError("--dtype must be one of float32, float16, or bfloat16")
    return value


def _progress_paths(output: Path) -> tuple[Path, Path]:
    return output.with_name(output.name + ".partial"), output.with_name(output.name + ".progress.json")


def _atomic_save_partial(path: Path, context: torch.Tensor, context_mask: torch.Tensor) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        save_file({"context": context, "context_mask": context_mask}, str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _dataset_tasks(args: argparse.Namespace) -> list[str]:
    metadata = LeRobotDatasetMetadata(
        args.repo_id,
        root=args.dataset_root.expanduser(),
        revision=args.revision,
    )
    episodes = _parse_episodes(args.episodes)
    available = {int(index) for index in metadata.episodes["episode_index"]}
    missing = sorted(set(episodes).difference(available))
    if missing:
        raise ValueError(f"Dataset metadata has no requested episode(s): {missing}")
    # The tasks table is the authoritative mapping used by LeRobotDataset.__getitem__.
    # Include every dataset task so the artifact covers train and held-out evaluation.
    if metadata.tasks is None:
        raise ValueError("Dataset metadata contains no tasks table.")
    return [str(task) for task in metadata.tasks.index.tolist()]


def _load_or_initialize_partial(
    output: Path,
    prompts: list[str],
    provenance: TextContextProvenance,
) -> tuple[torch.Tensor, torch.Tensor, set[str]] | None:
    partial, progress = _progress_paths(output)
    if not partial.exists() and not progress.exists():
        return None
    if not partial.exists() or not progress.exists():
        raise RuntimeError(f"Incomplete FastWAM artifact has only one resume file: {partial}, {progress}")
    state: dict[str, Any] = json.loads(progress.read_text())
    if state.get("prompts") != prompts or state.get("provenance") != provenance.as_dict():
        raise ValueError("Existing FastWAM text precompute progress does not match this invocation.")
    tensors = load_file(str(partial), device="cpu")
    context = tensors.get("context")
    context_mask = tensors.get("context_mask")
    expected_shape = (len(prompts), provenance.context_len, provenance.context_dim)
    if context is None or context_mask is None or tuple(context.shape) != expected_shape:
        raise ValueError(f"Existing partial context shape is incompatible; expected {expected_shape}.")
    if tuple(context_mask.shape) != expected_shape[:2]:
        raise ValueError("Existing partial context mask shape is incompatible.")
    completed = {str(prompt) for prompt in state.get("completed_prompts", [])}
    if not completed.issubset(set(prompts)):
        raise ValueError("Existing partial completion list contains an unexpected prompt.")
    return context, context_mask.to(dtype=torch.bool), completed


def _write_progress(
    output: Path, prompts: list[str], provenance: TextContextProvenance, completed: set[str]
) -> None:
    _, progress = _progress_paths(output)
    temporary = progress.with_name(f".{progress.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "format": "fastwam_text_context_progress",
                "prompts": prompts,
                "provenance": provenance.as_dict(),
                "completed_prompts": [prompt for prompt in prompts if prompt in completed],
            },
            indent=2,
        )
        + "\n"
    )
    os.replace(temporary, progress)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default=CUBE_OUT_OF_BOX_CONTRACT.repo_id)
    parser.add_argument("--revision", default=CUBE_OUT_OF_BOX_CONTRACT.revision)
    parser.add_argument("--episodes", default="0:32", help="Episode ids, e.g. 0:32 or 0,1,2")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-id", default="Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument("--tokenizer-model-id", default="google/umt5-xxl")
    parser.add_argument("--text-encoder-model-id", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument(
        "--prompt-template",
        default="A video recorded from a robot's point of view executing the following instruction: {task}",
    )
    parser.add_argument("--tokenizer-max-len", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    args = parser.parse_args(argv)
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")
    args.dataset_root = args.dataset_root.expanduser()
    args.output = args.output.expanduser()
    # Progress and partial artifacts are written before model work, so create the destination first.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prompts = format_task_prompts(_dataset_tasks(args), args.prompt_template)
    dtype = _dtype(args.dtype)
    provenance = TextContextProvenance(
        model_id=args.model_id,
        tokenizer_model_id=args.tokenizer_model_id,
        text_encoder_model_id=args.text_encoder_model_id,
        dtype=args.dtype,
        prompt_template=args.prompt_template,
        tokenizer_max_len=args.tokenizer_max_len,
        context_len=args.tokenizer_max_len,
        context_dim=4096,
    )

    if args.output.exists():
        artifact = load_text_context_artifact(args.output, expected_provenance=provenance)
        if set(artifact.prompts) != set(prompts):
            raise ValueError(
                "Existing FastWAM text context artifact prompt set does not match the dataset: "
                f"artifact={list(artifact.prompts)}, expected={prompts}"
            )
        print(f"TEXT_CONTEXT_ALREADY_COMPLETE path={args.output} prompts={len(prompts)}", flush=True)
        return 0

    partial_state = _load_or_initialize_partial(args.output, prompts, provenance)
    if partial_state is None:
        context = torch.zeros(
            (len(prompts), provenance.context_len, provenance.context_dim), dtype=dtype, device="cpu"
        )
        context_mask = torch.ones((len(prompts), provenance.context_len), dtype=torch.bool, device="cpu")
        completed: set[str] = set()
    else:
        context, context_mask, completed = partial_state
        if context.dtype != dtype:
            raise ValueError(f"Partial context dtype {context.dtype} does not match requested {dtype}.")

    tokenizer = build_wan_tokenizer(
        model_id=args.tokenizer_model_id, tokenizer_max_len=args.tokenizer_max_len
    )
    text_encoder = load_pretrained_wan_text_encoder(
        model_id=args.text_encoder_model_id, torch_dtype=dtype, device=args.device
    )
    try:
        _write_progress(args.output, prompts, provenance, completed)
        for index, prompt in enumerate(prompts):
            if prompt in completed:
                continue
            encoded, encoded_mask = encode_wan_text_context(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                prompts=[prompt],
                device=args.device,
            )
            if tuple(encoded.shape) != (1, provenance.context_len, provenance.context_dim):
                raise ValueError(
                    "UMT5 context shape does not match FastWAM config: "
                    f"got {tuple(encoded.shape)}, expected {(1, provenance.context_len, provenance.context_dim)}"
                )
            context[index].copy_(encoded[0].to(device="cpu", dtype=dtype))
            context_mask[index].copy_(encoded_mask[0].to(device="cpu", dtype=torch.bool))
            completed.add(prompt)
            if len(completed) % args.checkpoint_every == 0 or len(completed) == len(prompts):
                partial, _ = _progress_paths(args.output)
                _atomic_save_partial(partial, context, context_mask)
                _write_progress(args.output, prompts, provenance, completed)
                print(f"TEXT_CONTEXT_PROGRESS {len(completed)}/{len(prompts)} index={index}", flush=True)
        save_text_context_artifact(args.output, context, context_mask, prompts, provenance)
    finally:
        del text_encoder, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    partial, progress = _progress_paths(args.output)
    partial.unlink(missing_ok=True)
    progress.unlink(missing_ok=True)
    print(f"TEXT_CONTEXT_DONE path={args.output} prompts={len(prompts)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
