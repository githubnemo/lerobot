#!/usr/bin/env python3
"""Generate and round-trip verify the one-time official Cosmos T5 embedding."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

from lerobot.policies.vam.cosmos_prompt_embedding import (
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_PATH,
    PromptEmbeddingArtifact,
    build_provenance,
    generate_prompt_embedding,
    load_prompt_embedding,
    save_prompt_embedding,
    verify_t5_model,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse generator options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu"),
        default="auto",
        help="auto uses Accelerate device_map auto; cpu loads the complete bf16 model on CPU",
    )
    parser.add_argument(
        "--gpu-max-memory",
        default="20GiB",
        help="Accelerate max_memory limit for cuda:0 in auto mode (default: 20GiB)",
    )
    parser.add_argument(
        "--cpu-max-memory",
        default="40GiB",
        help="Accelerate max_memory limit for CPU (default: 40GiB)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacing an existing safetensors file or JSON sidecar",
    )
    parser.add_argument(
        "--skip-model-hash",
        action="store_true",
        help="TEST ONLY: skip the streamed T5 SHA256 check; presence and size remain required",
    )
    return parser.parse_args(argv)


def _max_memory(args: argparse.Namespace) -> dict[int | str, str]:
    """Build conservative Accelerate memory limits for the target machine."""
    if args.device == "cpu":
        return {"cpu": args.cpu_max_memory}
    if torch.cuda.is_available():
        return {0: args.gpu_max_memory, "cpu": args.cpu_max_memory}
    return {"cpu": args.cpu_max_memory}


def load_official_t5(model_path: Path, args: argparse.Namespace) -> tuple[Any, Any]:
    """Load the official local tokenizer and bf16 T5 model without network access."""
    try:
        from transformers import T5EncoderModel, T5TokenizerFast
    except ImportError as exc:
        raise RuntimeError("transformers is required to generate the T5 embedding") from exc

    tokenizer = T5TokenizerFast.from_pretrained(str(model_path), local_files_only=True)
    load_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "weights_only": True,
    }
    if args.device == "auto":
        load_kwargs.update(device_map="auto", max_memory=_max_memory(args))
    else:
        load_kwargs.update(device_map={"": "cpu"}, max_memory=_max_memory(args))
    model = T5EncoderModel.from_pretrained(str(model_path), **load_kwargs)
    model.eval()
    return tokenizer, model


def refuse_existing_output(output_path: Path, *, overwrite: bool) -> None:
    """Refuse an existing output before spending time loading the 45 GB model."""
    sidecar = output_path.with_suffix(".json")
    if not overwrite and (output_path.exists() or sidecar.exists()):
        raise FileExistsError(f"Refusing to overwrite existing output; pass --overwrite: {output_path}")


def generate(args: argparse.Namespace) -> PromptEmbeddingArtifact:
    """Verify, generate, save, reload, and compare one embedding artifact."""
    model_path = args.model_path.expanduser().resolve()
    output_path = args.output.expanduser()
    refuse_existing_output(output_path, overwrite=args.overwrite)
    verification = verify_t5_model(model_path, skip_hash=args.skip_model_hash)
    tokenizer, model = load_official_t5(model_path, args)
    generation = generate_prompt_embedding(tokenizer, model)

    import transformers

    provenance = build_provenance(
        generation,
        model_path=model_path,
        transformers_version=transformers.__version__,
        torch_version=torch.__version__,
    )
    artifact = PromptEmbeddingArtifact(generation.embedding, generation.attention_mask, provenance)
    save_prompt_embedding(artifact, output_path, overwrite=args.overwrite)
    reloaded = load_prompt_embedding(output_path)
    if not torch.equal(reloaded.embedding, artifact.embedding):
        raise RuntimeError("Round-trip embedding comparison failed")
    if not torch.equal(reloaded.attention_mask, artifact.attention_mask):
        raise RuntimeError("Round-trip attention-mask comparison failed")

    print("Prompt embedding generation succeeded")
    print(f"  output:       {output_path}")
    print(f"  provenance:   {output_path.with_suffix('.json')}")
    print(f"  model:        {verification.model_file}")
    print(f"  model_sha256: {verification.sha256} (verified={verification.hash_verified})")
    print(f"  shape:        {tuple(reloaded.embedding.shape)}")
    print(f"  dtype:        {reloaded.embedding.dtype}")
    print(f"  prompt:       {reloaded.provenance.prompt!r}")
    return reloaded


def main(argv: list[str] | None = None) -> int:
    """Run the safe one-time generator."""
    args = parse_args(argv)
    try:
        generate(args)
    except Exception as exc:  # noqa: BLE001 - CLI should report a concise failure.
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
