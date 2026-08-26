#!/usr/bin/env python3
"""Precompute the real LTX-2.5 Gemma-4 prompt context in a standalone process."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch

from lerobot.policies.vam.ltx_extractor import _add_official_source_paths
from lerobot.policies.vam.ltx_prompt_embedding import (
    DEFAULT_PROMPT,
    LTXPromptArtifact,
    build_provenance,
    load_ltx_prompt_artifact,
    save_ltx_prompt_artifact,
)

DEFAULT_TRANSFORMER = Path(
    "/home/anton/.cache/video-vam/ltx-2.5-models/diffusion_models/"
    "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
)
DEFAULT_TEXT_ENCODER = Path(
    "/home/anton/.cache/video-vam/ltx-2.5-models/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
)
DEFAULT_OUTPUT = Path(
    "/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-ltx25-gemma4.safetensors"
)


def _dispose(model: object | None) -> None:
    if model is None:
        return
    teardown = getattr(model, "teardown", None)
    if teardown is not None:
        teardown()
    dispose = getattr(model, "dispose", None)
    if dispose is not None:
        dispose()


def encode_official_prompt(
    prompt: str,
    *,
    transformer_path: Path,
    text_encoder_path: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Run the official streaming Gemma encoder and connector without pipeline media imports."""
    source_root = _add_official_source_paths()
    if source_root is None:
        raise RuntimeError("official LTX source closure is unavailable")
    from ltx_core.block_streaming import StreamingModelBuilder
    from ltx_core.loader.registry import ModelRegistry
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.text_encoders.gemma import (
        EMBEDDINGS_PROCESSOR_KEY_OPS,
        EmbeddingsProcessorConfigurator,
        GemmaTextEncoderConfigurator,
        get_gemma_ops,
        resolve_gemma_weight_paths,
    )

    dtype = torch.bfloat16
    registry = ModelRegistry(cache_models=True, cache_weights=False)
    sd_ops, module_ops = get_gemma_ops(str(text_encoder_path))
    text_builder = StreamingModelBuilder(
        model_path=resolve_gemma_weight_paths(str(text_encoder_path)),
        model_class_configurator=GemmaTextEncoderConfigurator.with_gemma_model_path(str(text_encoder_path)),
        model_sd_ops=sd_ops,
        module_ops=module_ops,
        registry=registry,
        blocks_attr="model.model.language_model.layers",
        blocks_prefix="model.model.language_model.layers",
    )
    processor_builder = SingleGPUModelBuilder(
        model_path=(str(transformer_path), str(text_encoder_path)),
        model_class_configurator=EmbeddingsProcessorConfigurator.with_gemma_model_path(
            str(text_encoder_path)
        ),
        model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
        registry=registry,
    )

    timings: dict[str, float] = {}
    text_encoder = None
    processor = None
    try:
        started = time.perf_counter()
        text_encoder = text_builder.build(device=device, dtype=dtype).eval()
        timings["text_encoder_load_seconds"] = time.perf_counter() - started
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            raw_outputs = text_encoder.encode([prompt])
        torch.cuda.synchronize(device)
        timings["gemma_forward_seconds"] = time.perf_counter() - started
    finally:
        _dispose(text_encoder)
        del text_encoder
        gc.collect()
        torch.cuda.empty_cache()

    try:
        started = time.perf_counter()
        processor = processor_builder.build(device=device, dtype=dtype).eval()
        timings["connector_load_seconds"] = time.perf_counter() - started
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            output = processor.process_hidden_states(*raw_outputs[0])
        torch.cuda.synchronize(device)
        timings["connector_forward_seconds"] = time.perf_counter() - started
        embedding = output.video_encoding.detach().to(device="cpu", dtype=dtype).contiguous()
        attention_mask = output.attention_mask.detach().to(device="cpu", dtype=torch.bool).contiguous()
    finally:
        _dispose(processor)
        del processor
        gc.collect()
        torch.cuda.empty_cache()
    return embedding, attention_mask, timings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer", type=Path, default=DEFAULT_TRANSFORMER)
    parser.add_argument("--text-encoder", type=Path, default=DEFAULT_TEXT_ENCODER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.prompt != DEFAULT_PROMPT:
        raise ValueError(f"this dataset gate requires exact task text {DEFAULT_PROMPT!r}")
    if args.output.exists() and not args.overwrite:
        artifact = load_ltx_prompt_artifact(args.output)
        print(
            f"LTX_PROMPT_ALREADY_COMPLETE path={args.output} "
            f"shape={tuple(artifact.embedding.shape)} valid_tokens={artifact.provenance.valid_tokens}",
            flush=True,
        )
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("real Gemma-4 prompt precompute requires CUDA")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    embedding, attention_mask, timings = encode_official_prompt(
        args.prompt,
        transformer_path=args.transformer.expanduser().resolve(),
        text_encoder_path=args.text_encoder.expanduser().resolve(),
        device=device,
    )
    encoding_seconds = time.perf_counter() - started
    provenance = build_provenance(
        embedding,
        attention_mask,
        text_encoder_path=args.text_encoder,
        transformer_path=args.transformer,
    )
    artifact = LTXPromptArtifact(embedding, attention_mask, provenance)
    save_ltx_prompt_artifact(artifact, args.output, overwrite=args.overwrite)
    reloaded = load_ltx_prompt_artifact(args.output)
    print("LTX Gemma-4 prompt precompute succeeded", flush=True)
    print(f"  prompt: {args.prompt!r}", flush=True)
    print(f"  shape/dtype: {tuple(reloaded.embedding.shape)} / {reloaded.embedding.dtype}", flush=True)
    print(f"  valid_tokens: {reloaded.provenance.valid_tokens}", flush=True)
    print(f"  embedding_sha256: {reloaded.provenance.embedding_sha256}", flush=True)
    print(f"  source_commit: {reloaded.provenance.source_commit}", flush=True)
    print(f"  model_revision: {reloaded.provenance.model_revision}", flush=True)
    print(f"  encoding_seconds: {encoding_seconds:.3f}", flush=True)
    print(f"  stage_timings: {timings}", flush=True)
    print(f"  peak_vram_bytes: {torch.cuda.max_memory_allocated(device)}", flush=True)
    print(f"  artifact: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
