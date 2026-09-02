#!/usr/bin/env python3
"""Benchmark end-to-end inference latency for Distilled Cosmos T=2 + SmolExpert."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from lerobot.policies.vam.cosmos_lora import inject_lora, load_lora_state_dict, merge_lora_file_into_base
from lerobot.policies.vam.cosmos_predict2_extractor import (
    CosmosPredict2Extractor,
    CosmosPredict2ExtractorConfig,
)
from lerobot.policies.vam.smol_expert import SmolExpertActionDecoder

DEFAULT_BASE_CHECKPOINT = Path(
    "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt"
)
DEFAULT_TOKENIZER = Path(
    "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth"
)
DEFAULT_PROMPT = Path("/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-t5-11b.safetensors")
DEFAULT_BASE_LORA = Path(
    "/home/anton/.cache/video-vam/runs/cosmos2b-video-lora-20260828/paused-step6000/best_lora.safetensors"
)
DEFAULT_DISTILLED_LORA = Path(
    "/home/anton/.cache/video-vam/runs/cosmos-t2-direct-distillation-aligned/best_lora.safetensors"
)
DEFAULT_EXPERT_DIR = Path(
    "/home/anton/.cache/video-vam/runs/cosmos-t2-direct-distilled-smolexpert-20260902/smolexpert"
)


def benchmark_stream(name: str, fn, warmups: int = 10, iterations: int = 30):
    for _ in range(warmups):
        fn()
        torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    median = np.median(times)
    p90 = np.percentile(times, 90)
    print(f"  {name:30s} | Median: {median:6.2f} ms | P90: {p90:6.2f} ms")
    return median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 70)
    print("COSMOS T=2 DISTILLED END-TO-END LATENCY BENCHMARK (RTX 4090)")
    print("=" * 70)

    # 1. Load Cosmos T=2 Extractor
    print("\n[1/3] Loading Distilled Cosmos T=2 Extractor...")
    config = CosmosPredict2ExtractorConfig(
        checkpoint_path=DEFAULT_BASE_CHECKPOINT,
        tokenizer_path=DEFAULT_TOKENIZER,
        device="cuda",
        dtype="bfloat16",
        hidden_layer=20,
        high_noise_sigma=80.0,
        seed=0,
        state_t=2,
        vae_input_mode="observed_prefix",
        input_frames=5,
    )
    extractor = CosmosPredict2Extractor(config)
    merge_lora_file_into_base(extractor.backbone, DEFAULT_BASE_LORA)
    inject_lora(extractor.backbone, rank=16, alpha=16.0, block_indices=range(20))
    load_lora_state_dict(extractor.backbone, DEFAULT_DISTILLED_LORA)
    extractor.backbone.eval()

    prompt = load_file(str(DEFAULT_PROMPT))["prompt_embedding"].to(device=device, dtype=torch.bfloat16)
    rgb = torch.randint(0, 256, (1, 3, 5, 480, 640), device=device, dtype=torch.uint8)

    # 2. Load SmolExpert
    print("[2/3] Loading SmolExpert Action Policy...")
    expert_ckpt = DEFAULT_EXPERT_DIR / "best.safetensors"
    norm_meta = json.loads((DEFAULT_EXPERT_DIR / "normalizer.json").read_text())
    norm_tensors = load_file(str(DEFAULT_EXPERT_DIR / "normalizer.safetensors"))
    from lerobot.policies.vam.smol_expert import SmolVLANormalizer

    normalizer = SmolVLANormalizer.from_training_tensors(norm_tensors, norm_meta)

    decoder = SmolExpertActionDecoder.from_training_checkpoint(
        expert_ckpt,
        normalizer=normalizer,
        expert_checkpoint="lerobot/smolvla_base",
        device="cuda",
        num_steps=args.steps,
        input_channels=2048,
    ).eval()

    state = torch.zeros(1, 6, device=device, dtype=torch.float32)

    # 3. Benchmark Eager Path
    print("\n[3/3] Running Latency Profiles:")
    print("-" * 70)
    print("ARM 1: Eager PyTorch (Uncompiled)")
    print("-" * 70)

    with torch.no_grad():
        benchmark_stream("VAE Encode (5 frames)", lambda: extractor.encode_observed_pixels(rgb))
        # latents = extractor.encode_observed_pixels(rgb)
        benchmark_stream("Cosmos 2B DiT (Layer 20)", lambda: extractor.forward_features(rgb, prompt))
        extraction = extractor.forward_features(rgb, prompt)
        context = extraction.tokens  # [1, 2400, 2048]
        benchmark_stream(
            f"SmolExpert ({args.steps} steps eager)",
            lambda: decoder.sample_actions(state, context, use_cuda_graph=False),
        )

        def full_eager():
            ext = extractor.forward_features(rgb, prompt)
            return decoder.sample_actions(state, ext.tokens, use_cuda_graph=False)

        e2e_eager = benchmark_stream("Total End-to-End (Eager)", full_eager)
        print(f"  --> Effective Control Rate: {1000.0 / e2e_eager:.1f} Hz")

    # 4. Benchmark CUDA Graph & Compiled Path
    print("\n" + "-" * 70)
    print(f"ARM 2: Optimized (CUDA Graph SmolExpert, {args.steps} steps)")
    print("-" * 70)
    with torch.no_grad():
        benchmark_stream(
            f"SmolExpert ({args.steps} steps CUDA Graph)",
            lambda: decoder.sample_actions(state, context, use_cuda_graph=True),
        )

        def full_graph():
            ext = extractor.forward_features(rgb, prompt)
            return decoder.sample_actions(state, ext.tokens, use_cuda_graph=True)

        e2e_graph = benchmark_stream("Total End-to-End (CUDA Graph)", full_graph)
        print(f"  --> Effective Control Rate: {1000.0 / e2e_graph:.1f} Hz")

    # 5. Benchmark 5-Step Fast Control
    print("\n" + "-" * 70)
    print("ARM 3: Fast Control (5 Euler steps + CUDA Graph)")
    print("-" * 70)
    with torch.no_grad():
        benchmark_stream(
            "SmolExpert (5 steps CUDA Graph)",
            lambda: decoder.sample_actions(state, context, num_steps=5, use_cuda_graph=True),
        )

        def full_5step():
            ext = extractor.forward_features(rgb, prompt)
            return decoder.sample_actions(state, ext.tokens, num_steps=5, use_cuda_graph=True)

        e2e_5step = benchmark_stream("Total End-to-End (5-step)", full_5step)
        print(f"  --> Effective Control Rate: {1000.0 / e2e_5step:.1f} Hz")
    print("=" * 70)


if __name__ == "__main__":
    import json

    main()
