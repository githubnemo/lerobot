#!/usr/bin/env python3
"""Unified Protocol 1.1 Dual-Evaluation Across Video Action Models and SmolVLA Baselines.

Evaluates all preserved and trained models under strict Protocol 1.1
(deterministic SHA-256 seeded noise, masked action RMSE, train-only normalization):
  1. Eval-Set 1 (V1 Held-Out): hubnemo/cube_out_of_box_dataset, eps 32..39 (88 anchors)
  2. Eval-Set 2 (V2 Held-Out): Orellius/cube_out_of_box_v2, eps 90..99 (100 anchors)

Outputs:
  - Markdown table printed to stdout
  - JSON matrix saved to outputs/evaluation/protocol1_1_dual_eval_matrix.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from lerobot.policies.vam.action_rmse import EvaluationBatch, SmolVLABackend, evaluate_backend
from lerobot.policies.vam.smol_expert import (
    SmolExpertActionDecoder,
    SmolVLANormalizer,
)
from scripts.video_vam.train_smolexpert import UnifiedFeatureCacheDataset, evaluate_validation

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_EVAL = REPO_ROOT / "outputs" / "evaluation"
OUTPUTS_EVAL.mkdir(parents=True, exist_ok=True)

DEFAULT_V1_REPO = "hubnemo/cube_out_of_box_dataset"
DEFAULT_V1_REV = "243370c3c08bcbd860133c4a0d658ea7c1d2e77e"
DEFAULT_V2_REPO = "Orellius/cube_out_of_box_v2"


def evaluate_smolexpert_on_cache(
    model_path: Path,
    val_manifest: Path,
    eval2_manifest: Path | None,
    device: torch.device,
    batch_size: int = 8,
    num_steps: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    norm_path = model_path.parent / "normalizer.safetensors"
    if not norm_path.is_file():
        norm_path = model_path.parent / "normalizer.json"
    if not norm_path.is_file():
        raise FileNotFoundError(f"Normalizer not found alongside {model_path}")

    norm_sd = (
        load_file(str(norm_path)) if norm_path.suffix == ".safetensors" else json.loads(norm_path.read_text())
    )
    normalizer = SmolVLANormalizer(
        state_mean=torch.tensor(norm_sd["state_mean"])
        if isinstance(norm_sd["state_mean"], list)
        else norm_sd["state_mean"],
        state_std=torch.tensor(norm_sd["state_std"])
        if isinstance(norm_sd["state_std"], list)
        else norm_sd["state_std"],
        action_mean=torch.tensor(norm_sd["action_mean"])
        if isinstance(norm_sd["action_mean"], list)
        else norm_sd["action_mean"],
        action_std=torch.tensor(norm_sd["action_std"])
        if isinstance(norm_sd["action_std"], list)
        else norm_sd["action_std"],
        eps=float(norm_sd.get("eps", 1e-8)),
        source_split=str(norm_sd.get("source_split", "train")),
    )

    val_dataset = UnifiedFeatureCacheDataset(val_manifest)
    sample = val_dataset[0]
    context_tokens, context_dim = sample.context.shape

    decoder = SmolExpertActionDecoder.from_pretrained(
        "lerobot/smolvla_base",
        normalizer=normalizer,
        device=device,
        num_steps=num_steps,
        input_channels=context_dim,
    )

    sd = load_file(str(model_path))
    cleaned_sd = {k.replace("model.", ""): v for k, v in sd.items()}
    decoder.load_state_dict(cleaned_sd)
    decoder.eval()

    print(f"  Evaluating Eval-Set 1 (V1, {len(val_dataset)} anchors)...", flush=True)
    res_v1 = evaluate_validation(
        decoder, val_dataset, device=device, batch_size=batch_size, num_steps=num_steps, seed=seed
    )

    res_v2 = None
    if eval2_manifest is not None and eval2_manifest.is_file():
        print(f"  Evaluating Eval-Set 2 (V2, {eval2_manifest})...", flush=True)
        eval2_dataset = UnifiedFeatureCacheDataset(eval2_manifest)
        res_v2 = evaluate_validation(
            decoder,
            eval2_dataset,
            device=device,
            batch_size=batch_size,
            num_steps=num_steps,
            seed=seed + 1000,
        )

    return {"v1": res_v1, "v2": res_v2}


def evaluate_smolvla(
    model_path: Path,
    batches_v1: list[EvaluationBatch],
    batches_v2: list[EvaluationBatch],
    device: torch.device,
) -> dict[str, Any]:
    backend = SmolVLABackend.from_pretrained(str(model_path), device=device)
    print("  Evaluating SmolVLA on V1...", flush=True)
    res_v1 = evaluate_backend(backend, batches_v1, seed=0)
    print("  Evaluating SmolVLA on V2...", flush=True)
    res_v2 = evaluate_backend(backend, batches_v2, seed=0)
    return {
        "v1": {
            "aggregate_rmse": res_v1["aggregate_rmse_deg"],
            "h1": res_v1.get("h1_deg"),
            "first5": res_v1.get("first5_deg"),
        },
        "v2": {
            "aggregate_rmse": res_v2["aggregate_rmse_deg"],
            "h1": res_v2.get("h1_deg"),
            "first5": res_v2.get("first5_deg"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-smolvla", action="store_true", help="Skip SmolVLA full-video evaluation")
    args = parser.parse_args()
    device = torch.device(args.device)

    print("\n" + "=" * 80)
    print("STARTING PROTOCOL 1.1 DUAL-EVALUATION BENCHMARK (V1 vs V2)")
    print("=" * 80)

    # Policy definitions
    policies = [
        {
            "name": "Cosmos 14B SmolExpert",
            "type": "smolexpert",
            "model_path": REPO_ROOT / "outputs/train/cosmos14b-protocol1-smolexpert/best.safetensors",
            "val_manifest": Path("/home/anton/.cache/video-vam/cosmos14b-protocol1-cache/val/manifest.json"),
            "eval2_manifest": Path(
                "/home/anton/.cache/video-vam/cosmos14b-scale100-cache/eval2/manifest.json"
            ),
        },
        {
            "name": "FLUX.2 klein SmolExpert",
            "type": "smolexpert",
            "model_path": REPO_ROOT / "outputs/train/flux2-klein-protocol1-smolexpert/best.safetensors",
            "val_manifest": Path(
                "/home/anton/.cache/video-vam/flux2-klein-protocol1-cache/val/manifest.json"
            ),
            "eval2_manifest": Path(
                "/home/anton/.cache/video-vam/flux2-klein-scale100-cache/eval2/manifest.json"
            ),
        },
        {
            "name": "Cosmos 7B SmolExpert",
            "type": "smolexpert",
            "model_path": REPO_ROOT / "outputs/train/cosmos7b-protocol1-smolexpert/best.safetensors",
            "val_manifest": Path("/home/anton/.cache/video-vam/cosmos7b-protocol1-cache/val/manifest.json"),
            "eval2_manifest": None,
        },
        {
            "name": "Cosmos 2B Pool2 Reference (T=16)",
            "type": "smolexpert",
            "model_path": REPO_ROOT
            / "outputs/train/cube-out-of-box-cosmos-pool2-smolexpert/best.safetensors",
            "val_manifest": REPO_ROOT
            / "outputs/features/cosmos2b_videolora_t2_unpooled/val/manifest.json",  # fallback or update
            "eval2_manifest": None,
        },
        {
            "name": "Cosmos 2B T=2 Undistilled",
            "type": "smolexpert",
            "model_path": REPO_ROOT / "outputs/train/cosmos2b-t2-undistilled-smolexpert/best.safetensors",
            "val_manifest": REPO_ROOT / "outputs/features/cosmos2b_videolora_t2_unpooled/val/manifest.json",
            "eval2_manifest": None,
        },
        {
            "name": "Cosmos 2B T=2 Distilled",
            "type": "smolexpert",
            "model_path": REPO_ROOT / "outputs/train/cosmos2b-t2-distilled-smolexpert/best.safetensors",
            "val_manifest": REPO_ROOT / "outputs/features/cosmos2b_distilled_t2_unpooled/val/manifest.json",
            "eval2_manifest": None,
        },
    ]

    results = {}

    for pol in policies:
        name = pol["name"]
        ckpt = pol["model_path"]
        if not ckpt.is_file():
            print(f"[-] Skipping {name}: checkpoint not found at {ckpt}")
            continue

        print(f"\n[+] Evaluating {name}...")
        try:
            res = evaluate_smolexpert_on_cache(
                ckpt,
                pol["val_manifest"],
                pol["eval2_manifest"],
                device=device,
                batch_size=args.batch_size,
            )
            results[name] = {
                "v1_rmse": round(float(res["v1"]["val_rmse"]), 2),
                "v1_h1": round(float(res["v1"]["val_h1"]), 2),
                "v1_first5": round(float(res["v1"]["val_first5"]), 2),
                "v2_rmse": round(float(res["v2"]["val_rmse"]), 2) if res["v2"] else "N/A",
                "v2_h1": round(float(res["v2"]["val_h1"]), 2) if res["v2"] else "N/A",
                "v2_first5": round(float(res["v2"]["val_first5"]), 2) if res["v2"] else "N/A",
            }
            print(f"  --> V1 RMSE: {results[name]['v1_rmse']} | V2 RMSE: {results[name]['v2_rmse']}")
        except Exception as e:
            print(f"  [!] Error evaluating {name}: {e}")

    # SmolVLA baselines
    if not args.skip_smolvla:
        smolvla_v1_ckpt = (
            REPO_ROOT
            / "outputs/train/cube_out_of_box_il_smolvla_train_only_stats_0_31_20260826_1hr/checkpoints/last/pretrained_model"
        )
        smolvla_v2_ckpt = (
            REPO_ROOT / "outputs/train/cube_out_of_box_scale100_smolvla_1hr/checkpoints/last/pretrained_model"
        )

        if smolvla_v1_ckpt.is_dir() or smolvla_v2_ckpt.is_dir():
            from scripts.video_vam.eval_both_v1_v2 import make_eval_batches

            print("\nBuilding raw-observation evaluation batches for SmolVLA...")
            batches_v1, _ = make_eval_batches(DEFAULT_V1_REPO, list(range(32, 40)), revision=DEFAULT_V1_REV)
            batches_v2, _ = make_eval_batches(DEFAULT_V2_REPO, list(range(90, 100)))

            if smolvla_v1_ckpt.is_dir():
                print("\n[+] Evaluating SmolVLA v1 (trained on eps 0..31)...")
                res = evaluate_smolvla(smolvla_v1_ckpt, batches_v1, batches_v2, device=device)
                results["SmolVLA v1 (canonical)"] = {
                    "v1_rmse": round(float(res["v1"]["val_rmse"]), 2),
                    "v1_h1": round(float(res["v1"]["val_h1"]), 2) if res["v1"].get("h1") else "N/A",
                    "v1_first5": round(float(res["v1"]["val_first5"]), 2)
                    if res["v1"].get("first5")
                    else "N/A",
                    "v2_rmse": round(float(res["v2"]["val_rmse"]), 2),
                    "v2_h1": round(float(res["v2"]["val_h1"]), 2) if res["v2"].get("h1") else "N/A",
                    "v2_first5": round(float(res["v2"]["val_first5"]), 2)
                    if res["v2"].get("first5")
                    else "N/A",
                }

            if smolvla_v2_ckpt.is_dir():
                print("\n[+] Evaluating SmolVLA v2 (trained on 82 eps)...")
                res = evaluate_smolvla(smolvla_v2_ckpt, batches_v1, batches_v2, device=device)
                results["SmolVLA v2 (scale100)"] = {
                    "v1_rmse": round(float(res["v1"]["val_rmse"]), 2),
                    "v1_h1": round(float(res["v1"]["val_h1"]), 2) if res["v1"].get("h1") else "N/A",
                    "v1_first5": round(float(res["v1"]["val_first5"]), 2)
                    if res["v1"].get("first5")
                    else "N/A",
                    "v2_rmse": round(float(res["v2"]["val_rmse"]), 2),
                    "v2_h1": round(float(res["v2"]["val_h1"]), 2) if res["v2"].get("h1") else "N/A",
                    "v2_first5": round(float(res["v2"]["val_first5"]), 2)
                    if res["v2"].get("first5")
                    else "N/A",
                }

    # Summary JSON
    out_file = OUTPUTS_EVAL / "protocol1_1_dual_eval_matrix.json"
    with open(out_file, "w") as f:
        json.dump(
            {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "results": results}, f, indent=2
        )

    print("\n" + "=" * 80)
    print("PROTOCOL 1.1 DUAL-EVALUATION MATRIX SUMMARY")
    print("=" * 80)
    print(f"{'Model / Architecture':<35} | {'V1 Eval (eps 32-39)':<20} | {'V2 Eval (eps 90-99)':<20}")
    print("-" * 80)
    for model_name, metrics in results.items():
        v1_str = f"RMSE: {metrics.get('v1_rmse')}°"
        v2_str = f"RMSE: {metrics.get('v2_rmse')}{'°' if metrics.get('v2_rmse') != 'N/A' else ''}"
        print(f"{model_name:<35} | {v1_str:<20} | {v2_str:<20}")
    print("=" * 80)
    print(f"\nSaved report to: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
