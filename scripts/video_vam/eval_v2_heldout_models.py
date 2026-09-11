#!/usr/bin/env python3
"""Evaluate V1-trained Video Action Models on V2 Held-Out Dataset (episodes 90..99).

Evaluates:
  1. Cosmos 2B T=2 Undistilled
  2. Cosmos 2B T=2 Distilled
  3. Cosmos 7B
under strict Protocol 1.1 evaluation semantics (10 Euler flow steps, deterministic seed,
masked action RMSE against ground truth).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import Tensor

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.vam.cosmos7b_extractor import Cosmos7BExtractor, Cosmos7BExtractorConfig
from lerobot.policies.vam.cosmos_lora import inject_lora, load_lora_state_dict, merge_lora_file_into_base
from lerobot.policies.vam.cosmos_predict2_extractor import (
    CosmosPredict2Extractor,
    CosmosPredict2ExtractorConfig,
)
from lerobot.policies.vam.cosmos_prompt_embedding import load_prompt_embedding
from lerobot.policies.vam.smol_expert import (
    SmolExpertActionDecoder,
    SmolVLANormalizer,
)

REPO_ROOT = Path("/home/anton/lerobot-video-vam")
MIMIC_DIR = Path("/home/anton/.cache/video-vam/mimic-video-f2833903")
COSMOS2B_PT = MIMIC_DIR / "video_backbone/v2w_pretrained_cosmos.pt"
COSMOS2B_TOKENIZER = MIMIC_DIR / "video_backbone/tokenizer/tokenizer.pth"
PROMPT_PATH = Path("/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-t5-11b.safetensors")
VIDEO_LORA = REPO_ROOT / "outputs/train/cosmos-video-lora-step6000/best_lora.safetensors"
DISTILLED_LORA = REPO_ROOT / "outputs/train/cosmos2b-t2-direct-distilled-lora/best_lora.safetensors"

V2_ROOT = Path(
    "/home/anton/.cache/huggingface/lerobot/hub/datasets--Orellius--cube_out_of_box_v2/snapshots/5d0325cc1412f4774223a0beb528958108814962"
)
ANCHOR_MANIFEST = Path("/home/anton/.cache/video-vam/flux2-klein-scale100-cache/eval2/manifest.json")
OUTPUT_MATRIX = REPO_ROOT / "outputs/evaluation/protocol1_1_dual_eval_matrix.json"


def load_normalizer(run_dir: Path) -> SmolVLANormalizer:
    norm_path = run_dir / "normalizer.safetensors"
    if not norm_path.is_file():
        norm_path = run_dir / "normalizer.json"
    norm_sd = (
        load_file(str(norm_path)) if norm_path.suffix == ".safetensors" else json.loads(norm_path.read_text())
    )
    return SmolVLANormalizer(
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


def load_decoder(run_dir: Path, input_channels: int, device: torch.device) -> SmolExpertActionDecoder:
    normalizer = load_normalizer(run_dir)
    decoder = SmolExpertActionDecoder.from_pretrained(
        "lerobot/smolvla_base",
        normalizer=normalizer,
        device=device,
        num_steps=10,
        input_channels=input_channels,
    )
    sd = load_file(str(run_dir / "best.safetensors"))
    cleaned = {k.replace("model.", ""): v for k, v in sd.items()}
    decoder.load_state_dict(cleaned)
    decoder.eval()
    return decoder


def evaluate_policy_on_contexts(
    decoder: SmolExpertActionDecoder,
    contexts: list[Tensor],
    states: list[Tensor],
    targets: list[Tensor],
    pads: list[Tensor],
    device: torch.device,
    seed: int = 1000,
) -> dict[str, float]:
    total_sq_err = 0.0
    total_arm_sq_err = 0.0
    total_grip_sq_err = 0.0
    h1_sq_err = 0.0
    first5_sq_err = 0.0
    valid_count = 0
    h1_valid_count = 0
    first5_valid_count = 0

    for i, (ctx, st, tgt, pad) in enumerate(zip(contexts, states, targets, pads)):
        st_in = st[None, None].to(device=device, dtype=torch.float32)
        ctx_in = ctx.to(device=device, dtype=torch.bfloat16)
        pred = decoder.sample_actions(st_in, ctx_in, seed=seed + i)[0].float().cpu()

        mask = ~pad  # [30]
        if not mask.any():
            continue

        err = (pred - tgt) ** 2  # [30, 6]

        # Valid masked counts
        for t in range(30):
            if mask[t]:
                total_sq_err += err[t].sum().item()
                total_arm_sq_err += err[t, :5].sum().item()
                total_grip_sq_err += err[t, 5].item()
                valid_count += 6

                if t == 0:
                    h1_sq_err += err[t].sum().item()
                    h1_valid_count += 6
                if t < 5:
                    first5_sq_err += err[t].sum().item()
                    first5_valid_count += 6

    overall_rmse = math.sqrt(total_sq_err / max(1, valid_count))
    arm_rmse = math.sqrt(total_arm_sq_err / max(1, valid_count * (5 / 6)))
    grip_rmse = math.sqrt(total_grip_sq_err / max(1, valid_count * (1 / 6)))
    h1_rmse = math.sqrt(h1_sq_err / max(1, h1_valid_count))
    first5_rmse = math.sqrt(first5_sq_err / max(1, first5_valid_count))

    return {
        "rmse": round(overall_rmse, 2),
        "arm_rmse": round(arm_rmse, 2),
        "gripper_rmse": round(grip_rmse, 2),
        "h1": round(h1_rmse, 2),
        "first5": round(first5_rmse, 2),
    }


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 80)
    print("EVALUATING COSMOS 2B AND 7B ON V2 HELD-OUT (EPISODES 90..99)")
    print("=" * 80)

    # 1. Load V2 anchors
    anchor_data = json.loads(ANCHOR_MANIFEST.read_text())["entries"]
    print(f"Loaded {len(anchor_data)} anchor specifications from {ANCHOR_MANIFEST.name}")

    # 2. Load dataset observations
    print("Loading V2 demonstration frames...", flush=True)
    ds = LeRobotDataset(
        "Orellius/cube_out_of_box_v2",
        root=V2_ROOT,
        episodes=list(range(90, 100)),
        delta_timestamps={"action": [i / 10 for i in range(30)]},
        return_uint8=True,
        download_videos=False,
    )

    # Prepare raw camera inputs and targets
    frames = []
    states = []
    targets = []
    pads = []

    print("Indexing dataset frame indices...", flush=True)
    frame_to_idx = {int(ds.reader.hf_dataset[i]["index"]): i for i in range(len(ds))}
    for entry in anchor_data:
        frame_idx = int(entry["frame_index"])
        if frame_idx not in frame_to_idx:
            continue
        # Stack 5 causal observation frames [t-4..t]
        window_frames = [
            ds[frame_to_idx[f]]["observation.images.front"] for f in range(frame_idx - 4, frame_idx + 1)
        ]
        cam_t = torch.stack(window_frames, dim=1).unsqueeze(0).contiguous()  # [1, 3, 5, 480, 640]
        row = ds[frame_to_idx[frame_idx]]
        frames.append(cam_t)
        states.append(row["observation.state"].float())
        targets.append(row["action"].float())
        pads.append(row["action_is_pad"].bool())

    print(f"Prepared {len(frames)} evaluation anchor batches.")

    # --------------------------------------------------------------------------
    # Model 1: Cosmos 2B T=2 Undistilled
    # --------------------------------------------------------------------------
    print("\n[+] Extracting features & evaluating: Cosmos 2B T=2 Undistilled...")
    cfg = CosmosPredict2ExtractorConfig(
        checkpoint_path=COSMOS2B_PT,
        tokenizer_path=COSMOS2B_TOKENIZER,
        device=str(device),
        dtype="bfloat16",
        state_t=2,
        hidden_layer=20,
        high_noise_sigma=80.0,
        seed=0,
        vae_input_mode="observed_prefix",
    )
    extractor = CosmosPredict2Extractor(cfg)
    merge_lora_file_into_base(extractor.backbone, VIDEO_LORA)
    prompt_emb = load_prompt_embedding(PROMPT_PATH).embedding.to(device)

    contexts_undistilled = []
    t0 = time.perf_counter()
    for i, frame in enumerate(frames):
        with torch.no_grad():
            res = extractor.extract(frame.to(device), prompt_emb, noise_seed=i)
            contexts_undistilled.append(res.tokens.cpu())
    print(f"  Extraction finished in {time.perf_counter() - t0:.1f}s")

    decoder_undist = load_decoder(
        REPO_ROOT / "outputs/train/cosmos2b-t2-undistilled-smolexpert", 2048, device
    )
    res_undist_v2 = evaluate_policy_on_contexts(
        decoder_undist, contexts_undistilled, states, targets, pads, device
    )
    print(f"  Cosmos 2B T=2 Undistilled V2 Score: {res_undist_v2}")

    # --------------------------------------------------------------------------
    # Model 2: Cosmos 2B T=2 Distilled
    # --------------------------------------------------------------------------
    print("\n[+] Extracting features & evaluating: Cosmos 2B T=2 Distilled...")
    inject_lora(extractor.backbone, block_indices=tuple(range(20)))
    load_lora_state_dict(extractor.backbone, str(DISTILLED_LORA))

    contexts_distilled = []
    t0 = time.perf_counter()
    for i, frame in enumerate(frames):
        with torch.no_grad():
            res = extractor.extract(frame.to(device), prompt_emb, noise_seed=i)
            contexts_distilled.append(res.tokens.cpu())
    print(f"  Extraction finished in {time.perf_counter() - t0:.1f}s")

    del extractor
    torch.cuda.empty_cache()

    decoder_dist = load_decoder(REPO_ROOT / "outputs/train/cosmos2b-t2-distilled-smolexpert", 2048, device)
    res_dist_v2 = evaluate_policy_on_contexts(decoder_dist, contexts_distilled, states, targets, pads, device)
    print(f"  Cosmos 2B T=2 Distilled V2 Score: {res_dist_v2}")

    # --------------------------------------------------------------------------
    # Model 3: Cosmos 7B
    # --------------------------------------------------------------------------
    print("\n[+] Extracting features & evaluating: Cosmos 7B...")
    c7b_snap = Path(
        "/home/anton/.cache/huggingface/hub/models--Arsh9210--Cosmos-1.0-Diffusion-7B-Video2World/snapshots/f348f5a46ed8d5979c1b3d3288b378b7c1ed963e/transformer"
    )
    c7b_cfg = Cosmos7BExtractorConfig(
        checkpoint_path=c7b_snap,
        device=str(device),
        dtype="bfloat16",
        hidden_layers=(14, 20),
        patch_size=(1, 2, 2),
        pool_spatial=2,
        concat_layers=True,
    )
    c7b_extractor = Cosmos7BExtractor(config=c7b_cfg)
    c7b_extractor.normalizer.vae_path = Path("/home/anton/.cache/video-vam/cosmos-14b/vae")

    contexts_7b = []
    t0 = time.perf_counter()
    for i, frame in enumerate(frames):
        with torch.no_grad():
            res = c7b_extractor.extract(frame.to(device))
            contexts_7b.append(res.tokens.cpu())
    print(f"  Cosmos 7B extraction finished in {time.perf_counter() - t0:.1f}s")

    del c7b_extractor
    torch.cuda.empty_cache()

    decoder_7b = load_decoder(REPO_ROOT / "outputs/train/cosmos7b-protocol1-smolexpert", 8192, device)
    res_7b_v2 = evaluate_policy_on_contexts(decoder_7b, contexts_7b, states, targets, pads, device)
    print(f"  Cosmos 7B V2 Score: {res_7b_v2}")

    # --------------------------------------------------------------------------
    # Update Benchmark Matrix JSON
    # --------------------------------------------------------------------------
    matrix = json.loads(OUTPUT_MATRIX.read_text()) if OUTPUT_MATRIX.is_file() else {"results": {}}
    matrix["results"]["Cosmos 2B T=2 Undistilled"]["v2_rmse"] = res_undist_v2["rmse"]
    matrix["results"]["Cosmos 2B T=2 Undistilled"]["v2_h1"] = res_undist_v2["h1"]
    matrix["results"]["Cosmos 2B T=2 Undistilled"]["v2_first5"] = res_undist_v2["first5"]

    matrix["results"]["Cosmos 2B T=2 Distilled"]["v2_rmse"] = res_dist_v2["rmse"]
    matrix["results"]["Cosmos 2B T=2 Distilled"]["v2_h1"] = res_dist_v2["h1"]
    matrix["results"]["Cosmos 2B T=2 Distilled"]["v2_first5"] = res_dist_v2["first5"]

    matrix["results"]["Cosmos 7B SmolExpert"]["v2_rmse"] = res_7b_v2["rmse"]
    matrix["results"]["Cosmos 7B SmolExpert"]["v2_h1"] = res_7b_v2["h1"]
    matrix["results"]["Cosmos 7B SmolExpert"]["v2_first5"] = res_7b_v2["first5"]

    OUTPUT_MATRIX.write_text(json.dumps(matrix, indent=2))
    print(f"\nUpdated {OUTPUT_MATRIX}")

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL COMPLETE BENCHMARK SUMMARY (V1 vs V2 under Protocol 1.1)")
    print("=" * 80)
    print(f"{'Model / Architecture':<35} | {'V1 Eval (eps 32-39)':<20} | {'V2 Eval (eps 90-99)':<20}")
    print("-" * 80)
    for model_name, m in matrix["results"].items():
        v1 = f"RMSE: {m.get('v1_rmse')}°"
        v2 = f"RMSE: {m.get('v2_rmse')}{'°' if m.get('v2_rmse') != 'N/A' else ''}"
        print(f"{model_name:<35} | {v1:<20} | {v2:<20}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
