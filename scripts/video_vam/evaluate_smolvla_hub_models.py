#!/usr/bin/env python3
"""Evaluate SmolVLA Hugging Face checkpoints on Orellius/so101_sort_cubes_no_top validation splits.

Computes exact metrics matching:
1. Protocol 1.0 (`train_smolexpert.py` / `OnlineVideoDataset`):
   - Episodes 68..76, 405 anchors (`f in range(4, ep_len, 20)`), native 30 Hz 30-step chunk
     with `~action_is_pad` masking and deterministic per-sample SHA256 noise seeding.
   - Reports: First-5 (Mean), First-5 (Pooled), First-5 Arm, First-5 Gripper,
     H1, Full-30, Arm RMSE, Gripper RMSE.
2. Protocol 1.0 @ 10 Hz Subsampled (`docs/video_vam_leaderboard.md` row 11-06):
   - Same 405 anchors on episodes 68..76, subsampled every 3rd step (`k * 3`) to match 10 Hz.
   - Reports: First-5 @ 10 Hz (`steps 0, 3, 6, 9, 12`), H1 (`step 0`), 1.0s (`10 steps @ 10 Hz`).
3. Canonical 5-Episode Slice `68..72` (`run_smolvla_vs_cosmos3_and_marek_sc.py` / row 10-31):
   - Episodes 68..72 (`f in range(4, ep_len, 20)` = 230 anchors, and `range(0, ep_len - 90, 20)`).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from lerobot.policies.vam.action_rmse import SmolVLABackend

DATASET_REPO_ID = "Orellius/so101_sort_cubes_no_top"
DATASET_REVISION = "a7c156de0e84a81e19457e6d783e4316e9f901e1"
TASK_PROMPT = "Grab the purple cube and put it into the cardboard box, ignore the other objects"

MODELS = [
    (
        "smolvla_lora_rank64_bs32_lr1e-3_10k",
        "hubnemo/so101_sort_cubes_no_top_smolvla_lora_rank64_bs32_lr1e-3_steps10000_schedulerdecaylr1e-4",
    ),
    (
        "smolvla_lora_main_rank64_bs32_lr1e-3_10k",
        "hubnemo/so101_sort_cubes_no_top_smolvla_lora_main_rank64_bs32_lr1e-3_steps10000_schedulerdecaylr1e-4",
    ),
    (
        "smolvla_exp01_bs16_steps20k_imgtf",
        "hubnemo/so101_smolvla_exp01_bs16_steps20k_imgtf",
    ),
    (
        "smolvla_100k_exact_heldout_clean",
        "outputs/train/sort-cubes-smolvla-100k-exact/checkpoints/last/pretrained_model",
    ),
    (
        "smolvla_200k_compiled_step50k_heldout_clean",
        "outputs/train/sort-cubes-smolvla-200k-compiled/checkpoints/050000/pretrained_model",
    ),
    (
        "orellius_smolvla_base_100k_full_dataset",
        "Orellius/so101_sort_cubes_no_top_smolvla_base_100k",
    ),
]


def stable_sample_seed(dataset_id: str, sample_id: str, base_seed: int, offset: int = 0) -> int:
    payload = f"{dataset_id}:{sample_id}:{base_seed}:{offset}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def load_dataset_tables(snap_dir: Path) -> tuple[dict[int, dict[str, Any]], np.ndarray, np.ndarray]:
    ep_files = sorted((snap_dir / "meta/episodes/chunk-000").glob("*.parquet"))
    ep_table = pa.concat_tables([pq.read_table(f) for f in ep_files]).to_pydict()
    ep_meta = {
        int(ep_id): {
            "from_idx": int(ep_table["dataset_from_index"][i]),
            "to_idx": int(ep_table["dataset_to_index"][i]),
            "length": int(ep_table["length"][i]),
            "vid_file": int(ep_table["videos/observation.images.front/file_index"][i]),
            "from_ts": float(ep_table["videos/observation.images.front/from_timestamp"][i]),
        }
        for i, ep_id in enumerate(ep_table["episode_index"])
    }
    data_files = sorted((snap_dir / "data/chunk-000").glob("*.parquet"))
    data_table = pa.concat_tables(
        [pq.read_table(f, columns=["observation.state", "action"]) for f in data_files]
    ).to_pydict()
    all_states_np = np.asarray(data_table["observation.state"], dtype=np.float32)
    all_actions_np = np.asarray(data_table["action"], dtype=np.float32)
    return ep_meta, all_states_np, all_actions_np


def decode_anchor_frames(
    snap_dir: Path,
    ep_meta: dict[int, dict[str, Any]],
    anchors: list[tuple[int, int]],
) -> dict[tuple[int, int], np.ndarray]:
    by_vid: dict[int, list[tuple[int, int]]] = {}
    for ep, rel in anchors:
        by_vid.setdefault(ep_meta[ep]["vid_file"], []).append((ep, rel))

    frames_rgb: dict[tuple[int, int], np.ndarray] = {}
    for vid_idx, ep_rels in sorted(by_vid.items()):
        mp4_path = snap_dir / f"videos/observation.images.front/chunk-000/file-{vid_idx:03d}.mp4"
        targets: dict[int, list[tuple[int, int]]] = {}
        for ep, rel in ep_rels:
            base_vf = int(round(ep_meta[ep]["from_ts"] * 30.0))
            targets.setdefault(base_vf + rel, []).append((ep, rel))
        max_vf = max(targets.keys())
        container = av.open(str(mp4_path))
        for vf_i, frame in enumerate(container.decode(video=0)):
            if vf_i in targets:
                arr = frame.to_ndarray(format="rgb24")
                for key in targets[vf_i]:
                    frames_rgb[key] = arr
            if vf_i >= max_vf:
                break
        container.close()
    return frames_rgb


def compute_metrics_from_predictions(
    preds_50: dict[tuple[int, int], torch.Tensor],
    anchors: list[tuple[int, int]],
    ep_meta: dict[int, dict[str, Any]],
    all_actions_np: np.ndarray,
) -> dict[str, Any]:
    action_dim = 6
    # 1. Native 30 Hz (30-step chunk, exact train_smolexpert.py OnlineVideoDataset protocol)
    sq_by_joint_30hz = torch.zeros(action_dim, dtype=torch.float64)
    prefix_sq_by_step_30hz = [torch.zeros(action_dim, dtype=torch.float64) for _ in range(5)]
    prefix_tokens_by_step_30hz = [0 for _ in range(5)]
    valid_tokens_30hz = 0

    # 2. Subsampled 10 Hz (every 3rd step: k*3 for k=0..9 [1.0s] and k=0..15 [1.6s])
    sq_by_joint_10hz_10 = torch.zeros(action_dim, dtype=torch.float64)
    prefix_sq_by_step_10hz = [torch.zeros(action_dim, dtype=torch.float64) for _ in range(5)]
    prefix_tokens_by_step_10hz = [0 for _ in range(5)]
    valid_tokens_10hz_10 = 0

    for ep, rel in anchors:
        pred = preds_50[(ep, rel)].double()  # [50, 6]
        f0 = ep_meta[ep]["from_idx"]
        ep_len = ep_meta[ep]["length"]

        # Native 30 Hz target & padding mask (30 steps)
        idx_30 = [f0 + min(ep_len - 1, rel + k) for k in range(30)]
        pad_30 = torch.tensor([rel + k >= ep_len for k in range(30)], dtype=torch.bool)
        valid_30 = (~pad_30).unsqueeze(-1)  # [30, 1]
        targ_30 = torch.from_numpy(all_actions_np[idx_30]).double()

        sq_30 = (pred[:30] - targ_30).square() * valid_30
        sq_by_joint_30hz += sq_30.sum(dim=0)
        valid_tokens_30hz += int((~pad_30).sum().item())

        for h in range(5):
            if not pad_30[h]:
                prefix_sq_by_step_30hz[h] += sq_30[h]
                prefix_tokens_by_step_30hz[h] += 1

        # 10 Hz subsampled target & padding mask (10 steps = 1.0s @ 10Hz, steps 0, 3, ..., 27)
        idx_10 = [f0 + min(ep_len - 1, rel + k * 3) for k in range(10)]
        pad_10 = torch.tensor([rel + k * 3 >= ep_len for k in range(10)], dtype=torch.bool)
        valid_10 = (~pad_10).unsqueeze(-1)
        targ_10 = torch.from_numpy(all_actions_np[idx_10]).double()
        pred_10 = pred[[k * 3 for k in range(10)]]

        sq_10 = (pred_10 - targ_10).square() * valid_10
        sq_by_joint_10hz_10 += sq_10.sum(dim=0)
        valid_tokens_10hz_10 += int((~pad_10).sum().item())

        for h in range(5):
            if not pad_10[h]:
                prefix_sq_by_step_10hz[h] += sq_10[h]
                prefix_tokens_by_step_10hz[h] += 1

    # Aggregate 30 Hz metrics
    full30_rmse = math.sqrt(float(sq_by_joint_30hz.sum().item()) / (valid_tokens_30hz * action_dim))
    arm_30_rmse = math.sqrt(float(sq_by_joint_30hz[:5].sum().item()) / (valid_tokens_30hz * 5))
    grip_30_rmse = math.sqrt(float(sq_by_joint_30hz[5].item()) / valid_tokens_30hz)
    h1_rmse = math.sqrt(
        float(prefix_sq_by_step_30hz[0].sum().item()) / (prefix_tokens_by_step_30hz[0] * action_dim)
    )

    step_rmses_30hz = [
        math.sqrt(
            float(prefix_sq_by_step_30hz[h].sum().item()) / (prefix_tokens_by_step_30hz[h] * action_dim)
        )
        for h in range(5)
    ]
    f5_mean_30hz = sum(step_rmses_30hz) / 5.0
    f5_sq_by_joint_30hz = sum(prefix_sq_by_step_30hz)
    f5_tokens_30hz = sum(prefix_tokens_by_step_30hz)
    f5_pooled_30hz = math.sqrt(float(f5_sq_by_joint_30hz.sum().item()) / (f5_tokens_30hz * action_dim))
    f5_arm_30hz = math.sqrt(float(f5_sq_by_joint_30hz[:5].sum().item()) / (f5_tokens_30hz * 5))
    f5_grip_30hz = math.sqrt(float(f5_sq_by_joint_30hz[5].item()) / f5_tokens_30hz)

    # Aggregate 10 Hz subsampled metrics
    full10_10hz_rmse = math.sqrt(
        float(sq_by_joint_10hz_10.sum().item()) / (valid_tokens_10hz_10 * action_dim)
    )
    step_rmses_10hz = [
        math.sqrt(
            float(prefix_sq_by_step_10hz[h].sum().item()) / (prefix_tokens_by_step_10hz[h] * action_dim)
        )
        for h in range(5)
    ]
    f5_mean_10hz = sum(step_rmses_10hz) / 5.0
    f5_sq_by_joint_10hz = sum(prefix_sq_by_step_10hz)
    f5_tokens_10hz = sum(prefix_tokens_by_step_10hz)
    f5_pooled_10hz = math.sqrt(float(f5_sq_by_joint_10hz.sum().item()) / (f5_tokens_10hz * action_dim))
    f5_arm_10hz = math.sqrt(float(f5_sq_by_joint_10hz[:5].sum().item()) / (f5_tokens_10hz * 5))
    f5_grip_10hz = math.sqrt(float(f5_sq_by_joint_10hz[5].item()) / f5_tokens_10hz)

    return {
        "n_anchors": len(anchors),
        "native_30hz": {
            "full30_rmse": round(full30_rmse, 3),
            "h1_rmse": round(h1_rmse, 3),
            "first5_mean_rmse": round(f5_mean_30hz, 3),
            "first5_pooled_rmse": round(f5_pooled_30hz, 3),
            "first5_arm_rmse_deg": round(f5_arm_30hz, 3),
            "first5_gripper_rmse": round(f5_grip_30hz, 3),
            "full30_arm_rmse_deg": round(arm_30_rmse, 3),
            "full30_gripper_rmse": round(grip_30_rmse, 3),
            "per_step_0_to_4_rmse": [round(x, 3) for x in step_rmses_30hz],
        },
        "subsampled_10hz": {
            "h10_1s_rmse": round(full10_10hz_rmse, 3),
            "h1_rmse": round(h1_rmse, 3),
            "first5_mean_rmse": round(f5_mean_10hz, 3),
            "first5_pooled_rmse": round(f5_pooled_10hz, 3),
            "first5_arm_rmse_deg": round(f5_arm_10hz, 3),
            "first5_gripper_rmse": round(f5_grip_10hz, 3),
            "per_step_0_3_6_9_12_rmse": [round(x, 3) for x in step_rmses_10hz],
        },
    }


@torch.no_grad()
def predict_all_anchors(
    backend: SmolVLABackend,
    anchors: list[tuple[int, int]],
    frames_rgb: dict[tuple[int, int], np.ndarray],
    ep_meta: dict[int, dict[str, Any]],
    all_states_np: np.ndarray,
    num_steps_override: int | None = None,
    batch_size: int = 16,
) -> tuple[dict[tuple[int, int], torch.Tensor], float]:
    policy = backend.policy
    base_policy = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
    orig_num_steps = base_policy.config.num_steps
    if num_steps_override is not None:
        base_policy.config.num_steps = num_steps_override
        if hasattr(policy, "config"):
            policy.config.num_steps = num_steps_override

    dev, dt = backend._device_and_dtype()
    chunk_size = int(base_policy.config.chunk_size)
    max_action_dim = int(base_policy.config.max_action_dim)
    dataset_id = f"{DATASET_REPO_ID}@{DATASET_REVISION}"

    preds: dict[tuple[int, int], torch.Tensor] = {}
    latencies_ms: list[float] = []

    try:
        # Measure single-sample latency on first 5 anchors
        for ep, rel in anchors[:5]:
            f0 = ep_meta[ep]["from_idx"]
            rgb = frames_rgb[(ep, rel)]
            st = all_states_np[f0 + rel]
            obs = {
                "observation.images.front": torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0,
                "observation.state": torch.from_numpy(st).float(),
                "task": TASK_PROMPT,
            }
            proc = backend.preprocessor(obs)
            proc = {k: (v.to(device=dev) if isinstance(v, torch.Tensor) else v) for k, v in proc.items()}
            gen = torch.Generator(device=dev).manual_seed(0)
            noise = torch.randn((1, chunk_size, max_action_dim), generator=gen, device=dev, dtype=dt)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            raw = policy.predict_action_chunk(proc, noise=noise)
            _ = backend.postprocessor(raw)
            torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        # Batched evaluation over all anchors with deterministic per-sample noise
        for start in range(0, len(anchors), batch_size):
            batch_anchors = anchors[start : start + batch_size]
            imgs_t = []
            states_t = []
            noises_t = []
            for ep, rel in batch_anchors:
                f0 = ep_meta[ep]["from_idx"]
                abs_idx = f0 + rel
                rgb = frames_rgb[(ep, rel)]
                st = all_states_np[abs_idx]
                imgs_t.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
                states_t.append(torch.from_numpy(st).float())

                sample_id = f"episode-{ep:04d}-frame-{abs_idx:06d}"
                s_seed = stable_sample_seed(dataset_id, sample_id, base_seed=42, offset=0)
                gen = torch.Generator(device=dev).manual_seed(s_seed)
                noises_t.append(
                    torch.randn((1, chunk_size, max_action_dim), generator=gen, device=dev, dtype=dt)
                )

            # Preprocess first sample to get tokenized language keys, then stack batch tensors
            proc_list = [
                backend.preprocessor(
                    {
                        "observation.images.front": img,
                        "observation.state": st,
                        "task": TASK_PROMPT,
                    }
                )
                for img, st in zip(imgs_t, states_t, strict=True)
            ]
            batch_proc: dict[str, Any] = {}
            for k in proc_list[0]:
                if isinstance(proc_list[0][k], torch.Tensor):
                    batch_proc[k] = torch.cat([p[k] for p in proc_list], dim=0).to(device=dev)
                else:
                    batch_proc[k] = proc_list[0][k]

            noise_batch = torch.cat(noises_t, dim=0)
            if hasattr(policy, "reset"):
                policy.reset()
            raw_chunk = policy.predict_action_chunk(batch_proc, noise=noise_batch)
            phys_chunk = backend.postprocessor(raw_chunk).cpu().float()[:, :50, :6]
            for i, key in enumerate(batch_anchors):
                preds[key] = phys_chunk[i]
    finally:
        base_policy.config.num_steps = orig_num_steps
        if hasattr(policy, "config"):
            policy.config.num_steps = orig_num_steps

    return preds, round(float(np.median(latencies_ms[1:])), 1)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snap_dir = sorted(
        (
            Path.home()
            / ".cache/huggingface/lerobot/hub/datasets--Orellius--so101_sort_cubes_no_top/snapshots"
        ).glob("*")
    )[-1]
    print(f"Using dataset snapshot: {snap_dir}", flush=True)
    ep_meta, all_states_np, all_actions_np = load_dataset_tables(snap_dir)

    # Protocol 1.0 anchors: episodes 68..76, f in range(4, ep_len, 20) -> 405 anchors
    anchors_68_76 = [(ep, f) for ep in range(68, 77) for f in range(4, ep_meta[ep]["length"], 20)]
    # 5-episode slice 68..72 from Protocol 1.0 anchors -> 230 anchors
    anchors_68_72_p1 = [(ep, f) for (ep, f) in anchors_68_76 if ep <= 72]
    # Legacy run_smolvla_vs_cosmos3_and_marek_sc.py anchors on 68..72: range(0, ep_len - 90, 20) -> 208 anchors
    anchors_68_72_legacy = [
        (ep, rel) for ep in range(68, 73) for rel in range(0, max(1, ep_meta[ep]["length"] - 90), 20)
    ]

    all_needed_anchors = sorted(set(anchors_68_76) | set(anchors_68_72_legacy))
    print(
        f"Decoding {len(all_needed_anchors)} unique anchor frames across episodes 68..76 "
        f"(Protocol 1.0 68..76: {len(anchors_68_76)}, 68..72 P1: {len(anchors_68_72_p1)}, "
        f"68..72 legacy: {len(anchors_68_72_legacy)})...",
        flush=True,
    )
    t0 = time.perf_counter()
    frames_rgb = decode_anchor_frames(snap_dir, ep_meta, all_needed_anchors)
    print(f"Decoded {len(frames_rgb)} frames in {time.perf_counter() - t0:.2f}s.", flush=True)

    out_path = Path("outputs/evaluation/smolvla_hub_models_eval_20261106.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = json.loads(out_path.read_text()) if out_path.is_file() else {}
    for short_name, repo_or_path in MODELS:
        if short_name in results:
            print(f"Skipping already evaluated: {short_name}", flush=True)
            continue
        print("\n=====================================================================", flush=True)
        print(f"Loading & Evaluating: {short_name} ({repo_or_path})", flush=True)
        print("=====================================================================", flush=True)
        backend = SmolVLABackend.from_pretrained(repo_or_path, device=device)
        base_policy = (
            backend.policy.get_base_model() if hasattr(backend.policy, "get_base_model") else backend.policy
        )
        native_steps = int(base_policy.config.num_steps)

        model_res: dict[str, Any] = {
            "repo_or_path": repo_or_path,
            "native_num_steps": native_steps,
            "use_peft": bool(getattr(base_policy.config, "use_peft", False)),
        }

        for steps_eval in sorted({native_steps, 3, 10}):
            preds, med_lat_ms = predict_all_anchors(
                backend,
                all_needed_anchors,
                frames_rgb,
                ep_meta,
                all_states_np,
                num_steps_override=steps_eval,
                batch_size=16,
            )
            m_68_76 = compute_metrics_from_predictions(preds, anchors_68_76, ep_meta, all_actions_np)
            m_68_72_p1 = compute_metrics_from_predictions(preds, anchors_68_72_p1, ep_meta, all_actions_np)
            m_68_72_leg = compute_metrics_from_predictions(
                preds, anchors_68_72_legacy, ep_meta, all_actions_np
            )
            model_res[f"euler_{steps_eval}_steps"] = {
                "median_latency_ms": med_lat_ms,
                "val_68_76_protocol1_405_anchors": m_68_76,
                "val_68_72_protocol1_230_anchors": m_68_72_p1,
                "val_68_72_legacy_208_anchors": m_68_72_leg,
            }
            p1_30 = m_68_76["native_30hz"]
            p1_10 = m_68_76["subsampled_10hz"]
            leg_30 = m_68_72_leg["native_30hz"]
            print(
                f"  [Euler={steps_eval:2d} | {med_lat_ms:5.1f}ms] "
                f"68..76 (405 anchors, 30Hz): F5_mean={p1_30['first5_mean_rmse']:.2f} "
                f"(pooled={p1_30['first5_pooled_rmse']:.2f}, arm={p1_30['first5_arm_rmse_deg']:.2f}°, grip={p1_30['first5_gripper_rmse']:.2f}) | "
                f"H1={p1_30['h1_rmse']:.2f} | Full30={p1_30['full30_rmse']:.2f} || "
                f"10Hz: F5_pool={p1_10['first5_pooled_rmse']:.2f}, 1.0s={p1_10['h10_1s_rmse']:.2f} || "
                f"68..72 (30Hz): F5_pool={leg_30['first5_pooled_rmse']:.2f}, Full30={leg_30['full30_rmse']:.2f}",
                flush=True,
            )

        results[short_name] = model_res
        out_path.write_text(json.dumps(results, indent=2) + "\n")
        del backend
        torch.cuda.empty_cache()

    print(f"\nSaved full evaluation report to {out_path}", flush=True)


if __name__ == "__main__":
    main()
