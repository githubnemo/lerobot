#!/usr/bin/env python3
"""Offline Real-Data Warm-Start for EXPO Residual Actor + Concave Critic with Marek Score Centering.

Trains `ExpoResidualCritic` (`delta_a_phi` + `Q_psi`) on real `Orellius/so101_sort_cubes_no_top`
episodes (`0..67`) anchored on top of a frozen Video-VAM checkpoint (default:
`outputs/train/sort-cubes-cosmos3-edge-optimized-bs64`), and validates held-out `68..76`
First-5 / H1 / Full-30 RMSE improvement when selecting & editing K=6 candidates.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from safetensors.torch import load_file

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.vam.config import VideoVAMDatasetConfig
from lerobot.policies.vam.cosmos3_features import Cosmos3ExtractorConfig, Cosmos3FeatureExtractor
from lerobot.policies.vam.expo_residual import (
    ExpoResidualCritic,
    compute_marek_score_centered_loss,
)
from lerobot.policies.vam.smol_expert import SmolExpertActionDecoder, SmolVLANormalizer
from scripts.video_vam.train_smolexpert import OnlineVideoDataset, unaugmented_temporal_window_gpu


def load_base_vam_components(
    checkpoint_dir: Path,
    device: torch.device,
) -> tuple[Cosmos3FeatureExtractor, SmolExpertActionDecoder, dict[str, Any]]:
    cfg = json.loads((checkpoint_dir / "config.json").read_text())
    norm_sd = load_file(str(checkpoint_dir / "normalizer.safetensors"))
    normalizer = SmolVLANormalizer(
        state_mean=norm_sd["state_mean"],
        state_std=norm_sd["state_std"],
        action_mean=norm_sd["action_mean"],
        action_std=norm_sd["action_std"],
        eps=float(norm_sd.get("eps", 1e-8)),
        source_split="train",
    )

    ext_cfg = Cosmos3ExtractorConfig(
        backbone_name="cosmos3-edge",
        checkpoint_path=Path(cfg.get("cosmos3_checkpoint", "/home/anton/.cache/video-vam/cosmos3-edge")),
        hidden_layers=(20,),
        device=str(device),
        dtype="bfloat16",
        fps=10.0,
        base_fps=24.0,
        prompt="take cube out of box",
        lora_checkpoint=Path(cfg["cosmos3_lora_weights"]) if cfg.get("cosmos3_lora_weights") else None,
        lora_rank=int(cfg.get("cosmos3_lora_rank", 16)),
        lora_alpha=float(cfg.get("cosmos3_lora_alpha", 32.0)),
        trainable_lora=False,
        compile_dit=True,
        compile_vae=True,
        vae_compile_mode="default",
    )
    extractor = Cosmos3FeatureExtractor(ext_cfg)
    extractor.eval()

    weights_path = checkpoint_dir / "best.safetensors"
    if not weights_path.is_file():
        weights_path = checkpoint_dir / "last.safetensors"
    decoder = SmolExpertActionDecoder.from_training_checkpoint(
        weights_path,
        normalizer=normalizer,
        expert_checkpoint="lerobot/smolvla_base",
        device=device,
        num_steps=10,
        input_channels=2048,
    )
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    return extractor, decoder, cfg


@torch.no_grad()
def build_cached_anchor_split(
    ds: OnlineVideoDataset,
    extractor: Cosmos3FeatureExtractor,
    decoder: SmolExpertActionDecoder,
    device: torch.device,
    k_candidates: int = 6,
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    ep_init_pooled: dict[int, torch.Tensor] = {}
    ep_lengths: dict[int, int] = {ep: int(ds.lerobot_ds.meta.episodes[ep]["length"]) for ep in ds.episodes}

    total = len(ds)
    t0 = time.time()
    for start in range(0, total, batch_size):
        batch_items = [ds[i] for i in range(start, min(start + batch_size, total))]
        raw_rgb = torch.stack([item.rgb.to(device=device, non_blocking=True) for item in batch_items])
        rgb_in = unaugmented_temporal_window_gpu(raw_rgb)
        states = torch.stack([item.state for item in batch_items]).to(device=device, dtype=torch.float32)
        actions = torch.stack([item.action for item in batch_items]).to(device=device, dtype=torch.float32)
        pads = torch.stack([item.action_is_pad for item in batch_items]).to(device=device, dtype=torch.bool)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            latents = extractor.encode_latents(rgb_in)
            tapped = extractor.forward_transformer_blocks(latents)
            ctx_batch = tapped[20].to(device=device)

        pooled_batch = ExpoResidualCritic.pool_ctx_tokens(ctx_batch).float().cpu()

        # Sample K candidate chunks from frozen base decoder
        cand_list = []
        for k in range(k_candidates):
            eps_k = torch.cat(
                [decoder._noise_for_seed(1, device, 42 + k * 97 + i) for i in range(len(batch_items))],
                dim=0,
            )
            c_k = decoder.sample_actions(states, ctx_batch, noise=eps_k, num_steps=10, use_cuda_graph=False)
            cand_list.append(c_k.float().cpu())
        # [B, K, 30, 6]
        cands_bk = torch.stack(cand_list, dim=1)

        for b_i, item in enumerate(batch_items):
            ep = int(item.episode_index)
            f_idx = int(item.frame_index)
            if ep not in ep_init_pooled:
                ep_init_pooled[ep] = pooled_batch[b_i].clone()
            ep_len = max(ep_lengths.get(ep, 600), 1)
            t_norm = torch.tensor([min(1.0, float(f_idx) / float(ep_len))], dtype=torch.float32)
            records.append(
                {
                    "episode_index": ep,
                    "frame_index": f_idx,
                    "ctx": pooled_batch[b_i],
                    "init_ctx": ep_init_pooled[ep],
                    "state": states[b_i].cpu(),
                    "t_norm": t_norm,
                    "gt_action": actions[b_i].cpu(),
                    "action_is_pad": pads[b_i].cpu(),
                    "base_cands": cands_bk[b_i],  # [K, 30, 6]
                }
            )
        if (start // batch_size) % 5 == 0 or start + batch_size >= total:
            print(
                f"  Cached {min(start + batch_size, total)}/{total} anchors ({time.time() - t0:.1f}s)",
                flush=True,
            )
    return records


@torch.no_grad()
def evaluate_expo_on_records(
    expo: ExpoResidualCritic,
    val_records: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    expo.eval()
    base_sq = torch.zeros(30, 6, dtype=torch.float64)
    expo_sq = torch.zeros(30, 6, dtype=torch.float64)
    valid_counts = torch.zeros(30, dtype=torch.float64)

    for rec in val_records:
        valid = (~rec["action_is_pad"]).to(dtype=torch.float64)  # [30]
        gt = rec["gt_action"].to(dtype=torch.float64)  # [30, 6]
        base0 = rec["base_cands"][0].to(dtype=torch.float64)  # [30, 6]

        cands_k = rec["base_cands"].to(device=device)  # [K, 30, 6]
        k_num = cands_k.shape[0]
        ctx_k = rec["ctx"].unsqueeze(0).to(device=device).expand(k_num, -1, -1)
        init_k = rec["init_ctx"].unsqueeze(0).to(device=device).expand(k_num, -1, -1)
        state_k = rec["state"].unsqueeze(0).to(device=device).expand(k_num, -1)
        t_k = rec["t_norm"].unsqueeze(0).to(device=device).expand(k_num, -1)

        edited_k, ctx_feat = expo.forward_edit_with_feat(ctx_k, state_k, t_k, cands_k, init_ctx=init_k)
        q_scores = expo.forward_q_from_feat(
            ctx_feat, state_k, t_k, edited_k, edited_k[:1].expand(k_num, -1, -1)
        )
        best_idx = int((q_scores - q_scores.mean()).argmax().item())
        pred_expo = edited_k[best_idx].cpu().to(dtype=torch.float64)

        base_sq += (base0 - gt).square() * valid.unsqueeze(-1)
        expo_sq += (pred_expo - gt).square() * valid.unsqueeze(-1)
        valid_counts += valid

    def _metrics(sq: torch.Tensor) -> dict[str, float]:
        tot_tokens = float(valid_counts.sum().item())
        full30 = math.sqrt(float(sq.sum().item()) / max(tot_tokens * 6.0, 1.0))
        h1 = math.sqrt(float(sq[0].sum().item()) / max(float(valid_counts[0].item()) * 6.0, 1.0))
        f5_list = [
            math.sqrt(float(sq[h].sum().item()) / max(float(valid_counts[h].item()) * 6.0, 1.0))
            for h in range(5)
        ]
        f5 = sum(f5_list) / 5.0
        arm = math.sqrt(float(sq[:, :5].sum().item()) / max(tot_tokens * 5.0, 1.0))
        grip = math.sqrt(float(sq[:, 5].sum().item()) / max(tot_tokens, 1.0))
        return {"full30": full30, "h1": h1, "f5": f5, "arm": arm, "grip": grip}

    m_base = _metrics(base_sq)
    m_expo = _metrics(expo_sq)
    return {
        "base_full30": m_base["full30"],
        "base_h1": m_base["h1"],
        "base_f5": m_base["f5"],
        "expo_full30": m_expo["full30"],
        "expo_h1": m_expo["h1"],
        "expo_f5": m_expo["f5"],
        "expo_arm": m_expo["arm"],
        "expo_grip": m_expo["grip"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path("outputs/train/sort-cubes-cosmos3-edge-optimized-bs64"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/sparse_seq_rl/expo_real_warmstart_bs64.pt"),
    )
    parser.add_argument("--dataset-repo-id", type=str, default="Orellius/so101_sort_cubes_no_top")
    parser.add_argument("--dataset-revision", type=str, default="a7c156de0e84a81e19457e6d783e4316e9f901e1")
    parser.add_argument("--train-stride", type=int, default=15)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"=== Option 4: Real-Data EXPO Residual Warm-Start ({args.base_checkpoint}) ===")
    extractor, decoder, _ = load_base_vam_components(args.base_checkpoint, device)

    meta = LeRobotDatasetMetadata(args.dataset_repo_id, revision=args.dataset_revision)
    contract = VideoVAMDatasetConfig(
        repo_id=args.dataset_repo_id,
        revision=args.dataset_revision,
        fps=meta.fps,
        total_episodes=meta.total_episodes,
        total_frames=meta.total_frames,
        task="Grab the purple cube and put it into the cardboard box, ignore the other objects",
    )

    train_ds = OnlineVideoDataset(
        repo_id=args.dataset_repo_id,
        episodes=tuple(range(0, 68)),
        stride=args.train_stride,
        contract=contract,
    )
    val_ds = OnlineVideoDataset(
        repo_id=args.dataset_repo_id,
        episodes=tuple(range(68, 77)),
        stride=20,
        contract=contract,
    )

    print(f"Caching train anchors ({len(train_ds)}) and val anchors ({len(val_ds)})...")
    train_records = build_cached_anchor_split(train_ds, extractor, decoder, device, k_candidates=6)
    val_records = build_cached_anchor_split(val_ds, extractor, decoder, device, k_candidates=6)

    # Free heavy backbone VRAM now that all train/val anchors and K=6 base proposals are cached
    del extractor
    del decoder
    torch.cuda.empty_cache()

    # Initialize EXPO Residual Actor + Concave Critic in standard residual mode around base actions
    # (`base_alpha=1.0` preserves 100% of the strong base trajectory at step 0, adding bounded `+-12 deg` edits)
    expo = ExpoResidualCritic(
        ctx_dim=2048,
        state_dim=6,
        chunk_len=30,
        use_init_anchor=True,
        base_alpha=1.0,
        curr_scale=1.0,
        max_delta_deg=12.0,
    ).to(device)

    # Initialize zero-edit output layer so step 0 starts at exact base policy performance
    nn.init.zeros_(expo.edit_net[-1].weight)
    nn.init.zeros_(expo.edit_net[-1].bias)

    all_inits = torch.stack([r["init_ctx"] for r in train_records], dim=0).to(device)
    expo.init_mean.copy_(all_inits.mean(dim=0, keepdim=True))
    expo.init_std.copy_(all_inits.std(dim=0, keepdim=True).clamp(min=1e-4))

    init_eval = evaluate_expo_on_records(expo, val_records, device)
    print(
        f"[Step 0] Base Full-30: {init_eval['base_full30']:.3f}° (H1: {init_eval['base_h1']:.3f}°, F5: {init_eval['base_f5']:.3f}°) | "
        f"EXPO Full-30: {init_eval['expo_full30']:.3f}° (H1: {init_eval['expo_h1']:.3f}°, F5: {init_eval['expo_f5']:.3f}°)",
        flush=True,
    )

    optimizer = torch.optim.AdamW(expo.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)

    # Stack tensors for fast GPU mini-batching
    ctx_t = torch.stack([r["ctx"] for r in train_records]).to(device)
    init_t = torch.stack([r["init_ctx"] for r in train_records]).to(device)
    state_t = torch.stack([r["state"] for r in train_records]).to(device)
    tnorm_t = torch.stack([r["t_norm"] for r in train_records]).to(device)
    gt_t = torch.stack([r["gt_action"] for r in train_records]).to(device)
    pad_t = torch.stack([r["action_is_pad"] for r in train_records]).to(device)
    cands_t = torch.stack([r["base_cands"] for r in train_records]).to(device)  # [N, K, 30, 6]
    sampler_mean_t = cands_t.mean(dim=1)  # [N, 30, 6]

    # Per-candidate empirical quality reward in [0, 1] relative to ground-truth trajectory
    valid_mask = (~pad_t).unsqueeze(1).unsqueeze(-1).float()  # [N, 1, 30, 1]
    cand_rmse = ((cands_t - gt_t.unsqueeze(1)).square() * valid_mask).mean(dim=(2, 3)).sqrt()  # [N, K]
    cand_rewards = torch.exp(-cand_rmse / 12.0)  # [N, K] in (0, 1]

    best_f5 = init_eval["expo_f5"]
    best_full30 = init_eval["expo_full30"]
    best_state_dict = {k: v.detach().cpu().clone() for k, v in expo.state_dict().items()}
    best_metrics = init_eval

    n_samples = ctx_t.shape[0]
    t_start = time.time()
    for step in range(1, args.steps + 1):
        expo.train()
        idx = torch.randint(0, n_samples, (args.batch_size,), device=device)
        k_idx = torch.randint(0, cands_t.shape[1], (args.batch_size,), device=device)

        b_ctx = ctx_t[idx]
        b_init = init_t[idx]
        b_state = state_t[idx]
        b_tnorm = tnorm_t[idx]
        b_gt = gt_t[idx]
        b_pad = pad_t[idx]
        b_base = cands_t[idx, k_idx]
        b_smean = sampler_mean_t[idx]

        pred_edit, ctx_feat = expo.forward_edit_with_feat(b_ctx, b_state, b_tnorm, b_base, init_ctx=b_init)

        # Group-centered Dr. GRPO + Marek Score-Centered residual actor loss
        # For each anchor group, compare positive expert execution (R=1.0) against the K candidate proposals
        group_rewards = torch.cat([torch.ones(args.batch_size, 1, device=device), cand_rewards[idx]], dim=1)
        group_mean = group_rewards.mean(dim=1)
        pos_adv = (1.0 - group_mean).clamp(min=0.05)

        valid_f = (~b_pad).unsqueeze(-1).float()
        actor_sc_loss = compute_marek_score_centered_loss(
            pred_edit * valid_f,
            b_gt * valid_f,
            b_smean * valid_f,
            pos_adv,
            sigma_deg=30.0,
            trust_beta=1.0,
        )
        # Emphasize First-5 accuracy (`t = 0..0.4s`) for crisp physical grasp entry
        f5_l1 = ((pred_edit[:, :5] - b_gt[:, :5]).abs() * valid_f[:, :5]).mean() / 30.0

        # Critic training on expert (`Q -> 1.0`), best candidate, and hard-negative perturbed actions (`Q -> exp(-err/12)`)
        ctx_det = ctx_feat.detach()
        ref_det = pred_edit.detach()
        q_gt = expo.forward_q_from_feat(ctx_det, b_state, b_tnorm, b_gt, ref_det)
        q_base = expo.forward_q_from_feat(ctx_det, b_state, b_tnorm, b_base, ref_det)
        r_base = cand_rewards[idx, k_idx]
        noise_neg = b_base + torch.randn_like(b_base) * 10.0
        r_neg = torch.exp(-(((noise_neg - b_gt).square() * valid_f).mean(dim=(1, 2)).sqrt()) / 12.0)
        q_neg = expo.forward_q_from_feat(ctx_det, b_state, b_tnorm, noise_neg, ref_det)

        critic_loss = (
            F.mse_loss(q_gt, torch.ones_like(q_gt)) + F.mse_loss(q_base, r_base) + F.mse_loss(q_neg, r_neg)
        )

        loss = actor_sc_loss + 1.5 * f5_l1 + 0.5 * critic_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(expo.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        if step % 300 == 0 or step == args.steps:
            ev = evaluate_expo_on_records(expo, val_records, device)
            is_best = (ev["expo_f5"] < best_f5) or (
                math.isclose(ev["expo_f5"], best_f5, abs_tol=0.02) and ev["expo_full30"] < best_full30
            )
            if is_best:
                best_f5 = ev["expo_f5"]
                best_full30 = ev["expo_full30"]
                best_state_dict = {k: v.detach().cpu().clone() for k, v in expo.state_dict().items()}
                best_metrics = ev
            mark = " *** BEST ***" if is_best else ""
            print(
                f"[Step {step}/{args.steps}] Loss: {loss.item():.4f} | "
                f"EXPO Full-30: {ev['expo_full30']:.3f}° (vs {ev['base_full30']:.3f}°), "
                f"H1: {ev['expo_h1']:.3f}° (vs {ev['base_h1']:.3f}°), "
                f"F5: {ev['expo_f5']:.3f}° (vs {ev['base_f5']:.3f}°){mark}",
                flush=True,
            )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state_dict,
            "config": {
                "ctx_dim": 2048,
                "state_dim": 6,
                "chunk_len": 30,
                "use_init_anchor": True,
                "base_alpha": 1.0,
                "curr_scale": 1.0,
                "max_delta_deg": 12.0,
            },
            "base_checkpoint": str(args.base_checkpoint),
            "val_metrics": best_metrics,
        },
        args.output_path,
    )
    summary_path = args.output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(best_metrics, indent=2) + "\n")
    print(
        f"Saved best real-data warm-started EXPO checkpoint to {args.output_path} "
        f"(Elapsed: {time.time() - t_start:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    import torch.nn as nn

    raise SystemExit(main())
