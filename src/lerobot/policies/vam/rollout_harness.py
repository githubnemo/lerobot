#!/usr/bin/env python3
"""Unified Rollout Engine for Video Action Model backbones.

Supports full-episode autoregressive rollouts for:
- cosmos3_edge (Cosmos 3 Edge)
- cosmos2b (Cosmos-Predict2-2B)
- cosmos7b (Cosmos 1.0 7B)
- cosmos14b (Cosmos 1.0 14B)

Automatically compares Base Model vs. Video-LoRA Model vs. Ground Truth on two
validation episodes (default: Episode 32 and Episode 35), renders 3-panel synchronized
MP4s (1920x540), and writes metrics summaries.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT
from lerobot.utils.io_utils import write_video


def compute_video_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Compute per-frame and aggregate MSE, PSNR, SSIM."""
    t, h, w, c = predicted.shape
    psnrs, mses, ssims = [], [], []

    for i in range(t):
        p = predicted[i].astype(np.float64)
        tgt = target[i].astype(np.float64)
        mse = np.mean((p - tgt) ** 2)
        mses.append(float(mse))
        if mse < 1e-10:
            psnr = 50.0
        else:
            psnr = 10.0 * np.log10((255.0**2) / mse)
        psnrs.append(float(psnr))

        mu_x = p.mean()
        mu_y = tgt.mean()
        sigma_x = np.var(p)
        sigma_y = np.var(tgt)
        sigma_xy = np.cov(p.flatten(), tgt.flatten())[0, 1]
        k1, k2, l = 0.01, 0.03, 255.0
        c1 = (k1 * l) ** 2
        c2 = (k2 * l) ** 2
        ssim = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / (
            (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
        )
        ssims.append(float(np.clip(ssim, 0.0, 1.0)))

    return {
        "mean_mse": float(np.mean(mses)),
        "mean_psnr_db": float(np.mean(psnrs)),
        "mean_ssim": float(np.mean(ssims)),
        "per_frame_psnr": psnrs,
        "per_frame_ssim": ssims,
        "per_frame_mse": mses,
    }


def assemble_3panel_video(
    base_frames: np.ndarray,
    lora_frames: np.ndarray,
    gt_frames: np.ndarray,
    base_metrics: dict[str, Any],
    lora_metrics: dict[str, Any],
    out_path: Path,
    backbone_name: str,
    episode: int,
    fps: int = 10,
) -> None:
    """Create a 3-panel 1920x540 side-by-side synchronized video."""
    num_frames, h, w, c = gt_frames.shape
    banner_h = 60
    total_w = 3 * w
    total_h = h + banner_h

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = font_sm = ImageFont.load_default()

    composite_frames = []
    for i in range(num_frames):
        canvas = Image.new("RGB", (total_w, total_h), color="#0f172a")
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([(0, 0), (total_w, banner_h)], fill="#1e293b")

        # Column 1: Base
        draw.text((20, 10), f"Base {backbone_name}", fill="#94a3b8", font=font)
        draw.text(
            (20, 34),
            f"PSNR: {base_metrics['per_frame_psnr'][i]:.1f} dB  |  SSIM: {base_metrics['per_frame_ssim'][i]:.3f}",
            fill="#64748b",
            font=font_sm,
        )

        # Column 2: Video-LoRA
        draw.text((w + 20, 10), f"Video-LoRA {backbone_name} (Ours)", fill="#38bdf8", font=font)
        draw.text(
            (w + 20, 34),
            f"PSNR: {lora_metrics['per_frame_psnr'][i]:.1f} dB  |  SSIM: {lora_metrics['per_frame_ssim'][i]:.3f}",
            fill="#0284c7",
            font=font_sm,
        )

        # Column 3: Ground Truth
        draw.text((2 * w + 20, 10), f"Ground Truth (Episode {episode})", fill="#4ade80", font=font)
        sec = i / fps
        draw.text(
            (2 * w + 20, 34),
            f"Frame {i + 1}/{num_frames}  ({sec:.1f}s / {num_frames / fps:.1f}s)",
            fill="#22c55e",
            font=font_sm,
        )

        canvas.paste(Image.fromarray(base_frames[i]), (0, banner_h))
        canvas.paste(Image.fromarray(lora_frames[i]), (w, banner_h))
        canvas.paste(Image.fromarray(gt_frames[i]), (2 * w, banner_h))

        draw.line([(w, 0), (w, total_h)], fill="#334155", width=2)
        draw.line([(2 * w, 0), (2 * w, total_h)], fill="#334155", width=2)

        composite_frames.append(np.array(canvas))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(out_path, composite_frames, fps)
    print(
        f"Saved synchronized 3-panel video to: {out_path} ({total_w}x{total_h}, {len(composite_frames)} frames @ {fps} fps)"
    )


def rollout_cosmos3_edge(
    checkpoint_dir: Path,
    lora_path: Path | None,
    init_pil_frames: list[Image.Image],
    total_frames: int,
    prompt: str = "take cube out of box",
    steps: int = 25,
    seed: int = 42,
    device: torch.device = torch.device("cuda"),
) -> np.ndarray:
    """Run autoregressive rollout on Cosmos 3 Edge."""
    from diffusers import Cosmos3OmniPipeline

    from lerobot.policies.vam.cosmos3_lora import load_cosmos3_lora

    pipe = Cosmos3OmniPipeline.from_pretrained(str(checkpoint_dir), dtype=torch.bfloat16).to(device)
    if lora_path is not None and Path(lora_path).is_file():
        load_cosmos3_lora(pipe.transformer, str(lora_path))
        pipe.transformer.eval()

    history_pil = list(init_pil_frames[:5])
    history_np = [np.array(img) for img in history_pil]
    chunk_idx = 0

    while len(history_np) < total_frames:
        chunk_idx += 1
        condition_frames = history_pil[-5:]
        generator = torch.Generator(device=device).manual_seed(seed + chunk_idx)
        with torch.inference_mode():
            out = pipe(
                prompt=prompt,
                video=condition_frames,
                condition_frame_indexes_vision=(0, 1),
                num_frames=17,
                height=480,
                width=640,
                fps=10.0,
                num_inference_steps=steps,
                generator=generator,
                output_type="np",
            )
        video = out.video
        if video.ndim == 5:
            video = video[0]
        if video.max() <= 1.0:
            video = (video * 255).round().astype(np.uint8)
        else:
            video = video.astype(np.uint8)

        new_frames = video[5:]
        for f in new_frames:
            history_np.append(f)
            history_pil.append(Image.fromarray(f))
            if len(history_np) >= total_frames:
                break

    del pipe
    torch.cuda.empty_cache()
    return np.stack(history_np[:total_frames], axis=0)


def rollout_cosmos2b(
    checkpoint_path: Path,
    tokenizer_path: Path,
    prompt_path: Path,
    lora_path: Path | None,
    init_rgb_uint8: list[np.ndarray],
    total_frames: int,
    steps: int = 25,
    seed: int = 42,
    device: torch.device = torch.device("cuda"),
) -> np.ndarray:
    """Run autoregressive rollout on Cosmos 2B."""
    from lerobot.policies.vam.cosmos_lora import merge_lora_file_into_base
    from lerobot.policies.vam.cosmos_prompt_embedding import load_prompt_embedding
    from lerobot.policies.vam.cosmos_video_prediction import CosmosVideo2WorldBackend, cosmos_2b_backbone_spec

    spec = cosmos_2b_backbone_spec(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        device=str(device),
        sampling_steps=steps,
    )
    backend = CosmosVideo2WorldBackend(spec, seed=seed)
    if lora_path is not None and Path(lora_path).is_file():
        merge_lora_file_into_base(backend.extractor.backbone, lora_path)

    prompt_emb = load_prompt_embedding(prompt_path).embedding.to(device)
    history_np = [f for f in init_rgb_uint8[:5]]
    chunk_idx = 0

    while len(history_np) < total_frames:
        chunk_idx += 1
        cond_5_np = history_np[-5:]
        # [5, 480, 640, 3] -> [1, 3, 5, 480, 640]
        cond_t = torch.from_numpy(np.stack(cond_5_np, axis=0)).permute(3, 0, 1, 2).unsqueeze(0).to(device)
        cond = backend.prepare_conditioning(cond_t, prompt_emb)
        with torch.inference_mode():
            decoded = backend.rollout(cond, seed=seed + chunk_idx)  # [1, 3, T, 480, 640]
        # decoded is in [-1, 1]
        decoded_np = (
            ((decoded[0].clamp(-1, 1) + 1.0) * 127.5)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(1, 2, 3, 0)
            .cpu()
            .numpy()
        )  # [T, 480, 640, 3]

        new_frames = decoded_np[5:]
        for f in new_frames:
            history_np.append(f)
            if len(history_np) >= total_frames:
                break

    del backend
    torch.cuda.empty_cache()
    return np.stack(history_np[:total_frames], axis=0)


def run_full_episode_rollout(
    backbone: str,
    lora_checkpoint: str | Path | None,
    episodes: Sequence[int] = (32, 35),
    output_dir: str | Path = Path("/home/anton/lerobot-video-vam/outputs/evaluation"),
    dataset_root: str | Path = Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset"),
    dataset_repo_id: str = "hubnemo/cube_out_of_box_dataset",
    steps: int = 25,
    seed: int = 42,
    device: str = "cuda",
) -> dict[int, dict[str, Any]]:
    """Execute full-episode autoregressive rollouts across specified validation episodes."""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)

    print("=" * 80)
    print(f"RUNNING FULL-EPISODE AUTOREGRESSIVE ROLLOUT FOR BACKBONE: {backbone.upper()}")
    print(f"Episodes: {list(episodes)} | Steps per chunk: {steps} | LoRA: {lora_checkpoint}")
    print("=" * 80, flush=True)

    dataset = LeRobotDataset(
        dataset_repo_id,
        root=str(dataset_root),
        revision=CUBE_OUT_OF_BOX_CONTRACT.revision,
        return_uint8=True,
        download_videos=False,
    )

    from scripts.video_vam.smoke_test_cosmos_extractor import _relative_index

    results: dict[int, dict[str, Any]] = {}

    for ep in episodes:
        print(f"\n>>> PROCESSING EPISODE {ep}...", flush=True)
        ep_meta = dataset.meta.episodes[ep]
        start_f = int(ep_meta["dataset_from_index"])
        end_f = int(ep_meta["dataset_to_index"])
        total_frames = end_f - start_f

        gt_np_list = []
        gt_pil_list = []
        for f in range(start_f, end_f):
            item = dataset[_relative_index(dataset, f)]
            img_t = item.get("observation.images.front", item.get("observation.images.phone"))
            img_np = img_t.permute(1, 2, 0).cpu().numpy()
            gt_np_list.append(img_np)
            gt_pil_list.append(Image.fromarray(img_np))

        gt_array = np.stack(gt_np_list, axis=0)

        # 1. Base Rollout
        print(f"  [1/3] Generating Base {backbone} rollout ({total_frames} frames)...", flush=True)
        t0 = time.time()
        if backbone == "cosmos3_edge":
            base_pred = rollout_cosmos3_edge(
                checkpoint_dir=Path("/home/anton/.cache/video-vam/cosmos3-edge"),
                lora_path=None,
                init_pil_frames=gt_pil_list[:5],
                total_frames=total_frames,
                steps=steps,
                seed=seed,
                device=dev,
            )
        elif backbone == "cosmos2b":
            base_pred = rollout_cosmos2b(
                checkpoint_path=Path(
                    "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt"
                ),
                tokenizer_path=Path(
                    "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth"
                ),
                prompt_path=Path(
                    "/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-t5-11b.safetensors"
                ),
                lora_path=None,
                init_rgb_uint8=gt_np_list[:5],
                total_frames=total_frames,
                steps=steps,
                seed=seed,
                device=dev,
            )
        else:
            raise NotImplementedError(f"Backbone {backbone} rollout not implemented yet")

        print(f"  Base finished in {time.time() - t0:.1f}s", flush=True)

        # 2. LoRA Rollout
        print(f"  [2/3] Generating Video-LoRA {backbone} rollout ({total_frames} frames)...", flush=True)
        t0 = time.time()
        if backbone == "cosmos3_edge":
            lora_pred = rollout_cosmos3_edge(
                checkpoint_dir=Path("/home/anton/.cache/video-vam/cosmos3-edge"),
                lora_path=Path(lora_checkpoint) if lora_checkpoint else None,
                init_pil_frames=gt_pil_list[:5],
                total_frames=total_frames,
                steps=steps,
                seed=seed,
                device=dev,
            )
        elif backbone == "cosmos2b":
            lora_pred = rollout_cosmos2b(
                checkpoint_path=Path(
                    "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt"
                ),
                tokenizer_path=Path(
                    "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth"
                ),
                prompt_path=Path(
                    "/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-t5-11b.safetensors"
                ),
                lora_path=Path(lora_checkpoint) if lora_checkpoint else None,
                init_rgb_uint8=gt_np_list[:5],
                total_frames=total_frames,
                steps=steps,
                seed=seed,
                device=dev,
            )
        print(f"  LoRA finished in {time.time() - t0:.1f}s", flush=True)

        # 3. Metrics & Assembly
        print("  [3/3] Computing metrics & assembling 3-panel comparison video...", flush=True)
        base_metrics = compute_video_metrics(base_pred, gt_array)
        lora_metrics = compute_video_metrics(lora_pred, gt_array)

        print(f"\n  Episode {ep} Summary:")
        print(
            f"    Base MSE: {base_metrics['mean_mse']:.1f} | PSNR: {base_metrics['mean_psnr_db']:.2f} dB | SSIM: {base_metrics['mean_ssim']:.4f}"
        )
        print(
            f"    LoRA MSE: {lora_metrics['mean_mse']:.1f} | PSNR: {lora_metrics['mean_psnr_db']:.2f} dB | SSIM: {lora_metrics['mean_ssim']:.4f}"
        )

        video_name = f"{backbone}_ep{ep}_full_episode_side_by_side.mp4"
        assemble_3panel_video(
            base_pred,
            lora_pred,
            gt_array,
            base_metrics,
            lora_metrics,
            output_dir / video_name,
            backbone_name=backbone,
            episode=ep,
            fps=10,
        )

        results[ep] = {
            "total_frames": total_frames,
            "duration_s": total_frames / 10.0,
            "base": base_metrics,
            "lora": lora_metrics,
            "video_path": str(output_dir / video_name),
        }

    summary_path = output_dir / f"{backbone}_full_episodes_metrics.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nAll episode rollouts saved to {output_dir}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["cosmos3_edge", "cosmos2b"], required=True)
    parser.add_argument("--lora-checkpoint", type=Path, default=None)
    parser.add_argument("--episodes", type=int, nargs="+", default=[32, 35])
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/home/anton/lerobot-video-vam/outputs/evaluation")
    )
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset")
    )
    parser.add_argument("--dataset-repo-id", type=str, default="hubnemo/cube_out_of_box_dataset")
    args = parser.parse_args()

    run_full_episode_rollout(
        backbone=args.backbone,
        lora_checkpoint=args.lora_checkpoint,
        episodes=args.episodes,
        output_dir=args.output_dir,
        dataset_root=args.dataset_root,
        dataset_repo_id=args.dataset_repo_id,
        steps=args.steps,
    )
