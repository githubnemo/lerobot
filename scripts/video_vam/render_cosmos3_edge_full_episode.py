#!/usr/bin/env python3
"""Generate a complete episode (165 frames, ~16.5s @ 10 FPS) autoregressive video rollout:
Base Cosmos 3 Edge vs. Video-LoRA Cosmos 3 Edge vs. Ground Truth.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import Cosmos3OmniPipeline
from PIL import Image, ImageDraw, ImageFont

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT
from lerobot.policies.vam.cosmos3_lora import load_cosmos3_lora
from lerobot.utils.io_utils import write_video

DEFAULT_CHECKPOINT_DIR = Path("/home/anton/.cache/video-vam/cosmos3-edge")
DEFAULT_DATASET_ROOT = Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset")
DEFAULT_LORA_PATH = Path(
    "/home/anton/lerobot-video-vam/outputs/train/cosmos3-edge-video-lora/best_lora.safetensors"
)
DEFAULT_OUTPUT_DIR = Path("/home/anton/lerobot-video-vam/outputs/evaluation")


def compute_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
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


def autoregressive_rollout(
    pipe: Cosmos3OmniPipeline,
    init_pil_frames: list[Image.Image],
    total_frames: int,
    prompt: str = "take cube out of box",
    steps: int = 25,
    seed: int = 42,
    device: torch.device = torch.device("cuda"),
) -> np.ndarray:
    """Generate total_frames autoregressively in 17-frame chunks (5 conditioning -> 12 generated)."""
    # Each pass conditions on last 5 frames and generates 12 new frames
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

        # First 5 frames are condition; next 12 are freshly predicted
        new_frames = video[5:]
        for f in new_frames:
            history_np.append(f)
            history_pil.append(Image.fromarray(f))
            if len(history_np) >= total_frames:
                break

        print(f"  Chunk {chunk_idx}: Generated up to frame {len(history_np)}/{total_frames}...", flush=True)

    return np.stack(history_np[:total_frames], axis=0)


def assemble_full_comparison_video(
    base_frames: np.ndarray,
    lora_frames: np.ndarray,
    gt_frames: np.ndarray,
    base_metrics: dict[str, Any],
    lora_metrics: dict[str, Any],
    out_path: Path,
    fps: int = 10,
) -> None:
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

        # Titles & Dynamic Metrics
        draw.text((20, 10), "Base Cosmos 3 Edge (Autoregressive)", fill="#94a3b8", font=font)
        draw.text(
            (20, 34),
            f"PSNR: {base_metrics['per_frame_psnr'][i]:.1f} dB  |  SSIM: {base_metrics['per_frame_ssim'][i]:.3f}",
            fill="#64748b",
            font=font_sm,
        )

        draw.text((w + 20, 10), "Video-LoRA Cosmos 3 Edge (Ours)", fill="#38bdf8", font=font)
        draw.text(
            (w + 20, 34),
            f"PSNR: {lora_metrics['per_frame_psnr'][i]:.1f} dB  |  SSIM: {lora_metrics['per_frame_ssim'][i]:.3f}",
            fill="#0284c7",
            font=font_sm,
        )

        draw.text((2 * w + 20, 10), "Ground Truth (Validation Episode 32)", fill="#4ade80", font=font)
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
        f"\nSaved FULL EPISODE 3-panel video to: {out_path} ({total_w}x{total_h}, {len(composite_frames)} frames @ {fps} fps)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Cosmos 3 Edge Full Episode Rollout.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--lora-path", type=Path, default=DEFAULT_LORA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episode", type=int, default=32)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COSMOS 3 EDGE FULL EPISODE AUTOREGRESSIVE VIDEO ROLLOUT (165 FRAMES)")
    print("=" * 80)

    # 1. Load Episode 32 Demonstration Frames
    print("\n[1/4] Loading complete Episode 32 Ground Truth demonstration...")
    dataset = LeRobotDataset(
        CUBE_OUT_OF_BOX_CONTRACT.repo_id,
        root=str(args.dataset_root),
        revision=CUBE_OUT_OF_BOX_CONTRACT.revision,
        return_uint8=True,
        download_videos=False,
    )
    from scripts.video_vam.smoke_test_cosmos_extractor import _relative_index

    ep_meta = dataset.meta.episodes[args.episode]
    start_f = int(ep_meta["dataset_from_index"])
    end_f = int(ep_meta["dataset_to_index"])
    total_episode_frames = end_f - start_f
    print(
        f"Episode {args.episode}: frames {start_f}..{end_f} (total: {total_episode_frames} frames, {total_episode_frames / 10:.1f}s)"
    )

    gt_pil_list = []
    gt_np_list = []
    for f in range(start_f, end_f):
        item = dataset[_relative_index(dataset, f)]
        img_t = item.get("observation.images.front", item.get("observation.images.phone"))
        img_np = img_t.permute(1, 2, 0).cpu().numpy()
        gt_np_list.append(img_np)
        gt_pil_list.append(Image.fromarray(img_np))

    gt_array = np.stack(gt_np_list, axis=0)
    gt_video_path = output_dir / "cosmos3_edge_full_episode_ground_truth.mp4"
    write_video(gt_video_path, gt_np_list, 10)
    print(f"Saved complete Ground Truth video: {gt_video_path}")

    # 2. Base Cosmos 3 Autoregressive Rollout
    print("\n[2/4] Loading base Cosmos 3 Edge pipeline and running full-episode rollout...")
    pipe = Cosmos3OmniPipeline.from_pretrained(
        str(args.checkpoint_dir),
        dtype=torch.bfloat16,
    ).to(device)

    t0 = time.time()
    base_video = autoregressive_rollout(
        pipe,
        gt_pil_list[:5],
        total_frames=total_episode_frames,
        prompt="take cube out of box",
        steps=args.steps,
        seed=args.seed,
        device=device,
    )
    print(f"Base rollout finished in {time.time() - t0:.1f}s (shape: {base_video.shape})")
    base_video_path = output_dir / "cosmos3_edge_full_episode_base.mp4"
    write_video(base_video_path, list(base_video), 10)

    # 3. Video-LoRA Cosmos 3 Autoregressive Rollout
    print("\n[3/4] Injecting Video-LoRA and running full-episode rollout...")
    load_cosmos3_lora(pipe.transformer, str(args.lora_path))
    pipe.transformer.eval()

    t0 = time.time()
    lora_video = autoregressive_rollout(
        pipe,
        gt_pil_list[:5],
        total_frames=total_episode_frames,
        prompt="take cube out of box",
        steps=args.steps,
        seed=args.seed,
        device=device,
    )
    print(f"LoRA rollout finished in {time.time() - t0:.1f}s (shape: {lora_video.shape})")
    lora_video_path = output_dir / "cosmos3_edge_full_episode_lora.mp4"
    write_video(lora_video_path, list(lora_video), 10)

    # 4. Metrics & Side-by-Side Video
    print("\n[4/4] Computing full-episode metrics and assembling synchronized video...")
    base_metrics = compute_metrics(base_video, gt_array)
    lora_metrics = compute_metrics(lora_video, gt_array)

    print("\n" + "=" * 65)
    print("FULL EPISODE (165 FRAMES, 16.5s) BENCHMARK RESULTS")
    print("=" * 65)
    print(f"{'Metric':<20} | {'Base Cosmos 3':<18} | {'Video-LoRA Cosmos 3':<18}")
    print("-" * 65)
    print(
        f"{'Mean PSNR (dB)':<20} | {base_metrics['mean_psnr_db']:<18.2f} | {lora_metrics['mean_psnr_db']:<18.2f}"
    )
    print(f"{'Mean SSIM':<20} | {base_metrics['mean_ssim']:<18.4f} | {lora_metrics['mean_ssim']:<18.4f}")
    print(f"{'Mean MSE':<20} | {base_metrics['mean_mse']:<18.2f} | {lora_metrics['mean_mse']:<18.2f}")
    print("=" * 65)

    metrics_path = output_dir / "cosmos3_edge_full_episode_metrics.json"
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "episode": args.episode,
        "total_frames": total_episode_frames,
        "duration_seconds": total_episode_frames / 10.0,
        "steps_per_chunk": args.steps,
        "base": base_metrics,
        "lora": lora_metrics,
    }
    metrics_path.write_text(json.dumps(summary, indent=2))

    comparison_path = output_dir / "cosmos3_edge_full_episode_side_by_side.mp4"
    assemble_full_comparison_video(
        base_video,
        lora_video,
        gt_array,
        base_metrics,
        lora_metrics,
        comparison_path,
        fps=10,
    )

    print("\nFull episode rollout rendering complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
