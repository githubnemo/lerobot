#!/usr/bin/env python3
"""Render 3-panel synchronized comparison video for Cosmos 3 Edge:
Base Cosmos 3 vs. Video-LoRA Cosmos 3 vs. Ground Truth.
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


def assemble_comparison_video(
    base_frames: np.ndarray,
    lora_frames: np.ndarray,
    gt_frames: np.ndarray,
    base_metrics: dict[str, Any],
    lora_metrics: dict[str, Any],
    out_path: Path,
    fps: int = 10,
) -> None:
    """Create a 3-panel 1920x540 side-by-side video with headers and frame overlays."""
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

        # Header background banner
        draw.rectangle([(0, 0), (total_w, banner_h)], fill="#1e293b")

        # Titles
        draw.text((20, 10), "Base Cosmos 3 Edge", fill="#94a3b8", font=font)
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

        draw.text((2 * w + 20, 10), "Ground Truth (Validation Ep 32)", fill="#4ade80", font=font)
        draw.text((2 * w + 20, 34), f"Frame {i + 1}/17  (10 FPS)", fill="#22c55e", font=font_sm)

        # Paste images
        img_base = Image.fromarray(base_frames[i])
        img_lora = Image.fromarray(lora_frames[i])
        img_gt = Image.fromarray(gt_frames[i])

        canvas.paste(img_base, (0, banner_h))
        canvas.paste(img_lora, (w, banner_h))
        canvas.paste(img_gt, (2 * w, banner_h))

        # Vertical borders
        draw.line([(w, 0), (w, total_h)], fill="#334155", width=2)
        draw.line([(2 * w, 0), (2 * w, total_h)], fill="#334155", width=2)

        composite_frames.append(np.array(canvas))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(out_path, composite_frames, fps)
    print(
        f"\nSaved synchronized 3-panel video to: {out_path} ({total_w}x{total_h}, {len(composite_frames)} frames)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Cosmos 3 Edge Base vs LoRA comparison.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--lora-path", type=Path, default=DEFAULT_LORA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episode", type=int, default=32)
    parser.add_argument("--start-frame", type=int, default=4820)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COSMOS 3 EDGE 3-WAY SYNCHRONIZED ROLLOUT RENDERING")
    print("=" * 80)
    print(f"Dataset Root: {args.dataset_root}")
    print(f"LoRA Path:    {args.lora_path}")
    print(f"Output Dir:   {output_dir}")

    # 1. Load Ground Truth Frames
    print("\n[1/4] Loading Ground Truth demonstration frames...")
    dataset = LeRobotDataset(
        CUBE_OUT_OF_BOX_CONTRACT.repo_id,
        root=str(args.dataset_root),
        revision=CUBE_OUT_OF_BOX_CONTRACT.revision,
        return_uint8=True,
        download_videos=False,
    )

    from scripts.video_vam.smoke_test_cosmos_extractor import _relative_index

    gt_pil_list = []
    gt_np_list = []
    for f in range(args.start_frame, args.start_frame + args.num_frames):
        item = dataset[_relative_index(dataset, f)]
        img_t = item.get("observation.images.front", item.get("observation.images.phone"))
        img_np = img_t.permute(1, 2, 0).cpu().numpy()
        gt_np_list.append(img_np)
        gt_pil_list.append(Image.fromarray(img_np))

    gt_array = np.stack(gt_np_list, axis=0)  # [17, 480, 640, 3]
    print(f"Loaded {len(gt_pil_list)} frames (shape: {gt_array.shape})")

    gt_video_path = output_dir / "cosmos3_edge_ground_truth.mp4"
    write_video(gt_video_path, list(gt_array), 10)
    print(f"Saved Ground Truth video: {gt_video_path}")

    # 2. Load Pipeline and Run Base Rollout
    print("\n[2/4] Loading base Cosmos 3 Edge pipeline and generating base rollout...")
    pipe = Cosmos3OmniPipeline.from_pretrained(
        str(args.checkpoint_dir),
        dtype=torch.bfloat16,
    ).to(device)

    condition_frames = gt_pil_list[:5]
    prompt_text = "take cube out of box"

    generator = torch.Generator(device=device).manual_seed(args.seed)
    t0 = time.perf_counter()
    with torch.inference_mode():
        base_output = pipe(
            prompt=prompt_text,
            video=condition_frames,
            condition_frame_indexes_vision=(0, 1),
            num_frames=args.num_frames,
            height=480,
            width=640,
            fps=10.0,
            num_inference_steps=args.steps,
            generator=generator,
            output_type="np",
        )
    print(f"Base generation finished in {time.perf_counter() - t0:.1f}s")

    base_video = base_output.video
    if base_video.ndim == 5:
        base_video = base_video[0]
    if base_video.max() <= 1.0:
        base_video = (base_video * 255).round().astype(np.uint8)
    else:
        base_video = base_video.astype(np.uint8)

    base_video_path = output_dir / "cosmos3_edge_base_prediction.mp4"
    write_video(base_video_path, list(base_video), 10)
    print(f"Saved Base Cosmos 3 prediction video: {base_video_path}")

    # 3. Inject and Load Trained Video-LoRA
    print("\n[3/4] Injecting and loading Video-LoRA...")
    load_cosmos3_lora(pipe.transformer, str(args.lora_path))
    pipe.transformer.eval()

    generator_lora = torch.Generator(device=device).manual_seed(args.seed)
    t0 = time.perf_counter()
    with torch.inference_mode():
        lora_output = pipe(
            prompt=prompt_text,
            video=condition_frames,
            condition_frame_indexes_vision=(0, 1),
            num_frames=args.num_frames,
            height=480,
            width=640,
            fps=10.0,
            num_inference_steps=args.steps,
            generator=generator_lora,
            output_type="np",
        )
    print(f"LoRA generation finished in {time.perf_counter() - t0:.1f}s")

    lora_video = lora_output.video
    if lora_video.ndim == 5:
        lora_video = lora_video[0]
    if lora_video.max() <= 1.0:
        lora_video = (lora_video * 255).round().astype(np.uint8)
    else:
        lora_video = lora_video.astype(np.uint8)

    lora_video_path = output_dir / "cosmos3_edge_lora_prediction.mp4"
    write_video(lora_video_path, list(lora_video), 10)
    print(f"Saved LoRA Cosmos 3 prediction video: {lora_video_path}")

    # 4. Compute Metrics & Render 3-Way Video
    print("\n[4/4] Computing visual quality metrics & rendering 3-panel comparison...")
    base_metrics = compute_metrics(base_video, gt_array)
    lora_metrics = compute_metrics(lora_video, gt_array)

    print("\n" + "=" * 65)
    print(f"{'Metric':<20} | {'Base Cosmos 3':<18} | {'Video-LoRA Cosmos 3':<18}")
    print("-" * 65)
    print(
        f"{'PSNR (dB)':<20} | {base_metrics['mean_psnr_db']:<18.2f} | {lora_metrics['mean_psnr_db']:<18.2f}"
    )
    print(f"{'SSIM':<20} | {base_metrics['mean_ssim']:<18.4f} | {lora_metrics['mean_ssim']:<18.4f}")
    print(f"{'MSE':<20} | {base_metrics['mean_mse']:<18.2f} | {lora_metrics['mean_mse']:<18.2f}")
    print("=" * 65)

    metrics_path = output_dir / "cosmos3_edge_rollout_metrics.json"
    metrics_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "episode": args.episode,
        "start_frame": args.start_frame,
        "num_frames": args.num_frames,
        "steps": args.steps,
        "seed": args.seed,
        "base": base_metrics,
        "lora": lora_metrics,
    }
    metrics_path.write_text(json.dumps(metrics_summary, indent=2))
    print(f"Saved metrics summary to: {metrics_path}")

    side_by_side_path = output_dir / "cosmos3_edge_side_by_side_comparison.mp4"
    assemble_comparison_video(
        base_video,
        lora_video,
        gt_array,
        base_metrics,
        lora_metrics,
        side_by_side_path,
        fps=10,
    )

    print("\nCosmos 3 Edge video rendering complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
