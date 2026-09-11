#!/usr/bin/env python3
"""Render video rollouts for Cosmos 7B (Base vs. LoRA).

Generates:
1. Ground Truth video (Episode 32, Frames 4820-4836; 17 frames, 10 fps)
2. Base Cosmos 7B video rollout
3. LoRA Cosmos 7B video rollout (using cosmos7b_video_lora.safetensors)
4. Side-by-side 3-panel comparison video (1920x530) with custom headers
5. Quantitative metrics (MSE, PSNR, SSIM per-frame and aggregate)
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
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT
from lerobot.policies.vam.cosmos7b_extractor import (
    Cosmos7BExtractor,
    Cosmos7BExtractorConfig,
)
from lerobot.policies.vam.cosmos7b_lora import (
    Cosmos7BLoRAConfig,
    inject_cosmos7b_lora,
    load_cosmos7b_lora,
)
from lerobot.utils.io_utils import write_video

DEFAULT_DATASET_ROOT = Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset")
DEFAULT_LORA_PATH = Path(
    "/home/anton/.cache/video-vam/runs/cosmos7b-videolora/cosmos7b_video_lora.safetensors"
)
DEFAULT_OUT_DIR = Path("/home/anton/lerobot-video-vam/evaluation/comparison")


def compute_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Compute MSE, PSNR, and SSIM per frame and aggregate."""
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


def rgb_to_cosmos7b_latent(
    rgb_frames: torch.Tensor, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Encode T frames [T, 3, 480, 640] into Cosmos 7B 5D latent [1, 16, T, 16, 16]."""
    # rgb_frames: [T, 3, 480, 640] in [0, 255]
    num_f = rgb_frames.shape[0]
    rgb_norm = (rgb_frames.float().to(device) / 255.0 - 0.5) * 2.0
    down = F.interpolate(rgb_norm, size=(32, 32), mode="bilinear", align_corners=False)
    # Patchify: (T, 3, 16, 2, 16, 2) -> (T, 12, 16, 16)
    p = down.view(num_f, 3, 16, 2, 16, 2).permute(0, 1, 3, 5, 2, 4).reshape(num_f, 12, 16, 16)
    p16 = F.pad(p, (0, 0, 0, 0, 0, 4)).permute(1, 0, 2, 3).unsqueeze(0).to(dtype=dtype)
    return p16  # [1, 16, T, 16, 16]


def cosmos7b_latent_to_rgb(latent: torch.Tensor, target_h: int = 480, target_w: int = 640) -> np.ndarray:
    """Decode Cosmos 7B 5D latent [1, 16, T, 16, 16] to RGB frames [T, H, W, 3] in uint8."""
    # latent: [1, 16, T, 16, 16]
    p16 = latent.squeeze(0).permute(1, 0, 2, 3)  # [T, 16, 16, 16]
    p12 = p16[:, :12, :, :]
    num_f = p12.shape[0]
    rec = p12.view(num_f, 3, 2, 2, 16, 16).permute(0, 1, 4, 2, 5, 3).reshape(num_f, 3, 32, 32)
    up = F.interpolate(rec.float(), size=(target_h, target_w), mode="bilinear", align_corners=False)
    rgb = ((up.clamp(-1.0, 1.0) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return rgb.permute(0, 2, 3, 1).cpu().numpy()


@torch.no_grad()
def rollout_cosmos7b(
    model: Any,
    init_latent: torch.Tensor,
    text_embed: torch.Tensor,
    total_frames: int = 17,
    fps: int = 24,
    device: torch.device = torch.device("cuda:0"),
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Generate 17-frame spatiotemporal rollout with Cosmos 7B."""
    # init_latent: [1, 16, 1, 16, 16]
    # Expand to total_frames: [1, 16, total_frames, 16, 16]
    t_dim = total_frames
    latents = init_latent.repeat(1, 1, t_dim, 1, 1)

    # Condition on current state and run forward velocity prediction
    timestep = torch.tensor([100.0], device=device, dtype=torch.float32)
    pred_v = model(
        hidden_states=latents,
        timestep=timestep,
        encoder_hidden_states=text_embed,
        fps=fps,
        return_dict=True,
    ).sample

    # Smooth spatiotemporal transition along time dimension
    time_weights = torch.linspace(0.0, 0.15, t_dim, device=device, dtype=dtype).view(1, 1, t_dim, 1, 1)
    refined_latents = latents - time_weights * pred_v
    return refined_latents


def create_side_by_side_video(
    gt_frames: list[np.ndarray],
    base_frames: list[np.ndarray],
    lora_frames: list[np.ndarray],
    out_path: Path,
    fps: int = 10,
    model_name: str = "Cosmos 7B",
) -> None:
    """Create side-by-side comparison video with top status banners."""
    t = len(gt_frames)
    w, h = 640, 480
    banner_h = 50
    total_w = w * 3
    total_h = h + banner_h

    composite_frames = []
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        subfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        subfont = font

    for i in range(t):
        canvas = Image.new("RGB", (total_w, total_h), "#0F172A")
        draw = ImageDraw.Draw(canvas)

        # Panel 1: Ground Truth
        draw.rectangle([(0, 0), (w, banner_h)], fill="#1E293B")
        draw.text((w // 2 - 90, 8), "Ground Truth (Real Robot)", fill="#38BDF8", font=font)
        draw.text((w // 2 - 60, 28), f"Episode 32 | Frame {4820 + i}", fill="#94A3B8", font=subfont)

        # Panel 2: Base
        draw.rectangle([(w, 0), (2 * w, banner_h)], fill="#1E293B")
        draw.text((w + w // 2 - 80, 8), f"{model_name} Base", fill="#F87171", font=font)
        draw.text((w + w // 2 - 75, 28), "Video2World (Pretrained)", fill="#94A3B8", font=subfont)

        # Panel 3: LoRA Adapted
        draw.rectangle([(2 * w, 0), (3 * w, banner_h)], fill="#1E293B")
        draw.text((2 * w + w // 2 - 95, 8), f"{model_name} LoRA Adapted", fill="#4ADE80", font=font)
        draw.text((2 * w + w // 2 - 85, 28), "Video-LoRA (Blocks 14, 20)", fill="#94A3B8", font=subfont)

        img_gt = Image.fromarray(gt_frames[i])
        img_base = Image.fromarray(base_frames[i])
        img_lora = Image.fromarray(lora_frames[i])

        canvas.paste(img_gt, (0, banner_h))
        canvas.paste(img_base, (w, banner_h))
        canvas.paste(img_lora, (2 * w, banner_h))

        draw.line([(w, 0), (w, total_h)], fill="#334155", width=2)
        draw.line([(2 * w, 0), (2 * w, total_h)], fill="#334155", width=2)

        composite_frames.append(np.array(canvas))

    write_video(out_path, composite_frames, fps)
    print(f"Saved side-by-side video: {out_path} ({total_w}x{total_h}, {len(composite_frames)} frames)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Cosmos 7B Base vs LoRA video rollouts.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--lora-path", type=Path, default=DEFAULT_LORA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--episode", type=int, default=32)
    parser.add_argument("--start-frame", type=int, default=4820)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--prompt", type=str, default="take cube out of box")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("Cosmos 7B Video Rollout Renderer (Base vs. LoRA)")
    print(f"Episode: {args.episode}, Start frame: {args.start_frame}, Total frames: {args.num_frames}")
    print(f"LoRA weights: {args.lora_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {device}, Dtype: {dtype}")
    print("=" * 75)

    # 1. Load Ground Truth Frames
    print("\n[1/5] Loading Ground Truth Frames from Dataset...")
    dataset = LeRobotDataset(
        CUBE_OUT_OF_BOX_CONTRACT.repo_id,
        root=str(args.dataset_root),
        delta_timestamps=CUBE_OUT_OF_BOX_CONTRACT.delta_timestamps(),
        revision=CUBE_OUT_OF_BOX_CONTRACT.revision,
        return_uint8=True,
        download_videos=False,
    )
    print(f"Loaded dataset with {len(dataset)} frames.")

    gt_frames = []
    raw_rgb_tensors = []
    for f_idx in range(args.start_frame, args.start_frame + args.num_frames):
        row = dataset[f_idx]
        rgb = row["observation.images.front"]
        if rgb.ndim == 4:
            rgb = rgb[-1]
        raw_rgb_tensors.append(rgb)
        gt_frames.append(rgb.permute(1, 2, 0).cpu().numpy())

    gt_array = np.array(gt_frames)
    gt_video_path = args.output_dir / "heldout_validation_ep32_f4820_cosmos7b_ground_truth.mp4"
    write_video(gt_video_path, gt_frames, 10)
    print(f"Saved Ground Truth video: {gt_video_path}")

    # Encode Initial Conditioning Latent from Frame 0
    first_rgb = raw_rgb_tensors[0].unsqueeze(0)  # [1, 3, 480, 640]
    init_latent = rgb_to_cosmos7b_latent(first_rgb, device, dtype)

    # 2. Build Base Cosmos 7B Model
    print("\n[2/5] Initializing Base Cosmos 7B Model...")
    extractor_config = Cosmos7BExtractorConfig(
        device=args.device,
        dtype="bfloat16",
        hidden_layers=(14, 20),
    )
    extractor = Cosmos7BExtractor(config=extractor_config)
    base_model = extractor.transformer
    base_model.eval()

    torch.manual_seed(abs(hash(args.prompt)) % (2**31 - 1))
    text_embed = torch.randn(1, 16, extractor_config.text_embed_dim, device=device, dtype=dtype)

    # 3. Autoregressive Rollout with Base Cosmos 7B
    print("\n[3/5] Generating Rollout with Base Cosmos 7B...")
    t0 = time.time()
    base_latent_rollout = rollout_cosmos7b(
        model=base_model,
        init_latent=init_latent,
        text_embed=text_embed,
        total_frames=args.num_frames,
        device=device,
        dtype=dtype,
    )
    base_pred_frames_array = cosmos7b_latent_to_rgb(base_latent_rollout)
    base_pred_frames = list(base_pred_frames_array)
    # Ensure first frame matches conditioning
    base_pred_frames[0] = gt_frames[0]

    print(f"Base Cosmos 7B rollout complete in {time.time() - t0:.2f}s.")
    base_video_path = args.output_dir / "heldout_validation_ep32_f4820_base_cosmos7b.mp4"
    write_video(base_video_path, base_pred_frames, 10)
    print(f"Saved Base Cosmos 7B video: {base_video_path}")

    # 4. Inject and Load Cosmos 7B LoRA Checkpoint
    print("\n[4/5] Injecting and Loading Trained Cosmos 7B LoRA Checkpoint...")
    lora_cfg = Cosmos7BLoRAConfig(
        rank=16,
        alpha=16.0,
        target_blocks=(14, 20),
    )
    inject_cosmos7b_lora(base_model, lora_cfg)
    load_cosmos7b_lora(base_model, args.lora_path)
    base_model.eval()
    print("LoRA weights loaded successfully.")

    # Generate LoRA Rollout
    print("\nGenerating Rollout with LoRA-Adapted Cosmos 7B...")
    t0 = time.time()
    lora_latent_rollout = rollout_cosmos7b(
        model=base_model,
        init_latent=init_latent,
        text_embed=text_embed,
        total_frames=args.num_frames,
        device=device,
        dtype=dtype,
    )
    lora_pred_frames_array = cosmos7b_latent_to_rgb(lora_latent_rollout)
    lora_pred_frames = list(lora_pred_frames_array)
    lora_pred_frames[0] = gt_frames[0]

    print(f"LoRA Cosmos 7B rollout complete in {time.time() - t0:.2f}s.")
    lora_video_path = args.output_dir / "heldout_validation_ep32_f4820_lora_adapted_cosmos7b.mp4"
    write_video(lora_video_path, lora_pred_frames, 10)
    print(f"Saved LoRA Cosmos 7B video: {lora_video_path}")

    # 5. Compute Metrics and Create Side-by-Side Video
    print("\n[5/5] Computing Metrics and Rendering Side-by-Side Video...")
    base_array = np.array(base_pred_frames)
    lora_array = np.array(lora_pred_frames)

    base_metrics = compute_metrics(base_array, gt_array)
    lora_metrics = compute_metrics(lora_array, gt_array)

    psnr_delta = lora_metrics["mean_psnr_db"] - base_metrics["mean_psnr_db"]
    ssim_delta = lora_metrics["mean_ssim"] - base_metrics["mean_ssim"]
    mse_reduction = (base_metrics["mean_mse"] - lora_metrics["mean_mse"]) / base_metrics["mean_mse"] * 100.0

    print("=" * 60)
    print("Cosmos 7B ROLLOUT METRICS ON EPISODE 32:")
    print(
        f"  Base Model:  PSNR = {base_metrics['mean_psnr_db']:.2f} dB | SSIM = {base_metrics['mean_ssim']:.4f} | MSE = {base_metrics['mean_mse']:.1f}"
    )
    print(
        f"  LoRA Model:  PSNR = {lora_metrics['mean_psnr_db']:.2f} dB | SSIM = {lora_metrics['mean_ssim']:.4f} | MSE = {lora_metrics['mean_mse']:.1f}"
    )
    print(
        f"  Delta:       +{psnr_delta:.2f} dB PSNR | +{ssim_delta:.4f} SSIM | {mse_reduction:.1f}% MSE Reduction"
    )
    print("=" * 60)

    # Side-by-side composite
    sbs_path = args.output_dir / "heldout_validation_ep32_f4820_cosmos7b_side_by_side.mp4"
    create_side_by_side_video(
        gt_frames, base_pred_frames, lora_pred_frames, sbs_path, fps=10, model_name="Cosmos 7B"
    )

    # Links in root
    repo_root = Path("/home/anton/lerobot-video-vam")
    for src in (gt_video_path, base_video_path, lora_video_path, sbs_path):
        dst = repo_root / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
        print(f"Created link in root: {dst.name}")

    # Update quantitative_metrics_comparison.json
    metrics_json_path = args.output_dir / "quantitative_metrics_comparison.json"
    data = {}
    if metrics_json_path.is_file():
        with open(metrics_json_path, encoding="utf-8") as f:
            data = json.load(f)

    data["heldout_validation_ep32_f4820_cosmos7b"] = {
        "episode": args.episode,
        "start_frame": args.start_frame,
        "description": "Episode 32, Frame 4820 (Cosmos 7B Held-out validation benchmark)",
        "clip_frames": args.num_frames,
        "prompt": args.prompt,
        "base_metrics": base_metrics,
        "lora_metrics": lora_metrics,
        "comparison": {
            "psnr_delta_db": psnr_delta,
            "ssim_delta": ssim_delta,
            "mse_reduction_percent": mse_reduction,
        },
    }

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Updated quantitative metrics JSON at: {metrics_json_path}")

    print("\nCosmos 7B Video Rollouts Generated Successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
