#!/usr/bin/env python3
"""Render autoregressive video rollouts for FLUX.2 [klein] (Base vs. LoRA).

Generates:
1. Ground Truth video (Episode 32, Frames 4820-4836; 17 frames, 10 fps)
2. Base FLUX.2 video rollout
3. LoRA FLUX.2 video rollout (using flux2_klein_lora.safetensors)
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
from lerobot.policies.vam.flux2_klein_extractor import (
    Flux2KleinExtractor,
    Flux2KleinExtractorConfig,
    prepare_multi_reference_conditioning,
)
from lerobot.policies.vam.flux2_klein_lora import (
    Flux2KleinLoRAConfig,
    inject_flux2_klein_lora,
    load_flux2_klein_lora,
)
from lerobot.utils.io_utils import write_video

DEFAULT_DATASET_ROOT = Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset")
DEFAULT_LORA_PATH = Path(
    "/home/anton/.cache/video-vam/runs/flux2-klein-videolora/flux2_klein_lora.safetensors"
)
DEFAULT_OUT_DIR = Path("/home/anton/lerobot-video-vam/evaluation/comparison")


def compute_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Compute MSE, PSNR, and SSIM per frame and aggregate."""
    # predicted, target: [T, H, W, C] in uint8 [0, 255]
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

        # SSIM calculation
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


def rgb_to_flux_latent(rgb_tensor: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Encode RGB frame [C=3, H=480, W=640] into FLUX.2 latent [1, 128, 8, 8]."""
    rgb_norm = (rgb_tensor.float().to(device) / 255.0 - 0.5) * 2.0
    if rgb_norm.ndim == 3:
        rgb_norm = rgb_norm.unsqueeze(0)
    down = F.interpolate(rgb_norm, size=(32, 32), mode="bilinear", align_corners=False)
    p48 = down.view(1, 3, 8, 4, 8, 4).permute(0, 1, 3, 5, 2, 4).reshape(1, 48, 8, 8)
    p128 = F.pad(p48, (0, 0, 0, 0, 0, 80)).to(dtype=dtype)
    return p128


def flux_latent_to_rgb(latent: torch.Tensor, target_h: int = 480, target_w: int = 640) -> np.ndarray:
    """Decode FLUX.2 latent [1, 128, 8, 8] to RGB frame [H, W, 3] in uint8."""
    # Extract original 48 channels
    p48 = latent[:, :48, :, :]
    rec = p48.view(1, 3, 4, 4, 8, 8).permute(0, 1, 4, 2, 5, 3).reshape(1, 3, 32, 32)
    up = F.interpolate(rec.float(), size=(target_h, target_w), mode="bilinear", align_corners=False)
    rgb = ((up.squeeze(0).clamp(-1.0, 1.0) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return rgb.permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def sample_next_frame_rf(
    model: Any,
    history_latents: list[torch.Tensor],
    goal_latent: torch.Tensor,
    joint_attention_dim: int,
    timestep_val: float = 200.0,
    guidance: float = 3.5,
    seed: int = 0,
    device: torch.device = torch.device("cuda:0"),
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run flow matching prediction from current history towards next frame."""
    torch.manual_seed(seed)
    current_frame = history_latents[-1]

    cond = prepare_multi_reference_conditioning(
        target_latent=current_frame,
        observation_history=history_latents,
        goal_latent=goal_latent,
        joint_attention_dim=joint_attention_dim,
        device=device,
        dtype=dtype,
    )

    t_batch = torch.tensor([float(timestep_val)], device=device, dtype=torch.float32)
    g_tensor = torch.tensor([float(guidance)], device=device, dtype=torch.float32)

    pred_sample = model(
        hidden_states=cond.hidden_states,
        encoder_hidden_states=cond.encoder_hidden_states,
        timestep=t_batch,
        img_ids=cond.img_ids,
        txt_ids=cond.txt_ids,
        guidance=g_tensor,
        return_dict=True,
    ).sample

    # Target slice
    pred_v = pred_sample[:, : cond.target_tokens_count]
    v_latent = pred_v.permute(0, 2, 1).reshape(1, 128, 8, 8)

    # Transition step towards next temporal frame
    next_latent = current_frame - (timestep_val / 1000.0) * v_latent
    return next_latent


def create_side_by_side_video(
    gt_frames: list[np.ndarray],
    base_frames: list[np.ndarray],
    lora_frames: list[np.ndarray],
    out_path: Path,
    fps: int = 10,
    model_name: str = "FLUX.2 [klein]",
) -> None:
    """Create side-by-side comparison video with top status banners."""
    t = len(gt_frames)
    w, h = 640, 480
    banner_h = 50
    total_w = w * 3
    total_h = h + banner_h

    # Create composite frames
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
        draw.text((w + w // 2 - 75, 28), "Autoregressive (Unadapted)", fill="#94A3B8", font=subfont)

        # Panel 3: LoRA Adapted
        draw.rectangle([(2 * w, 0), (3 * w, banner_h)], fill="#1E293B")
        draw.text((2 * w + w // 2 - 95, 8), f"{model_name} LoRA Adapted", fill="#4ADE80", font=font)
        draw.text((2 * w + w // 2 - 85, 28), "Autoregressive (Adapted)", fill="#94A3B8", font=subfont)

        # Paste images
        img_gt = Image.fromarray(gt_frames[i])
        img_base = Image.fromarray(base_frames[i])
        img_lora = Image.fromarray(lora_frames[i])

        canvas.paste(img_gt, (0, banner_h))
        canvas.paste(img_base, (w, banner_h))
        canvas.paste(img_lora, (2 * w, banner_h))

        # Thin dividing lines between panels
        draw.line([(w, 0), (w, total_h)], fill="#334155", width=2)
        draw.line([(2 * w, 0), (2 * w, total_h)], fill="#334155", width=2)

        composite_frames.append(np.array(canvas))

    write_video(out_path, composite_frames, fps)
    print(f"Saved side-by-side video: {out_path} ({total_w}x{total_h}, {len(composite_frames)} frames)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render FLUX.2 [klein] Base vs LoRA video rollouts.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--lora-path", type=Path, default=DEFAULT_LORA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--episode", type=int, default=32)
    parser.add_argument("--start-frame", type=int, default=4820)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("FLUX.2 [klein] Video Rollout Renderer (Base vs. LoRA)")
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
    for f_idx in range(args.start_frame, args.start_frame + args.num_frames):
        row = dataset[f_idx]
        rgb = row["observation.images.front"]  # [3, 480, 640] or [1, 3, 480, 640]
        if rgb.ndim == 4:
            rgb = rgb[-1]
        gt_frames.append(rgb.permute(1, 2, 0).cpu().numpy())

    gt_array = np.array(gt_frames)
    gt_video_path = args.output_dir / "heldout_validation_ep32_f4820_flux2_klein_ground_truth.mp4"
    write_video(gt_video_path, gt_frames, 10)
    print(f"Saved Ground Truth video: {gt_video_path}")

    # Encode Goal Frame (Frame 16)
    goal_rgb = torch.from_numpy(gt_frames[-1]).permute(2, 0, 1)
    goal_latent = rgb_to_flux_latent(goal_rgb, device, dtype)

    # 2. Build Base FLUX.2 Model
    print("\n[2/5] Initializing Base FLUX.2 Model...")
    extractor_cfg = Flux2KleinExtractorConfig(
        device=args.device,
        dtype="bfloat16",
        num_layers=8,
        num_single_layers=0,
    )
    extractor = Flux2KleinExtractor(config=extractor_cfg)
    base_model = extractor.transformer
    base_model.eval()

    # 3. Autoregressive Rollout with Base Model
    print("\n[3/5] Generating Autoregressive Rollout with Base FLUX.2...")
    base_pred_frames = [gt_frames[0]]  # First frame is conditioning
    # Start with first frame repeated as initial history: t-2, t-1, t
    hist_latents = [
        rgb_to_flux_latent(torch.from_numpy(gt_frames[0]).permute(2, 0, 1), device, dtype) for _ in range(3)
    ]

    t0 = time.time()
    for k in range(1, args.num_frames):
        # Generate next frame
        next_latent = sample_next_frame_rf(
            model=base_model,
            history_latents=hist_latents,
            goal_latent=goal_latent,
            joint_attention_dim=extractor_cfg.joint_attention_dim,
            timestep_val=150.0,
            seed=42 + k,
            device=device,
            dtype=dtype,
        )
        pred_rgb = flux_latent_to_rgb(next_latent)
        base_pred_frames.append(pred_rgb)

        # Autoregressively update history: pop oldest, append newly generated latent
        hist_latents = hist_latents[1:] + [next_latent]
        print(f"  Base FLUX.2 generated frame {k + 1}/{args.num_frames}...")

    print(f"Base FLUX.2 rollout complete in {time.time() - t0:.2f}s.")
    base_video_path = args.output_dir / "heldout_validation_ep32_f4820_base_flux2_klein.mp4"
    write_video(base_video_path, base_pred_frames, 10)
    print(f"Saved Base FLUX.2 video: {base_video_path}")

    # 4. Inject and Load FLUX.2 LoRA Checkpoint
    print("\n[4/5] Injecting and Loading Trained FLUX.2 LoRA Checkpoint...")
    lora_cfg = Flux2KleinLoRAConfig(
        rank=16,
        alpha=16.0,
        target_double_blocks=tuple(range(8)),
        target_single_blocks=(),
    )
    inject_flux2_klein_lora(base_model, lora_cfg)
    load_flux2_klein_lora(base_model, args.lora_path)
    base_model.eval()
    print("LoRA weights loaded successfully.")

    # Generate LoRA Autoregressive Rollout
    print("\nGenerating Autoregressive Rollout with LoRA-Adapted FLUX.2...")
    lora_pred_frames = [gt_frames[0]]
    hist_latents = [
        rgb_to_flux_latent(torch.from_numpy(gt_frames[0]).permute(2, 0, 1), device, dtype) for _ in range(3)
    ]

    t0 = time.time()
    for k in range(1, args.num_frames):
        next_latent = sample_next_frame_rf(
            model=base_model,
            history_latents=hist_latents,
            goal_latent=goal_latent,
            joint_attention_dim=extractor_cfg.joint_attention_dim,
            timestep_val=150.0,
            seed=42 + k,
            device=device,
            dtype=dtype,
        )
        pred_rgb = flux_latent_to_rgb(next_latent)
        lora_pred_frames.append(pred_rgb)
        hist_latents = hist_latents[1:] + [next_latent]
        print(f"  LoRA FLUX.2 generated frame {k + 1}/{args.num_frames}...")

    print(f"LoRA FLUX.2 rollout complete in {time.time() - t0:.2f}s.")
    lora_video_path = args.output_dir / "heldout_validation_ep32_f4820_lora_adapted_flux2_klein.mp4"
    write_video(lora_video_path, lora_pred_frames, 10)
    print(f"Saved LoRA FLUX.2 video: {lora_video_path}")

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
    print("FLUX.2 [klein] ROLLOUT METRICS ON EPISODE 32:")
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
    sbs_path = args.output_dir / "heldout_validation_ep32_f4820_flux2_klein_side_by_side.mp4"
    create_side_by_side_video(
        gt_frames, base_pred_frames, lora_pred_frames, sbs_path, fps=10, model_name="FLUX.2 [klein]"
    )

    # Also symlink or copy to root dir if needed
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

    data["heldout_validation_ep32_f4820_flux2_klein"] = {
        "episode": args.episode,
        "start_frame": args.start_frame,
        "description": "Episode 32, Frame 4820 (FLUX.2 [klein] Held-out validation benchmark)",
        "clip_frames": args.num_frames,
        "inference_steps": args.steps,
        "prompt": "take cube out of box",
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

    print("\nFLUX.2 [klein] Video Rollouts Generated Successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
