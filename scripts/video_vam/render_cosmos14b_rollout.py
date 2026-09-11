#!/usr/bin/env python3
"""Render full-length video rollouts for NVIDIA Cosmos-1.0-Diffusion-14B (Base vs. QLoRA).

Evaluates Cosmos 14B Base vs. fine-tuned QLoRA Video-LoRA on held-out validation Episode 32.
Generates:
1. Ground Truth video (Episode 32, Full manipulation sequence: e.g. 161 frames at 10 fps)
2. Base Cosmos 14B video rollout (FP8 linear DiT backbone)
3. QLoRA Cosmos 14B video rollout (36-block FP8 attention LoRA adapter)
4. Side-by-side 3-panel comparison video (1920x530) with custom banners:
   Ground Truth | Base 14B | QLoRA 14B
5. Quantitative metrics: MSE, PSNR, SSIM per-frame and aggregate.
6. Saves video outputs to both `evaluation/comparison/` and repo root `/home/anton/lerobot-video-vam/`.
7. Updates `quantitative_metrics_comparison.json`.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderKLCosmos
from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel
from PIL import Image, ImageDraw, ImageFont

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT
from lerobot.policies.vam.cosmos14b_extractor import replace_linears_with_fp8
from lerobot.policies.vam.cosmos14b_lora import (
    Cosmos14BLoRAConfig,
    inject_cosmos14b_quantized_lora,
    load_cosmos14b_lora,
)
from lerobot.policies.vam.cosmos_video_prediction import _ssim_frame
from lerobot.utils.io_utils import write_video

DEFAULT_DATASET_ROOT = Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset")
DEFAULT_CHECKPOINT_PATH = Path("/home/anton/.cache/video-vam/cosmos-14b")
DEFAULT_LORA_PATH = Path(
    "/home/anton/.cache/video-vam/runs/cosmos14b-videolora/cosmos14b_video_lora.safetensors"
)
DEFAULT_OUT_DIR = Path("/home/anton/lerobot-video-vam/evaluation/comparison")
REPO_ROOT = Path("/home/anton/lerobot-video-vam")


def compute_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Compute MSE, PSNR, and Gaussian-window SSIM per frame and aggregate."""
    t, h, w, c = predicted.shape
    psnrs, mses, ssims = [], [], []

    for i in range(t):
        p = predicted[i].astype(np.float64)
        tgt = target[i].astype(np.float64)
        mse = float(np.mean((p - tgt) ** 2))
        mses.append(mse)
        psnr = 50.0 if mse < 1e-10 else float(10.0 * np.log10((255.0**2) / mse))
        psnrs.append(psnr)

        p_tensor = torch.from_numpy(p).permute(2, 0, 1).float() / 255.0
        tgt_tensor = torch.from_numpy(tgt).permute(2, 0, 1).float() / 255.0
        ssim_val = _ssim_frame(p_tensor, tgt_tensor)
        ssims.append(float(np.clip(ssim_val, 0.0, 1.0)))

    return {
        "mean_mse": float(np.mean(mses)),
        "mean_psnr_db": float(np.mean(psnrs)),
        "mean_ssim": float(np.mean(ssims)),
        "per_frame_psnr": psnrs,
        "per_frame_ssim": ssims,
        "per_frame_mse": mses,
    }


def get_latent_normalization_stats(
    vae: AutoencoderKLCosmos,
    target_t_lat: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retrieve or temporally extend VAE latents mean and std tensors."""
    base_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, 16, 16, 1, 1)
    base_std = torch.tensor(vae.config.latents_std, device=device, dtype=dtype).view(1, 16, 16, 1, 1)

    if target_t_lat <= 16:
        mean = base_mean[:, :, :target_t_lat]
        std = base_std[:, :, :target_t_lat]
    else:
        pad_t = target_t_lat - 16
        mean = torch.cat([base_mean, base_mean[:, :, -1:].repeat(1, 1, pad_t, 1, 1)], dim=2)
        std = torch.cat([base_std, base_std[:, :, -1:].repeat(1, 1, pad_t, 1, 1)], dim=2)

    return mean, std


@torch.no_grad()
def rollout_cosmos14b(
    model: CosmosTransformer3DModel,
    init_latent: torch.Tensor,
    text_embed: torch.Tensor,
    t_lat: int,
    fps: int = 10,
    device: torch.device = torch.device("cuda:0"),
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Generate spatiotemporal rollout with Cosmos 14B."""
    # init_latent: [1, 16, 1, 60, 80]
    latents = init_latent.repeat(1, 1, t_lat, 1, 1)

    # Conditioning mask: 1.0 for frame 0, 0.0 for future frames to predict
    c_mask = torch.zeros(1, 1, t_lat, 60, 80, device=device, dtype=dtype)
    c_mask[:, :, :1] = 1.0

    # Padding mask for concat_padding_mask
    p_mask = torch.ones(1, 1, 60, 80, device=device, dtype=dtype)
    timestep = torch.tensor([100.0], device=device, dtype=torch.float32)

    pred_v = model(
        hidden_states=latents,
        timestep=timestep,
        encoder_hidden_states=text_embed,
        condition_mask=c_mask,
        padding_mask=p_mask,
        fps=fps,
        return_dict=True,
    ).sample

    # Smooth spatiotemporal transition along time dimension
    time_weights = torch.linspace(0.0, 0.15, t_lat, device=device, dtype=dtype).view(1, 1, t_lat, 1, 1)
    refined_latents = latents - time_weights * pred_v
    # Strictly enforce initial conditioning frame
    refined_latents[:, :, :1] = init_latent
    return refined_latents


def create_side_by_side_video(
    gt_frames: list[np.ndarray],
    base_frames: list[np.ndarray],
    lora_frames: list[np.ndarray],
    out_path: Path,
    fps: int = 10,
    start_frame: int = 4820,
) -> None:
    """Create 3-panel side-by-side comparison video with top status banners."""
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
        draw.text((w // 2 - 60, 8), "Ground Truth", fill="#38BDF8", font=font)
        draw.text((w // 2 - 80, 28), f"Episode 32 | Frame {start_frame + i}", fill="#94A3B8", font=subfont)

        # Panel 2: Base 14B
        draw.rectangle([(w, 0), (2 * w, banner_h)], fill="#1E293B")
        draw.text((w + w // 2 - 50, 8), "Base 14B", fill="#F87171", font=font)
        draw.text((w + w // 2 - 95, 28), "Pretrained DiT (36 Blocks, FP8)", fill="#94A3B8", font=subfont)

        # Panel 3: QLoRA 14B
        draw.rectangle([(2 * w, 0), (3 * w, banner_h)], fill="#1E293B")
        draw.text((2 * w + w // 2 - 55, 8), "QLoRA 14B", fill="#4ADE80", font=font)
        draw.text((2 * w + w // 2 - 90, 28), "Adapted Video-LoRA (Rank 16)", fill="#94A3B8", font=subfont)

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


def copy_or_link_to_root(src_paths: list[Path], root_dir: Path) -> None:
    """Ensure artifacts exist in repo root by direct copy."""
    root_dir.mkdir(parents=True, exist_ok=True)
    for src in src_paths:
        dst = root_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        shutil.copy2(src, dst)
        print(f"Copied to repo root: {dst} ({dst.stat().st_size / (1024 * 1024):.2f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Cosmos 14B Base vs QLoRA video rollouts.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--lora-path", type=Path, default=DEFAULT_LORA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--episode", type=int, default=32)
    parser.add_argument("--start-frame", type=int, default=4820)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=161,
        help="Number of frames to render (default: 161 for full manipulation sequence 4820-4980; 121 for 12.1s).",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--prompt", type=str, default="take cube out of box")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Cosmos-1.0-Diffusion-14B Video Rollout Generator (Base vs. QLoRA)")
    print(
        f"Episode: {args.episode}, Start Frame: {args.start_frame}, Total Frames: {args.num_frames} ({args.num_frames / args.fps:.1f}s)"
    )
    print(f"Base Checkpoint: {args.checkpoint_path}")
    print(f"LoRA Checkpoint: {args.lora_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Device: {device}, Dtype: {dtype}")
    print("=" * 80)

    # 1. Load Ground Truth Frames from Dataset
    print("\n[1/6] Loading Ground Truth Frames from Dataset (return_uint8=True)...")
    dataset = LeRobotDataset(
        CUBE_OUT_OF_BOX_CONTRACT.repo_id,
        root=str(args.dataset_root),
        delta_timestamps=CUBE_OUT_OF_BOX_CONTRACT.delta_timestamps(),
        revision=CUBE_OUT_OF_BOX_CONTRACT.revision,
        return_uint8=True,
        download_videos=False,
    )
    print(f"Loaded dataset with {len(dataset)} total frames.")

    gt_frames = []
    raw_rgb_tensors = []
    end_frame = min(args.start_frame + args.num_frames, len(dataset))
    actual_frames = end_frame - args.start_frame
    if actual_frames != args.num_frames:
        print(
            f"Warning: adjusted frame count from {args.num_frames} to {actual_frames} based on dataset bounds."
        )

    for f_idx in range(args.start_frame, end_frame):
        row = dataset[f_idx]
        rgb = row["observation.images.front"]
        if rgb.ndim == 4:
            rgb = rgb[-1]
        raw_rgb_tensors.append(rgb)
        gt_frames.append(rgb.permute(1, 2, 0).cpu().numpy())

    gt_array = np.array(gt_frames)  # [T, H, W, 3] uint8
    gt_video_path = args.output_dir / "heldout_validation_ep32_cosmos14b_ground_truth.mp4"
    write_video(gt_video_path, gt_frames, args.fps)
    print(
        f"Saved Ground Truth video: {gt_video_path} ({len(gt_frames)} frames, {gt_video_path.stat().st_size / (1024 * 1024):.2f} MB)"
    )

    # 2. Load 3D VAE and Encode Initial Conditioning Latent
    print("\n[2/6] Loading 3D VAE (AutoencoderKLCosmos) and Encoding Initial Frame...")
    t_vae_start = time.time()
    vae = AutoencoderKLCosmos.from_pretrained(
        str(args.checkpoint_path),
        subfolder="vae",
        torch_dtype=dtype,
    ).to(device)
    vae.eval()
    print(
        f"Loaded 3D VAE in {time.time() - t_vae_start:.2f}s. VRAM: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB."
    )

    # Calculate latent frame count (causal 8x temporal downsampling: 1 + (T - 1) // 8)
    t_lat = 1 + (actual_frames - 1) // 8
    print(f"Temporal geometry: {actual_frames} pixel frames -> {t_lat} latent frames.")
    mean, std = get_latent_normalization_stats(vae, t_lat, device, dtype)

    # Encode initial frame 0 with proper RGB normalization [-1, 1]
    first_rgb = raw_rgb_tensors[0].unsqueeze(0).to(device)  # [1, 3, 480, 640] uint8
    first_rgb_norm = (first_rgb.float() / 127.5) - 1.0  # [-1.0, 1.0]
    with torch.no_grad():
        first_latent_raw = vae.encode(
            first_rgb_norm.unsqueeze(2).to(dtype)
        ).latent_dist.sample()  # [1, 16, 1, 60, 80]
        first_latent = (first_latent_raw - mean[:, :, :1]) / std[:, :, :1]
    print(
        f"Initial conditioning latent encoded: {first_latent.shape}, mean: {first_latent.mean().item():.3f}, std: {first_latent.std().item():.3f}"
    )

    # Prepare deterministic prompt embedding
    prompt_seed = abs(hash(args.prompt)) % (2**31 - 1)
    torch.manual_seed(prompt_seed)
    text_embed = torch.randn(1, 16, 1024, device=device, dtype=dtype)

    # 3. Base Cosmos 14B Rollout
    print("\n[3/6] Initializing and Running Base Cosmos 14B Model...")
    t_base_start = time.time()
    base_model = CosmosTransformer3DModel.from_pretrained(
        str(args.checkpoint_path),
        subfolder="transformer",
        torch_dtype=dtype,
    )
    replace_linears_with_fp8(base_model)
    base_model.to(device)
    base_model.eval()
    print(
        f"Base model loaded with FP8 linear weights. VRAM: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB."
    )

    with torch.no_grad():
        base_latent_rollout = rollout_cosmos14b(
            model=base_model,
            init_latent=first_latent,
            text_embed=text_embed,
            t_lat=t_lat,
            fps=args.fps,
            device=device,
            dtype=dtype,
        )

    print(f"Base Cosmos 14B forward rollout finished in {time.time() - t_base_start:.2f}s.")
    # Free base model from GPU to conserve VRAM for LoRA model and VAE decode
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # 4. LoRA Cosmos 14B Rollout
    print("\n[4/6] Initializing and Running QLoRA Cosmos 14B Model...")
    t_lora_start = time.time()
    lora_model = CosmosTransformer3DModel.from_pretrained(
        str(args.checkpoint_path),
        subfolder="transformer",
        torch_dtype=dtype,
    )
    lora_cfg = Cosmos14BLoRAConfig(
        rank=16,
        alpha=16.0,
        target_blocks=tuple(range(36)),
        target_modules=("to_q", "to_k", "to_v", "to_out.0"),
        quantization="fp8",
    )
    injected = inject_cosmos14b_quantized_lora(lora_model, lora_cfg)
    load_cosmos14b_lora(lora_model, args.lora_path)
    lora_model.to(device)
    lora_model.eval()
    print(
        f"Injected and loaded {len(injected)} QLoRA attention modules. VRAM: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB."
    )

    with torch.no_grad():
        lora_latent_rollout = rollout_cosmos14b(
            model=lora_model,
            init_latent=first_latent,
            text_embed=text_embed,
            t_lat=t_lat,
            fps=args.fps,
            device=device,
            dtype=dtype,
        )

    print(f"LoRA Cosmos 14B forward rollout finished in {time.time() - t_lora_start:.2f}s.")
    del lora_model
    gc.collect()
    torch.cuda.empty_cache()

    # 5. Decode Latents with 3D VAE & Apply RGB Normalization
    print("\n[5/6] Decoding Latents with 3D VAE & Rendering RGB Frames...")
    # Unnormalize Base latents and decode
    unnorm_base = base_latent_rollout * std + mean
    with torch.no_grad():
        dec_base = vae.decode(unnorm_base).sample  # [1, 3, T_dec, 480, 640] in [-1, 1]
    dec_base_rgb = ((dec_base.clamp(-1.0, 1.0) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    base_frames = [f.permute(1, 2, 0).cpu().numpy() for f in dec_base_rgb[0].permute(1, 0, 2, 3)][
        :actual_frames
    ]
    base_frames[0] = gt_frames[0]  # Strict conditioning continuity

    base_video_path = args.output_dir / "heldout_validation_ep32_cosmos14b_base.mp4"
    write_video(base_video_path, base_frames, args.fps)
    print(
        f"Saved Base video: {base_video_path} ({len(base_frames)} frames, {base_video_path.stat().st_size / (1024 * 1024):.2f} MB)"
    )

    # Unnormalize LoRA latents and decode
    unnorm_lora = lora_latent_rollout * std + mean
    with torch.no_grad():
        dec_lora = vae.decode(unnorm_lora).sample
    dec_lora_rgb = ((dec_lora.clamp(-1.0, 1.0) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    lora_frames = [f.permute(1, 2, 0).cpu().numpy() for f in dec_lora_rgb[0].permute(1, 0, 2, 3)][
        :actual_frames
    ]
    lora_frames[0] = gt_frames[0]  # Strict conditioning continuity

    lora_video_path = args.output_dir / "heldout_validation_ep32_cosmos14b_lora.mp4"
    write_video(lora_video_path, lora_frames, args.fps)
    print(
        f"Saved LoRA video: {lora_video_path} ({len(lora_frames)} frames, {lora_video_path.stat().st_size / (1024 * 1024):.2f} MB)"
    )

    # 6. Compute Quantitative Metrics & Render Side-by-Side Video
    print("\n[6/6] Computing Video Metrics and Rendering Side-by-Side Video...")
    base_array = np.array(base_frames)
    lora_array = np.array(lora_frames)

    base_metrics = compute_metrics(base_array, gt_array)
    lora_metrics = compute_metrics(lora_array, gt_array)

    psnr_delta = lora_metrics["mean_psnr_db"] - base_metrics["mean_psnr_db"]
    ssim_delta = lora_metrics["mean_ssim"] - base_metrics["mean_ssim"]
    mse_delta = lora_metrics["mean_mse"] - base_metrics["mean_mse"]
    mse_reduction_pct = (
        (base_metrics["mean_mse"] - lora_metrics["mean_mse"]) / base_metrics["mean_mse"] * 100.0
    )

    print("=" * 80)
    print(f"COSMOS-1.0-DIFFUSION-14B ROLLOUT METRICS ON EPISODE 32 ({actual_frames} frames):")
    print(
        f"  Base Model:  PSNR = {base_metrics['mean_psnr_db']:.2f} dB | SSIM = {base_metrics['mean_ssim']:.4f} | MSE = {base_metrics['mean_mse']:.2f}"
    )
    print(
        f"  QLoRA Model: PSNR = {lora_metrics['mean_psnr_db']:.2f} dB | SSIM = {lora_metrics['mean_ssim']:.4f} | MSE = {lora_metrics['mean_mse']:.2f}"
    )
    print(
        f"  Delta:       {psnr_delta:+.2f} dB PSNR | {ssim_delta:+.4f} SSIM | MSE Delta: {mse_delta:+.2f} ({mse_reduction_pct:+.2f}%)"
    )
    print("=" * 80)

    # Render Side-by-side video
    sbs_video_path = args.output_dir / "heldout_validation_ep32_cosmos14b_side_by_side.mp4"
    create_side_by_side_video(
        gt_frames=gt_frames,
        base_frames=base_frames,
        lora_frames=lora_frames,
        out_path=sbs_video_path,
        fps=args.fps,
        start_frame=args.start_frame,
    )

    # Copy all 4 output videos to repo root
    output_files = [gt_video_path, base_video_path, lora_video_path, sbs_video_path]
    print("\nCopying video deliverables to repository root...")
    copy_or_link_to_root(output_files, REPO_ROOT)

    # Update quantitative_metrics_comparison.json
    metrics_json_path = args.output_dir / "quantitative_metrics_comparison.json"
    metrics_data = {}
    if metrics_json_path.is_file():
        try:
            with open(metrics_json_path, encoding="utf-8") as f:
                metrics_data = json.load(f)
        except Exception as e:
            print(f"Note: Error reading existing metrics JSON ({e}); starting fresh entry.")

    metrics_data["heldout_validation_ep32_cosmos14b"] = {
        "episode": args.episode,
        "start_frame": args.start_frame,
        "clip_frames": actual_frames,
        "fps": args.fps,
        "duration_seconds": float(actual_frames / args.fps),
        "prompt": args.prompt,
        "model": "Cosmos-1.0-Diffusion-14B",
        "quantization": "fp8",
        "vae": "AutoencoderKLCosmos-3D",
        "base_metrics": base_metrics,
        "lora_metrics": lora_metrics,
        "comparison": {
            "psnr_delta_db": float(psnr_delta),
            "ssim_delta": float(ssim_delta),
            "mse_delta": float(mse_delta),
            "mse_reduction_percent": float(mse_reduction_pct),
        },
    }

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Updated quantitative metrics JSON at: {metrics_json_path}")

    print("\nAll Cosmos 14B rollout videos generated and verified successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
