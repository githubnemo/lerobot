#!/usr/bin/env python3
"""Cosmos3-Edge feature extractor for Physical AI representation learning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from diffusers import Cosmos3OmniPipeline
from PIL import Image

DEFAULT_COSMOS3_EDGE_DIR = Path("/home/anton/.cache/video-vam/cosmos3-edge")


@dataclass(frozen=True)
class Cosmos3EdgeConfig:
    checkpoint_dir: Path = DEFAULT_COSMOS3_EDGE_DIR
    device: str = "cuda"
    dtype: str = "bfloat16"
    hidden_layer: int = 20
    num_frames: int = 16


class Cosmos3EdgeExtractor(nn.Module):
    """Wraps Cosmos3OmniPipeline to extract Layer-20 spatiotemporal hidden representations."""

    def __init__(self, config: Cosmos3EdgeConfig | None = None) -> None:
        super().__init__()
        self.config = config or Cosmos3EdgeConfig()
        self.device = torch.device(self.config.device)
        self.dtype = torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float32

        self.pipe = Cosmos3OmniPipeline.from_pretrained(
            str(self.config.checkpoint_dir),
            dtype=self.dtype,
        ).to(self.device)

        # Freeze pipeline weights
        self.pipe.transformer.eval()
        self.pipe.vae.eval()
        for p in self.pipe.transformer.parameters():
            p.requires_grad_(False)
        for p in self.pipe.vae.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def extract_features(
        self,
        image_or_video: Any,
        prompt: str = "take cube out of box",
        num_frames: int | None = None,
    ) -> torch.Tensor:
        """Extract Layer-20 hidden states [N, 2048] from Cosmos3-Edge."""
        target_frames = num_frames or self.config.num_frames
        captured = []

        def hook_fn(module: nn.Module, inp: Any, out: tuple[torch.Tensor, torch.Tensor]) -> None:
            und_seq, gen_seq = out
            # Concat understanding sequence and generation sequence
            full_seq = torch.cat([und_seq, gen_seq], dim=0)
            captured.append(full_seq.detach())

        handle = self.pipe.transformer.layers[self.config.hidden_layer].register_forward_hook(hook_fn)

        if isinstance(image_or_video, torch.Tensor):
            # Convert tensor [B, 3, T, H, W] to PIL images if needed
            if image_or_video.ndim == 5:
                # [1, 3, 5, H, W]
                frames = [
                    Image.fromarray(image_or_video[0, :, i].permute(1, 2, 0).cpu().numpy().astype("uint8"))
                    for i in range(image_or_video.shape[2])
                ]
                self.pipe(
                    prompt=prompt,
                    video=frames,
                    num_frames=target_frames,
                    num_inference_steps=1,
                    guidance_scale=1.0,
                )
            else:
                img = Image.fromarray(image_or_video[0].permute(1, 2, 0).cpu().numpy().astype("uint8"))
                self.pipe(
                    prompt=prompt,
                    image=img,
                    num_frames=target_frames,
                    num_inference_steps=1,
                    guidance_scale=1.0,
                )
        else:
            self.pipe(
                prompt=prompt,
                image=image_or_video,
                num_frames=target_frames,
                num_inference_steps=1,
                guidance_scale=1.0,
            )

        handle.remove()
        if not captured:
            raise RuntimeError("Layer-20 forward hook did not capture any hidden states")
        return captured[0]
