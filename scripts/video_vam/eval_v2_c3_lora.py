#!/usr/bin/env python3
"""Evaluate Cosmos 3 Edge LoRA SmolExpert on V2 Held-Out (episodes 90..99)."""

import json
import math
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.vam.cosmos3_features import Cosmos3ExtractorConfig, Cosmos3FeatureExtractor
from lerobot.policies.vam.smol_expert import SmolExpertActionDecoder, SmolVLANormalizer

REPO_ROOT = Path("/home/anton/lerobot-video-vam")
V2_ROOT = Path(
    "/home/anton/.cache/huggingface/lerobot/hub/datasets--Orellius--cube_out_of_box_v2/snapshots/5d0325cc1412f4774223a0beb528958108814962"
)
ANCHOR_MANIFEST = Path("/home/anton/.cache/video-vam/flux2-klein-scale100-cache/eval2/manifest.json")
OUTPUT_MATRIX = REPO_ROOT / "outputs/evaluation/protocol1_1_dual_eval_matrix.json"
LORA_PATH = REPO_ROOT / "outputs/train/cosmos3-edge-video-lora/best_lora.safetensors"
POLICY_DIR = REPO_ROOT / "outputs/train/cosmos3-edge-lora-smolexpert"

device = torch.device("cuda")
anchor_data = json.loads(ANCHOR_MANIFEST.read_text())["entries"]
ds = LeRobotDataset(
    "Orellius/cube_out_of_box_v2",
    root=V2_ROOT,
    episodes=list(range(90, 100)),
    delta_timestamps={"action": [i / 10 for i in range(30)]},
    return_uint8=True,
    download_videos=False,
)

frame_to_idx = {int(ds.reader.hf_dataset[i]["index"]): i for i in range(len(ds))}
frames = []
states, targets, pads = [], [], []

for entry in anchor_data:
    fidx = int(entry["frame_index"])
    if fidx not in frame_to_idx:
        continue
    w_frames = [ds[frame_to_idx[f]]["observation.images.front"] for f in range(fidx - 4, fidx + 1)]
    cam = torch.stack(w_frames, dim=1).unsqueeze(0).contiguous()
    row = ds[frame_to_idx[fidx]]
    frames.append(cam)
    states.append(row["observation.state"].float())
    targets.append(row["action"].float())
    pads.append(row["action_is_pad"].bool())

print(f"Prepared {len(states)} evaluation anchors from V2 held-out.", flush=True)

c3_checkpoint = Path("/home/anton/.cache/video-vam/cosmos3-edge")
c3_cfg = Cosmos3ExtractorConfig(
    backbone_name="cosmos3-edge",
    checkpoint_path=c3_checkpoint,
    hidden_layers=(20,),
    device=str(device),
    dtype="bfloat16",
    fps=10.0,
    base_fps=24.0,
    prompt="take cube out of box",
    lora_checkpoint=LORA_PATH,
)
c3_extractor = Cosmos3FeatureExtractor(c3_cfg)

contexts = []
t0 = time.time()
for i, frame in enumerate(frames):
    with torch.no_grad():
        tks = c3_extractor.extract(frame.to(device))
        contexts.append(tks.features.cpu())
print(f"Extracted features in {time.time() - t0:.1f}s", flush=True)

norm_sd = load_file(str(POLICY_DIR / "normalizer.safetensors"))
normalizer = SmolVLANormalizer(
    state_mean=norm_sd["state_mean"],
    state_std=norm_sd["state_std"],
    action_mean=norm_sd["action_mean"],
    action_std=norm_sd["action_std"],
    eps=float(norm_sd.get("eps", 1e-8)),
    source_split=str(norm_sd.get("source_split", "train")),
)

decoder = SmolExpertActionDecoder.from_pretrained(
    "lerobot/smolvla_base",
    normalizer=normalizer,
    device=device,
    num_steps=10,
    input_channels=2048,
)
sd = load_file(str(POLICY_DIR / "best.safetensors"))
cleaned = {k.replace("model.", ""): v for k, v in sd.items()}
decoder.load_state_dict(cleaned)
decoder.eval()

total_sq_err, valid_count = 0.0, 0
total_arm_sq_err, total_grip_sq_err = 0.0, 0.0
h1_sq_err, h1_count = 0.0, 0
first5_sq_err, first5_count = 0.0, 0
for i, (ctx, st, tgt, pad) in enumerate(zip(contexts, states, targets, pads)):
    st_in = st[None, None].to(device=device, dtype=torch.float32)
    ctx_in = ctx.to(device=device, dtype=torch.bfloat16)
    pred = decoder.sample_actions(st_in, ctx_in, seed=1000 + i)[0].float().cpu()
    mask = ~pad
    err = (pred - tgt) ** 2
    for t in range(30):
        if mask[t]:
            total_sq_err += err[t].sum().item()
            total_arm_sq_err += err[t, :5].sum().item()
            total_grip_sq_err += err[t, 5].item()
            valid_count += 6
            if t == 0:
                h1_sq_err += err[t].sum().item()
                h1_count += 6
            if t < 5:
                first5_sq_err += err[t].sum().item()
                first5_count += 6

metrics = {
    "rmse": round(math.sqrt(total_sq_err / max(1, valid_count)), 2),
    "arm_rmse": round(math.sqrt(total_arm_sq_err / max(1, valid_count * (5 / 6))), 2),
    "gripper_rmse": round(math.sqrt(total_grip_sq_err / max(1, valid_count * (1 / 6))), 2),
    "h1": round(math.sqrt(h1_sq_err / max(1, h1_count)), 2),
    "first5": round(math.sqrt(first5_sq_err / max(1, first5_count)), 2),
}
print("\n>>> Cosmos 3 Edge LoRA V2 Score:", metrics, flush=True)

matrix = json.loads(OUTPUT_MATRIX.read_text()) if OUTPUT_MATRIX.is_file() else {"results": {}}
matrix["results"]["Cosmos 3 Edge (Video-LoRA)"] = {
    "v1_rmse": 13.62,
    "v1_h1": 4.11,
    "v1_first5": 5.97,
    "v2_rmse": metrics["rmse"],
    "v2_arm_rmse": metrics["arm_rmse"],
    "v2_gripper_rmse": metrics["gripper_rmse"],
    "v2_h1": metrics["h1"],
    "v2_first5": metrics["first5"],
    "latency_ms": 80,
}
OUTPUT_MATRIX.write_text(json.dumps(matrix, indent=2))
print(f"Updated {OUTPUT_MATRIX} successfully!", flush=True)
