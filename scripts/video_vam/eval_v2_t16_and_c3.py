#!/usr/bin/env python3
"""Evaluate Cosmos 2B T=16 cond_frames and Cosmos 3 Edge und_seq on V2 Held-Out (episodes 90..99)."""

import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.vam.context_transform import apply_context_transform
from lerobot.policies.vam.cosmos3_features import Cosmos3ExtractorConfig, Cosmos3FeatureExtractor
from lerobot.policies.vam.cosmos_lora import merge_lora_file_into_base
from lerobot.policies.vam.cosmos_predict2_extractor import (
    CosmosPredict2Extractor,
    CosmosPredict2ExtractorConfig,
)
from lerobot.policies.vam.cosmos_prompt_embedding import load_prompt_embedding
from lerobot.policies.vam.smol_expert import SmolExpertActionDecoder, SmolVLANormalizer

REPO_ROOT = Path("/home/anton/lerobot-video-vam")
MIMIC_DIR = Path("/home/anton/.cache/video-vam/mimic-video-f2833903")
COSMOS2B_PT = MIMIC_DIR / "video_backbone/v2w_pretrained_cosmos.pt"
COSMOS2B_TOKENIZER = MIMIC_DIR / "video_backbone/tokenizer/tokenizer.pth"
PROMPT_PATH = Path("/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-t5-11b.safetensors")
VIDEO_LORA = REPO_ROOT / "outputs/train/cosmos-video-lora-step6000/best_lora.safetensors"
V2_ROOT = Path(
    "/home/anton/.cache/huggingface/lerobot/hub/datasets--Orellius--cube_out_of_box_v2/snapshots/5d0325cc1412f4774223a0beb528958108814962"
)
ANCHOR_MANIFEST = Path("/home/anton/.cache/video-vam/flux2-klein-scale100-cache/eval2/manifest.json")
OUTPUT_MATRIX = REPO_ROOT / "outputs/evaluation/protocol1_1_dual_eval_matrix.json"

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
frames_2b = []
states, targets, pads = [], [], []

for entry in anchor_data:
    fidx = int(entry["frame_index"])
    if fidx not in frame_to_idx:
        continue
    w_frames = [ds[frame_to_idx[f]]["observation.images.front"] for f in range(fidx - 4, fidx + 1)]
    cam_2b = torch.stack(w_frames, dim=1).unsqueeze(0).contiguous()  # [1, 3, 5, 480, 640]
    row = ds[frame_to_idx[fidx]]
    frames_2b.append(cam_2b)
    states.append(row["observation.state"].float())
    targets.append(row["action"].float())
    pads.append(row["action_is_pad"].bool())

print(f"Prepared {len(states)} evaluation anchors from V2 held-out.", flush=True)


def compute_metrics(decoder, contexts, states, targets, pads, seed=1000):
    total_sq_err, valid_count = 0.0, 0
    total_arm_sq_err, total_grip_sq_err = 0.0, 0.0
    h1_sq_err, h1_count = 0.0, 0
    first5_sq_err, first5_count = 0.0, 0
    for i, (ctx, st, tgt, pad) in enumerate(zip(contexts, states, targets, pads)):
        st_in = st[None, None].to(device=device, dtype=torch.float32)
        ctx_in = ctx.to(device=device, dtype=torch.bfloat16)
        pred = decoder.sample_actions(st_in, ctx_in, seed=seed + i)[0].float().cpu()
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
    return {
        "rmse": round(math.sqrt(total_sq_err / max(1, valid_count)), 2),
        "arm_rmse": round(math.sqrt(total_arm_sq_err / max(1, valid_count * (5 / 6))), 2),
        "gripper_rmse": round(math.sqrt(total_grip_sq_err / max(1, valid_count * (1 / 6))), 2),
        "h1": round(math.sqrt(h1_sq_err / max(1, h1_count)), 2),
        "first5": round(math.sqrt(first5_sq_err / max(1, first5_count)), 2),
    }


# ==============================================================================
# 1. EVALUATE COSMOS 2B T=16 COND_FRAMES (UNPOOLED)
# ==============================================================================
print("\n[+] Extracting & evaluating: Cosmos 2B T=16 cond_frames (unpooled)...", flush=True)
cfg = CosmosPredict2ExtractorConfig(
    checkpoint_path=COSMOS2B_PT,
    tokenizer_path=COSMOS2B_TOKENIZER,
    device=str(device),
    dtype="bfloat16",
    state_t=16,
    hidden_layer=20,
    high_noise_sigma=80.0,
    seed=0,
    vae_input_mode="observed_prefix",
)
extractor = CosmosPredict2Extractor(cfg)
merge_lora_file_into_base(extractor.backbone, VIDEO_LORA)
prompt_emb = load_prompt_embedding(PROMPT_PATH).embedding.to(device)

contexts_t16 = []
for i, frame in enumerate(frames_2b):
    with torch.no_grad():
        res = extractor.extract(frame.to(device), prompt_emb, noise_seed=i)
        cond = apply_context_transform(res.tokens, "cond_frames")  # [1, 2400, 2048]
        contexts_t16.append(cond.cpu())

run_dir = REPO_ROOT / "outputs/train/cosmos2b-t16-condframes-smolexpert"
norm_sd = load_file(str(run_dir / "normalizer.safetensors"))
normalizer = SmolVLANormalizer(
    state_mean=norm_sd["state_mean"],
    state_std=norm_sd["state_std"],
    action_mean=norm_sd["action_mean"],
    action_std=norm_sd["action_std"],
    eps=float(norm_sd.get("eps", 1e-8)),
    source_split=str(norm_sd.get("source_split", "train")),
)

decoder_t16 = SmolExpertActionDecoder.from_pretrained(
    "lerobot/smolvla_base",
    normalizer=normalizer,
    device=device,
    num_steps=10,
    input_channels=2048,
)
sd = load_file(str(run_dir / "best.safetensors"))
cleaned = {k.replace("model.", ""): v for k, v in sd.items()}
decoder_t16.load_state_dict(cleaned)
decoder_t16.eval()

metrics_t16 = compute_metrics(decoder_t16, contexts_t16, states, targets, pads)
print("Cosmos 2B T=16 cond_frames V2 Score:", metrics_t16, flush=True)

del extractor, decoder_t16
torch.cuda.empty_cache()

# ==============================================================================
# 2. EVALUATE COSMOS 3 EDGE (und_seq, 600 tokens)
# ==============================================================================
print("\n[+] Extracting & evaluating: Cosmos 3 Edge und_seq (600 tokens)...", flush=True)
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
)
c3_extractor = Cosmos3FeatureExtractor(c3_cfg)

contexts_c3 = []
for i, frame in enumerate(frames_2b):
    with torch.no_grad():
        tks = c3_extractor.extract(frame.to(device))  # [1, 600, 2048]
        contexts_c3.append(tks.features.cpu())

run_dir_c3 = REPO_ROOT / "outputs/train/cosmos3-edge-undseq-smolexpert"
norm_sd_c3 = load_file(str(run_dir_c3 / "normalizer.safetensors"))
normalizer_c3 = SmolVLANormalizer(
    state_mean=norm_sd_c3["state_mean"],
    state_std=norm_sd_c3["state_std"],
    action_mean=norm_sd_c3["action_mean"],
    action_std=norm_sd_c3["action_std"],
    eps=float(norm_sd_c3.get("eps", 1e-8)),
    source_split=str(norm_sd_c3.get("source_split", "train")),
)

decoder_c3 = SmolExpertActionDecoder.from_pretrained(
    "lerobot/smolvla_base",
    normalizer=normalizer_c3,
    device=device,
    num_steps=10,
    input_channels=2048,
)
sd_c3 = load_file(str(run_dir_c3 / "best.safetensors"))
cleaned_c3 = {k.replace("model.", ""): v for k, v in sd_c3.items()}
decoder_c3.load_state_dict(cleaned_c3)
decoder_c3.eval()

metrics_c3 = compute_metrics(decoder_c3, contexts_c3, states, targets, pads)
print("Cosmos 3 Edge und_seq V2 Score:", metrics_c3, flush=True)

# Update matrix JSON
matrix = json.loads(OUTPUT_MATRIX.read_text()) if OUTPUT_MATRIX.is_file() else {"results": {}}
matrix["results"]["Cosmos 2B T=16 cond_frames (Oracle)"] = {
    "v1_rmse": 13.13,
    "v1_h1": 4.23,
    "v1_first5": 5.95,
    "v2_rmse": metrics_t16["rmse"],
    "v2_h1": metrics_t16["h1"],
    "v2_first5": metrics_t16["first5"],
    "latency_ms": 1200,
}
matrix["results"]["Cosmos 3 Edge (und_seq)"] = {
    "v1_rmse": 14.25,
    "v1_h1": 4.92,
    "v1_first5": 6.57,
    "v2_rmse": metrics_c3["rmse"],
    "v2_h1": metrics_c3["h1"],
    "v2_first5": metrics_c3["first5"],
    "latency_ms": 80,
}
OUTPUT_MATRIX.write_text(json.dumps(matrix, indent=2))
print(f"\nUpdated {OUTPUT_MATRIX} successfully!", flush=True)
