import json
from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.vam.action_rmse import EvaluationBatch, SmolVLABackend, evaluate_backend

print("Loading dataset...", flush=True)
dataset = LeRobotDataset(
    "hubnemo/cube_out_of_box_dataset",
    revision="243370c3c08bcbd860133c4a0d658ea7c1d2e77e",
    episodes=list(range(32, 40)),
    delta_timestamps={"action": [i / 10 for i in range(30)]},
    download_videos=False,
    video_backend="torchcodec",
)
raw = dataset.reader.hf_dataset
anchors = []
for idx in range(len(raw)):
    row = raw[idx]
    fidx = int(row["frame_index"])
    if fidx >= 4 and (fidx - 4) % 20 == 0:
        anchors.append(idx)
print("Found", len(anchors), "evaluation anchors.", flush=True)

chk = "/home/anton/lerobot-video-vam/outputs/train/cube_out_of_box_scale100_smolvla_1hr/checkpoints/025000/pretrained_model"
backend = SmolVLABackend.from_pretrained(chk, device=torch.device("cuda"))

eval_batches = []
for aidx in anchors:
    row = dataset[aidx]
    target = row["action"].float().unsqueeze(0)
    pad = row["action_is_pad"].bool().unsqueeze(0)
    state = row["observation.state"].float().unsqueeze(0)
    obs = {k: v for k, v in row.items() if k.startswith("observation.")}
    obs["task"] = row["task"]
    eval_batches.append(
        EvaluationBatch(
            sample_ids=(f"anchor-{aidx}",),
            target_actions=target,
            action_is_pad=pad,
            current_state=state,
            observations=(obs,),
        )
    )

results = {}
for seed in (0, 1, 2):
    res = evaluate_backend(backend, eval_batches, seed=seed)
    agg = res["aggregate_rmse_deg"]
    print("SmolVLA v2 (25k) Seed", seed, "RMSE:", round(agg, 2), flush=True)
    results[f"seed_{seed}"] = res

out_file = Path("/home/anton/lerobot-video-vam/outputs/evaluation/smolvla_v2_25k_on_original_val.json")
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print("Saved to", out_file, flush=True)
