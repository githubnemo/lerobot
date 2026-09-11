import json

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.vam.action_rmse import EvaluationBatch, SmolVLABackend, evaluate_backend

device = torch.device("cuda")
chk1 = "/home/anton/lerobot-video-vam/outputs/train/cube_out_of_box_il_smolvla_train_only_stats_0_31_20260826_1hr/checkpoints/029200/pretrained_model"
chk2 = "/home/anton/lerobot-video-vam/outputs/train/cube_out_of_box_scale100_smolvla_1hr/checkpoints/025000/pretrained_model"

print("Loading SmolVLA v1 (29.2k)...", flush=True)
b1 = SmolVLABackend.from_pretrained(chk1, device=device)
print("Loading SmolVLA v2 (25k)...", flush=True)
b2 = SmolVLABackend.from_pretrained(chk2, device=device)


def make_eval_batches(dataset_name, episodes, revision=None):
    print(f"Loading {dataset_name} episodes {episodes}...", flush=True)
    kwargs = {
        "episodes": episodes,
        "delta_timestamps": {"action": [i / 10 for i in range(30)]},
        "download_videos": False,
        "video_backend": "torchcodec",
    }
    if revision:
        kwargs["revision"] = revision
    ds = LeRobotDataset(dataset_name, **kwargs)
    raw = ds.reader.hf_dataset
    anchors = []
    for idx in range(len(raw)):
        fidx = int(raw[idx]["frame_index"])
        if fidx >= 4 and (fidx - 4) % 20 == 0:
            anchors.append(idx)
    batches = []
    for aidx in anchors:
        row = ds[aidx]
        batches.append(
            EvaluationBatch(
                sample_ids=(f"eval-{aidx}",),
                target_actions=row["action"].float().unsqueeze(0),
                action_is_pad=row["action_is_pad"].bool().unsqueeze(0),
                current_state=row["observation.state"].float().unsqueeze(0),
                observations=(
                    {k: v for k, v in row.items() if k.startswith("observation.")} | {"task": row["task"]},
                ),
            )
        )
    return batches, len(anchors)


# 1. v1 Eval Set (Original held-out episodes 32-39)
batches_v1, n_v1 = make_eval_batches(
    "hubnemo/cube_out_of_box_dataset",
    list(range(32, 40)),
    revision="243370c3c08bcbd860133c4a0d658ea7c1d2e77e",
)
print(f"v1 Eval Anchors: {n_v1}", flush=True)

# 2. v2 Eval Set (New held-out episodes 90-99)
batches_v2, n_v2 = make_eval_batches("Orellius/cube_out_of_box_v2", list(range(90, 100)))
print(f"v2 Eval Anchors: {n_v2}", flush=True)

print("\n--- EVALUATING ON V1 EVAL SET (Episodes 32-39) ---", flush=True)
res_v1_on_v1 = evaluate_backend(b1, batches_v1, seed=0)
res_v2_on_v1 = evaluate_backend(b2, batches_v1, seed=0)

print("\n--- EVALUATING ON V2 EVAL SET (Episodes 90-99) ---", flush=True)
res_v1_on_v2 = evaluate_backend(b1, batches_v2, seed=0)
res_v2_on_v2 = evaluate_backend(b2, batches_v2, seed=0)

final_results = {
    "v1_eval_set_episodes_32_39": {
        "smolvla_v1": round(res_v1_on_v1["aggregate_rmse_deg"], 2),
        "smolvla_v2": round(res_v2_on_v1["aggregate_rmse_deg"], 2),
    },
    "v2_eval_set_episodes_90_99": {
        "smolvla_v1": round(res_v1_on_v2["aggregate_rmse_deg"], 2),
        "smolvla_v2": round(res_v2_on_v2["aggregate_rmse_deg"], 2),
    },
}

out_path = "/home/anton/lerobot-video-vam/outputs/evaluation/comparison_v1_vs_v2.json"
with open(out_path, "w") as f:
    json.dump(final_results, f, indent=2)

print("\nFINAL SUMMARY MATRIX:", flush=True)
print(json.dumps(final_results, indent=2), flush=True)
