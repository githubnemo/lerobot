#!/usr/bin/env python3
"""Fetch val RMSE histories for the head comparison (World2Action vs SmolExpert) and save JSON."""

import json
from pathlib import Path

import wandb

RUNS = {"world2action": "kn787l8b", "smolexpert": "oy0an3ag"}
OUT = Path(__file__).resolve().parent / "data"
api = wandb.Api()
for name, rid in RUNS.items():
    run = api.run(f"hubnemo-hugging-face/video-vam-world2action/{rid}")
    rows = []
    for row in run.scan_history(
        keys=["_step", "_runtime"]
        + (["val_aggregate_rmse"] if name == "world2action" else ["val_aggregate_rmse_deg"])
    ):
        v = row.get("val_aggregate_rmse_deg") or row.get("val_aggregate_rmse")
        if v is None:
            continue
        rows.append({"step": row["_step"], "rmse": v, "hours": row["_runtime"] / 3600.0})
    print(name, rid, len(rows), "points", "best", min((r["rmse"] for r in rows), default=None))
    (OUT / f"head_{name}.json").write_text(json.dumps(rows))
