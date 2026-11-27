# Video-VAM documentation index

**Repo:** `lerobot-video-vam` (branch `feat-video-vam`; GPU host `abakus`: `/home/anton/lerobot-video-vam`)
**Question:** do pretrained video-diffusion backbones give better features for SO-101 manipulation policies than 2D VLA encoders?

Start with **[status](./video_vam_status.md)**. Every other document has one job; if two documents disagree, the owner in this table wins.

## Owners (single source of truth)

| Topic                                                                                          | Document                                                                                                                                                                                                           |
| :--------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Current state, open issues, next steps, taxonomy                                               | [`video_vam_status.md`](./video_vam_status.md)                                                                                                                                                                     |
| Goals & plans (PEFT blog, T=2 distillation, FLUX 3 Action, JEPA world model, research direction + post-training) | [`plan_peft_blog.md`](./plan_peft_blog.md), [`plan_t2_distillation.md`](./plan_t2_distillation.md), [`plan_flux3_action.md`](./plan_flux3_action.md), [`plan_jepa_world_model.md`](./plan_jepa_world_model.md), [`plan_research_direction.md`](./plan_research_direction.md) |
| **All numbers** (validity-tagged, append-only)                                                 | [`video_vam_leaderboard.md`](./video_vam_leaderboard.md)                                                                                                                                                           |
| Metric / split contract (Protocol 1.0, rev 1.1)                                                | [`video_vam_action_rmse_protocol.md`](./video_vam_action_rmse_protocol.md)                                                                                                                                         |
| Bug catalogue and invariants                                                                   | [`video_vam_correctness_audit.md`](./video_vam_correctness_audit.md)                                                                                                                                               |
| Chronological log (history, not current truth)                                                 | [`video_vam_research_diary.md`](./video_vam_research_diary.md)                                                                                                                                                     |
| Daily session traces (one file per day, `YYYY-MM-DD.md`)                                       | [`daily/README.md`](./daily/README.md)                                                                                                                                                                             |
| How to run things (presets, launcher, scripts)                                                 | [`video_vam_scripts.md`](./video_vam_scripts.md)                                                                                                                                                                   |
| Robot / RPC rollout and physical testing                                                       | [`video_vam_rollout.md`](./video_vam_rollout.md)                                                                                                                                                                   |
| Checkpoint inventory and artifact rules                                                        | [`video_vam_artifacts.md`](./video_vam_artifacts.md)                                                                                                                                                               |

## Reference material (stable, rarely changes)

| Topic                                    | Document                                                                                                                                                 |
| :--------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream mimic-video recipe & deviations | [`mimic_video_reference.md`](./mimic_video_reference.md)                                                                                                 |
| Cosmos-Predict2 extractor contract       | [`video_vam_cosmos_extractor.md`](./video_vam_cosmos_extractor.md)                                                                                       |
| LTX-2.5 extractor                        | [`video_vam_ltx_extractor.md`](./video_vam_ltx_extractor.md)                                                                                             |
| T5 prompt embedding artifact             | [`video_vam_prompt_embedding.md`](./video_vam_prompt_embedding.md)                                                                                       |
| World2Action decoder                     | [`video_vam_world2action.md`](./video_vam_world2action.md)                                                                                               |
| T=2 world expert design                  | [`video_vam_world_expert.md`](./video_vam_world_expert.md)                                                                                               |
| Policy/rollout code contract             | [`video_vam_rollout_path.md`](./video_vam_rollout_path.md)                                                                                               |
| Latency: Cosmos path / LTX path          | [`video_vam_latency_cosmos.md`](./video_vam_latency_cosmos.md), [`video_vam_latency_ltx.md`](./video_vam_latency_ltx.md)                                 |
| Connector literature / ablation          | [`video_vam_connector_survey.md`](./video_vam_connector_survey.md), [`video_vam_connector_ablation_report.md`](./video_vam_connector_ablation_report.md) |
| World-model usage comparison             | [`video_vam_world_model_usage_comparison.md`](./video_vam_world_model_usage_comparison.md)                                                               |
| Temporal consistency / RTC               | [`video_vam_temporal_consistency_report.md`](./video_vam_temporal_consistency_report.md)                                                                 |
| Video prediction preview harness         | [`video_vam_video_prediction_preview.md`](./video_vam_video_prediction_preview.md)                                                                       |
| RL notes (FLUX-mimic reconstruction)     | [`video_vam_rl_notes.md`](./video_vam_rl_notes.md)                                                                                                       |
| Training-efficiency figure               | [`blog/training_efficiency/README.md`](./blog/training_efficiency/README.md)                                                                             |

Reference docs describe contracts and measurements at the date they state. Numbers in them are **not** authoritative; the leaderboard is.

## Archive

[`archive/`](./archive/) holds superseded documents (old roadmap, technical report, experiments overview, 09-08 execution plan, German audit copy, LoRA pause note, old RPC/rollout/robot-plan docs, LoRA co-training design). Kept for provenance only; do not update them.

## Conventions

- Run on `abakus` with `uv run` (or `.venv/bin/python`); long jobs in named tmux sessions; never kill a PID whose cmdline starts with `tmux`.
- GPU jobs go through `scripts/video_vam/run_experiment.sh` (acquires `gpu_lock.sh`, logs to `outputs/logs/`).
- New results: append to the leaderboard with dataset + revision, split, seed, and validity tag, then add a diary entry. Never edit historical rows except to add a validity note.
