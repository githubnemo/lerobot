# Video-VAM Master Index & Navigation Guide

**Repository:** `lerobot-video-vam` (Remote: `/home/anton/lerobot-video-vam` on `abakus`)
**Status:** Consolidated Master Index

---

## 1. Documentation Source of Truth Map

To maintain strict scientific and operational consistency, use the following authoritative sources of truth:

| Role / Domain                             | Authoritative Document                                                                        | Description                                                                                                                                                       |
| :---------------------------------------- | :-------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Overnight Execution & Retraining Plan** | [`docs/video_vam_execution_plan.md`](./video_vam_execution_plan.md)                           | Canonical operational sequence for retraining missing T=2 models and verified Protocol 1.0 queue execution.                                                       |
| **Current Priorities & Strategy**         | [`docs/video_vam_roadmap.md`](./video_vam_roadmap.md)                                         | Current implementation priorities, active research bets, and execution roadmap.                                                                                   |
| **Bugs, Validity & Correctness Audit**    | [`docs/video_vam_correctness_audit.md`](./video_vam_correctness_audit.md)                     | Exhaustive catalog of identified bugs, data leakage, pseudo-latent mocks, schedule mismatches, and evaluation invariants.                                         |
| **Metrics & Evaluation Protocol**         | [`docs/video_vam_action_rmse_protocol.md`](./video_vam_action_rmse_protocol.md)               | Frozen Protocol 1.0 specification: mixed-unit aggregate RMSE, horizons (H1 offset 0, first-5 $0..4$, full-30 $0..29$), masking rules, and deterministic sampling. |
| **Benchmark Results Ledger**              | [`docs/video_vam_cube_out_of_box_leaderboard.md`](./video_vam_cube_out_of_box_leaderboard.md) | Append-only results ledger tracking all validated runs and explicitly annotating retracted historical entries.                                                    |
| **Technical Synthesis**                   | [`docs/video_vam_technical_report.md`](./video_vam_technical_report.md)                       | Comprehensive architectural taxonomy, layer-targeting formulations, conditioning geometries, and policy benchmarks.                                               |
| **Historical Journey**                    | [`docs/video_vam_research_diary.md`](./video_vam_research_diary.md)                           | Chronological development log recording daily hypotheses, experiments, and debugging breakthroughs.                                                               |
| **Architectural Audit**                   | [`docs/ARCHITECTURAL_AUDIT_AND_ABSTRACTIONS.md`](./ARCHITECTURAL_AUDIT_AND_ABSTRACTIONS.md)   | Code audit detailing duplication, base abstractions, and modular refactoring blueprints.                                                                          |
| **Rollouts & Experiments**                | [`docs/EXPERIMENTS_OVERVIEW.md`](./EXPERIMENTS_OVERVIEW.md)                                   | Video rollout specifications (e.g. 14B 161-frame rollout) and multi-model visual benchmarks.                                                                      |
| **Artifacts & Model Storage Guide**       | [`docs/video_vam_artifacts.md`](./video_vam_artifacts.md)                                     | Authoritative storage rules, saved checkpoint inventory, and Hugging Face upload standards.                                                                       |
| **Developer / Agent Guide**               | [`AGENTS.md`](../AGENTS.md)                                                                   | Developer guidelines, CLI conventions, and mandatory coding rules.                                                                                                |

---

## 2. Canonical Workflows

### A. Feature Extraction & Caching

1. Always use the unified builder `scripts/video_vam/build_vam_feature_cache.py`.
2. Specify `--dataset-repo-id hubnemo/cube_out_of_box_dataset`, `--train-episodes 0-31`, and `--val-episodes 32-39`.
3. Ensures genuine causal VAE latent encoding, `action_is_pad` tracking, and Protocol 1.0 episode isolation via `split_guard.py`.

### B. SmolExpert Action Decoder Training

1. Canonical training entry point: `scripts/video_vam/train_smolexpert.py`.
2. Pass `--train-manifest` and `--val-manifest`.
3. Strictly uses train-only normalizer derivation, masks padded actions in flow matching loss, and evaluates global mixed-unit RMSE.

### C. Evaluation & Leaderboard Logging

1. Execute standardized evaluation via `scripts/video_vam/evaluate_action_rmse.py`.
2. Report Full-30 Mixed RMSE, Horizon-1 (H1 at offset 0), First-5 mean (offsets $0..4$), and per-joint degrees.
3. Append results to `docs/video_vam_cube_out_of_box_leaderboard.md` following the append-only convention.

---

## 3. Operational Conventions on `abakus`

- **Working Directory:** `/home/anton/lerobot-video-vam`.
- **Environment:** Always prefix execution with `uv run` (or activate `.venv`).
- **GPU Resource Management:** Source `scripts/video_vam/gpu_lock.sh` and acquire a GPU lock before long-running GPU processes.
- **Cache Storage:** Primary artifact cache root is `/home/anton/.cache/video-vam/`.
