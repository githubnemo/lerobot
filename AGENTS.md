This file provides guidance to AI agents when working with code in this repository.

> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
```

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                 # All tests
DEVICE=cuda make test-end-to-end                      # All E2E tests
pre-commit run --all-files                           # Lint + format (ruff, typos, bandit, etc.)
```

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` options in `configs.py`; `make_env()` / `make_task_env()` in `factory.py` build the envs. A config's `sim_settings()` maps its options to its `lerobot.sim.integrations.<type>` module's `env_fns()` kwargs.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (one tier per extra, selected by pytest mark, every PR), `full_tests.yml` (all extras + E2E, GPU job runs `-m gpu`, post-approval), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **`docs/video_vam_*.md`** — Video-VAM research (this fork). Start at `docs/video_vam_index.md`; current state in `docs/video_vam_status.md`, all numbers in `docs/video_vam_leaderboard.md`, chronological log in `docs/video_vam_research_diary.md`. Superseded docs in `docs/archive/`.
- **`scripts/video_vam/`** — Video-VAM tooling. Launch experiments via `run_experiment.sh <preset>` (presets in `scripts/video_vam/presets/`); see `docs/video_vam_scripts.md`. Do not add new one-off `run_*.sh` wrappers; add a preset instead.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`) and CI (`Dockerfile.internal`).
- **`benchmarks/`** — Performance benchmarking scripts.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`, `CONTRIBUTING.md` & `README.md` (general information).

## Notes

- **Mypy**: one ruleset (`[tool.mypy]` in `pyproject.toml`) checks the paths in its `files` setting, except `lerobot.rl`, the vendored `molmoact2_hf_model` and the generated `*_pb2` modules. Run it with `pre-commit run mypy --all-files`. Add type annotations when modifying code.
- **Imports**: prefer top-level imports; relative (`from .sibling import X`) across sibling files within a module, absolute (`from lerobot.module import X`) across modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`, see `pyproject.toml`). Guard optional imports with `TYPE_CHECKING or _foo_available` at module top + a `require_package(...)` check at use time. Reuse the `_foo_available` flags in `utils/import_utils.py`; don't call `is_package_available`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).
- **Test marks**: Fast Tests runs one tier per extra (`-m base|dataset|hardware|viz`) and the GPU job runs `-m gpu`. A test that needs the `dataset`, `hardware` or `viz` extra must be marked (`pytestmark = pytest.mark.dataset`), otherwise it only runs in `base` and skips there. A test that uses CUDA outside `tests/policies` and without `@require_cuda` needs `@pytest.mark.gpu`. Details in CONTRIBUTING.md, "Running Tests".

---

## Agent Delegation & Model Conventions

Parent-model tokens should be used primarily for judgment, architecture, and review.

- **Subagent Model**: Always use **Gemini 4 Hydrogen Priority** (`Regulaido Gateway/gemini-4-hydrogen-priority`) as the subagent model.
- **Exploration (Parallel)**: Delegate broad codebase exploration to one or more parallel explore subagents. Provide precise briefs requesting relevant paths, symbols, data flows, and constraints.
- **Design & Architecture**: Decide the solution at the orchestrator/parent level. Define affected files, interfaces, behavior, and invariants.
- **Implementation**: Delegate multi-file or substantial edits with an explicit implementation specification. Review the resulting diff and resume the same subagent for corrections.
- **Small Changes**: Make trivial, low-risk edits directly when delegation overhead is greater than the work.
- **Keep Output Concise**: Keep user-facing messages compact and outcome-first. Do not paste entire files or redundant command outputs.

---

## Search & Tool Discipline

- **Bounded Searches**: When using `grep`, **ALWAYS** provide `include_pattern` (e.g. `src/lerobot/**/*.py`, `tests/**/*.py`). Never run unbounded regex searches across the entire repo for generic terms.
- Prefer `find_path` for locating files by filename before searching file contents.
- Delegate broad, exploratory searches to an exploration subagent so raw tool matches do not flood the primary context window.

---

## Scope & Autonomy

- Implement explicit, low-risk requests completely without unnecessary confirmation.
- Ask questions only when ambiguity materially affects behavior, scope, robot hardware safety, or destructive actions.
- Use the minimal viable interpretation when ambiguity is minor.

---

## Communication Modes

If user prompt ends with:

- `GATHER`: Gather information only. Use explore subagents for broad investigation. Report relevant files, symbols, flows, and conventions. Do not recommend or implement a solution.
- `EXECUTE`: Provide a short list of possible approaches. Briefly elaborate the three strongest options. Select one and proceed when the choice is clear; ask only if the decision is materially the user's to make.

---

## Git Hygiene & Multi-Host Workflow (`abakus` / `sahni`)

- **Branches**: New branch names use `feat-ABC` or `fix-ABC` (avoid slashes if possible).
- **Commits**: Never amend commits; create follow-up commits. Do not commit or push unless explicitly requested.
- **Hosts & Remotes**:
  - **`abakus`** (`/home/anton/lerobot-video-vam`, `.venv`, RTX 4090 24 GB): primary GPU training & RPC inference host; holds push access to `origin` (`git@github.com:githubnemo/lerobot.git`). When pushing from Mac, push to `abakus` and fast-forward/push `origin` from `abakus`.
  - **`sahni`** (`~/code/hf/lerobot/githubnemo`, `/home/nemo/envs/lerobot`, RTX 3050 8 GB): physical SO-101 robot (`shabby` on `/dev/ttyACM0`, wrist cam `/dev/video0`). Always check `ps -ef | grep -E "run_mac_vam_rpc|lerobot-rollout"` before modifying git state or touching hardware on `sahni`.
  - **Git LFS `.stl` quirk**: 15 `src/lerobot/envs/so101_assets/assets/*.stl` files were historically committed as raw blobs under `.gitattributes` `filter=lfs`. If branch sync on `sahni` marks them modified (or inside `firejail` where `git-lfs` and `~/.ssh` are unavailable), clear them with `git -c filter.lfs.smudge= -c filter.lfs.process= -c filter.lfs.clean= -c filter.lfs.required=false checkout -- src/lerobot/envs/so101_assets/assets/`.
- **Physical Eval Defaults (`run_mac_vam_rpc.py` / `eval_cosmos3_best.sh`)**:
  - Pass `--no-rpc` to run inference directly in-process on the local GPU without SSH tunnels or `rpc_server.py` (works inside `firejail` too).
  - For 30 Hz `Orellius/so101_sort_cubes_no_top` policies, always run with `--camera-fps 30 --fps 30` and an in-distribution start pose (`--home-pose -0.3 -86.7 88.2 71.1 -53.4 17.7`).
  - Keep checkpoint configs portable across `/home/anton` and `/home/nemo` (`_resolve_cosmos3_checkpoint_path` checks `~/.cache/video-vam/cosmos3-edge`, `$HF_HOME/hub`, `~/.cache/huggingface/hub`, and `~/.cache/pysandbox-lerobot/huggingface/hub`).

---

## Video-VAM & SO-101 Real-Robot Post-Training Goals (Nemo & Anton)

1. **Real-Robot First Philosophy (Sim is for Validation & Algorithm Testing Only)**:
   - All methods must work on the **physical SO-101 robot arm with a wrist-only camera** (`observation.images.front`) and 6-D joint state (`observation.state`).
   - Do **not** rely on simulation-only privileged state (`cube_pos`, `dist_to_cube`) or large-scale sim-to-real pretraining; train on real robot data (`Orellius/so101_sort_cubes_no_top`) and realistic episode budgets (~15–70 episodes).
   - The MuJoCo digital twin (`lerobot/SO101CubeBox-v0`) is strictly a controllable testbed to benchmark offline-to-online correlation, sparse-reward RL post-training, and learned reward detectors before/alongside physical robot runs.
2. **Consistent Validation Loss & Offline-to-Physical Correlation**:
   - Track validation loss and **First-5 Action RMSE** across all training runs on the exact same held-out real episodes (`68..72` of `Orellius/so101_sort_cubes_no_top`).
   - Verify that offline First-5 RMSE correlates with actual closed-loop picking and tracking performance.
3. **PEFT (LoRA / DoRA) on Video Backbones**:
   - Use LoRA / DoRA on `Cosmos 3 Edge` (and `FLUX 3 Action`) for fast training, higher batch sizes (`BS 16`), and improved visual alignment over frozen-backbone extraction.
   - Upload Frozen, DoRA, and LoRA checkpoints to separate Hugging Face repositories for clean benchmarking.
4. **EXPO Residual RL & Score-Centered / Dr. GRPO Post-Training under Sparse Rewards**:
   - Keep the pretrained video backbone (`Cosmos 3 Edge LoRA`) and flow-matching decoder (`SmolExpert`) **100% frozen** during online/post-training to prevent cross-attention overfitting on small rollout buffers.
   - Train a lightweight **EXPO Residual Actor ($\Delta a_\phi$) + Concave Critic ($Q_\psi$)** with $3 \times 4$ spatial token pooling and **Dr. GRPO / Score-Centered advantages** (mean-centered without $\div \sigma$ explosion).
   - Validate post-training using **pure sparse rewards (`reward_mode="sparse"`, `0/1`)** and using **only a learned reward detector** (zero ground-truth reward from the simulator).
5. **Multi-Frame Sequence Success Detector (Replacing Single-Frame ResNet-50)**:
   - Single-frame ResNet-50 reward classifiers glitch on wrist-camera views (hallucinating `reward = 1.0` whenever the open gripper looks at the cardboard box).
   - Train and validate a **multi-frame temporal sequence + proprioception (`agent_pos`) success detector** that observes the causal grasp-carry-release transition into the box:
     - Train on real dataset episodes (`0..67`) and evaluate on held-out episodes (`68..72`), including hard negatives (looking at the box without holding/dropping the cube).
     - Train the same sequence detector architecture on a realistic small set of simulated episodes and run RL post-training driven **100% by the learned sequence detector** (no ground-truth environment reward).
