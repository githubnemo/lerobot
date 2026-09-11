# Video-VAM Execution & Retraining Plan (2026-09-08)

**Target:** Controlled retraining of missing Cosmos 2B T=2 models, artifact preservation, and genuine non-placeholder queue execution under Protocol 1.0.
**Repository:** `/home/anton/lerobot-video-vam` on `abakus`
**Status:** Canonical operational plan for next agent / overnight execution.

---

## 1. Ground-Truth State as of 2026-09-08 22:30 CEST

1. **GPU & Jobs:** GPU is idle (RTX 4090, 0% util). Only resident process is the read-only SmolVLA RPC server on port 8766.
2. **Placeholder Queue Discovered:** `scripts/video_vam/run_full_autonomous_pipeline.sh` printed evaluation placeholders and overwrote `grand_evaluation_summary.json` with hard-coded numbers. It is disabled and must not be used.
3. **Artifact Inventory:**
   - **Preserved in `outputs/train/`:** SmolVLA v1 (step 29,200), SmolVLA v2 (step 25,000), Cosmos 2B Pool2 SmolExpert (`best`+`last`), LTX-2.5 Pool2 & Unpooled (`best`+`last`), Cosmos video-LoRA step 6,000, Cosmos 7B Protocol-1 SmolExpert.
   - **Missing Weights (Must Retrain):** Cosmos 2B T=2 undistilled head, direct-distilled student LoRA, direct-distilled SmolExpert head, teacher `cond_frames` reference head. (Historical logs exist; weight files absent).
   - **Preserved Foundation Models in `outputs/models/`:** Cosmos 2B backbone (`v2w_pretrained_cosmos.pt`, 3.9 GB) verified intact.
4. **Dataset Guards:**
   - **v1 (Canonical):** `hubnemo/cube_out_of_box_dataset` @ `243370c3c08bcbd860133c4a0d658ea7c1d2e77e` (40 eps, 6,536 frames). Verified.
   - **v2 (Quarantined):** `Orellius/cube_out_of_box_v2` blocked due to metadata mismatch (100 eps/12,163 frames declared vs 140 eps/15,998 rows present). No runs on v2 until resolved.

---

## 2. Overnight Execution Sequence (Strict Dependency Order)

All jobs must use verified dataset v1 and log to dedicated files under `outputs/train/`.

### Phase 1: Environment & Prerequisite Check (< 5 min)

- Ensure Cosmos 2B prompt embeddings exist or generate via `scripts/video_vam/generate_cosmos_prompt_embedding.py`.
- Check GPU availability (drain or coordinate port 8766 RPC server if full 24GB VRAM required).

### Phase 2: Undistilled T=2 Branch (~1.5 h)

1. **Feature Cache Extraction (T=2, unpooled):**
   - Script: `scripts/video_vam/build_cosmos_feature_cache.py`
   - Inputs: Pinned v1, `outputs/models/cosmos2b/video_backbone/v2w_pretrained_cosmos.pt`, adapter `outputs/train/cosmos-video-lora-step6000/best_lora.safetensors`.
   - Settings: `--state-t 2 --context-transform none --vae-input-mode observed_prefix --sigma 80 --seed 0`.
   - Output: `outputs/features/cosmos2b_videolora_t2_unpooled/{train,val}/`.
2. **SmolExpert Action Head Training:**
   - Script: `scripts/video_vam/train_smolexpert.py`
   - Settings: `--backbone cosmos --context-transform none --batch-size 8 --lr 1e-4 --max-steps 40000 --patience 10 --val-every 1000`.
   - Output: `outputs/train/cosmos2b-t2-undistilled-smolexpert/` (`best.safetensors`, `last.safetensors`, `run_manifest.json`).

### Phase 3: Teacher Cache & Direct Distillation Branch (~3 h)

1. **Teacher Feature Cache (T=16, unpooled `cond_frames` targets):**
   - Extract unpooled layer-20 features for the first 2 latent slots from the T=16 forward.
   - Output: `outputs/features/cosmos2b_teacher_condframes_unpooled/{train,val}/`.
2. **T=2 Student LoRA Distillation:**
   - Script: `scripts/video_vam/train_cosmos_t2_distillation.py`
   - Enforce: Canonical causal window `[t-4..t]` (5 RGB frames), blocks 0-19 LoRA, MSE + CosSim loss against teacher `cond_frames`.
   - Output: `outputs/train/cosmos2b-t2-direct-distilled-lora/best_lora.safetensors`.
3. **Distilled T=2 Feature Cache Extraction:**
   - Merge step-6000 video-LoRA + distilled student LoRA; extract T=2 unpooled features.
   - Output: `outputs/features/cosmos2b_distilled_t2_unpooled/{train,val}/`.
4. **Distilled SmolExpert Head Training:**
   - Script: `scripts/video_vam/train_smolexpert.py`
   - Output: `outputs/train/cosmos2b-t2-distilled-smolexpert/`.

### Phase 4: Standardized Protocol 1.0 Evaluation (~30 min)

- Evaluate both newly trained heads under exact Protocol 1.0 (88 fixed validation windows, masked global RMSE).
- Output: `outputs/evaluation/protocol1_cosmos2b_t2_comparison.json`.

---

## 3. Artifact Preservation & Hugging Face Checklist

1. **Local Integrity:**
   - Every completed run must write `best.safetensors`, `last.safetensors`, `normalizer.json`, and `run_manifest.json`.
   - Store inside `outputs/train/`, never in `/tmp/` or volatile cache roots.
2. **Hugging Face Hub Staging (When Authorized):**
   - **Category 1 (Heads):** `Orellius/cube-out-of-box-cosmos-t2-undistilled-smolexpert`, `Orellius/cube-out-of-box-cosmos-t2-distilled-smolexpert`.
   - **Category 2 (LoRA Adapters):** `Orellius/cube-out-of-box-cosmos-t2-distilled-student-lora`.
   - Upload using versioned staging script with verification of remote sha256.
