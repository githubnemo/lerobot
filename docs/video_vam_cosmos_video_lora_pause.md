# Cosmos video-LoRA pause / handoff

Paused 2026-08-28 at approximately 12:51 CEST by the user to use the trained
adapters as the Cosmos backbone for the next cache/SmolExpert experiment.

## Frozen adapter snapshot

- Best: `/home/anton/.cache/video-vam/runs/cosmos2b-video-lora-20260828/paused-step6000/best_lora.safetensors`
- Best provenance: the adjacent `best_lora.json`; validation video loss `0.23523374549812365` (0.23523) at step `6000`, rank `16`, alpha `16`.
- The same best files are also copied at `/home/anton/.cache/video-vam/runs/cosmos2b-video-lora-20260828/best_lora.safetensors`.
- `last_lora.safetensors` and `last_lora.json` are also retained. The last adapter save was the 6k save (which was also best); training continued to approximately step 6400 without another validation, so `last_lora` may be later than the best snapshot.
- The paused and root best adapter files currently have identical SHA256 hashes. Keep both original adapter snapshots; do not delete them.

## What was running

The stopped process no longer exists, so the command below is the exact expanded
trainer invocation recorded by the old queue script and adapter provenance (the
queue had supplied these arguments to the process):

```bash
/home/anton/lerobot-video-vam/.venv/bin/python -m scripts.video_vam.train_cosmos_video_lora \
  --manifest /home/anton/.cache/video-vam/cosmos-train0-31-stride3-sigma80-prefix-pool2/manifest.json \
  --val-manifest /home/anton/.cache/video-vam/cosmos-rehearsal-stride20-sigma80-prefix-pool2/manifest.json \
  --split /home/anton/.cache/video-vam/splits/rehearsal-stride20.json \
  --dataset-root /home/anton/.cache/video-vam/cube-out-of-box-dataset \
  --backbone-checkpoint /home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt \
  --tokenizer /home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth \
  --prompt /home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-t5-11b.safetensors \
  --output-dir /home/anton/.cache/video-vam/runs/cosmos2b-video-lora-20260828 \
  --latent-cache-dir /home/anton/.cache/video-vam/runs/cosmos2b-video-lora-20260828/latent-cache \
  --train-episodes 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
  --max-steps 500000 --max-hours 12 --batch-size 1 --val-every 1000 --save-every 1000 \
  --patience 10 --min-delta 0 --lora-rank 16 --lora-alpha 16 --lora-lr 1e-4 \
  --weight-decay 0.1 --grad-clip 1 --eval-sigma 80 --seed 0 \
  --wandb-project video-vam-world2action --run-name cosmos2b-video-lora-20260828 \
  --overwrite
```

## Resume warning

Optimizer and scheduler state were **not saved**. A true optimizer resume is
therefore not possible. If continuing later, load the frozen adapters above and
start a new Adam/AdamW run, or add optimizer and scheduler checkpointing before
attempting a resume. Do not use `--overwrite` blindly on the adapter directory:
it can replace the preserved adapter artifacts. The old queue is marked paused
at `/home/anton/.cache/video-vam/runs/cosmos2b-video-lora-20260828/PAUSED` and
its script checks that marker so it cannot restart LoRA on top of this handoff.

## Latent cache

The full-clip VAE latent cache is already populated at
`$LORA_RUN/latent-cache`, where `LORA_RUN` is
`/home/anton/.cache/video-vam/runs/cosmos2b-video-lora-20260828`. Keep it for any
future new Adam run; it is independent of the LoRA adapter weights.

## Next experiment

The queue script `scripts/video_vam/run_videolora_smolexpert_from_best.sh`
consumes the best adapters at extraction time with the generic checkpoint and
records the adapter SHA256/rank/alpha/step in each feature-cache manifest. It
uses `--vae-input-mode observed_prefix --context-transform pool2 --sigma 80
--seed 0` and the default Cosmos hidden layer 20. An actual fused checkpoint,
when produced, is `/home/anton/.cache/video-vam/runs/cosmos2b-video-lora-20260828/fused-step6000.pt`
with adjacent JSON provenance; the original adapters remain the canonical audit
source for the cache build.
