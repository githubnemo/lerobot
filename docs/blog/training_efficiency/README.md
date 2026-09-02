# Decoder training-efficiency figure

Reproduce the cube-out-of-box RMSE-vs-steps / RMSE-vs-hours figure.

```bash
cd /home/anton/lerobot-video-vam/docs/blog/training_efficiency
python3 -m pip install matplotlib
python3 plot_training_efficiency.py
```

Writes `vam-smolexpert-training-efficiency.png` next to the script.

`data/eff_*.json` is the plotted series:

| File               | Run                                                              | Best                     |
| ------------------ | ---------------------------------------------------------------- | ------------------------ |
| `eff_lora.json`    | Cosmos video-LoRA + SmolExpert (`ixzworl4`)                      | 13.06° at 38k / 1.91 h   |
| `eff_cosmos.json`  | Generic Cosmos pool2 + SmolExpert (`aqwcei5u`)                   | 13.81° at 36k / 1.67 h   |
| `eff_ltx.json`     | LTX unpooled + SmolExpert (`3rypmjug`)                           | 13.84° at 45k / 2.99 h   |
| `eff_smolvla.json` | SmolVLA train-only (`smolvla-trainonly-20260826`)                | 14.93° at 29.2k / 1.45 h |
| `eff_statet2.json` | Cosmos video-LoRA `state_t=2` unpooled + SmolExpert (`jri7vehq`) | 13.74° at 27k / 0.77 h   |

VAM points are W&B `val_aggregate_rmse_deg` every 1k steps. SmolVLA had `eval_steps=0`; those five points are frozen-protocol RMSE on the same 88 val windows.
