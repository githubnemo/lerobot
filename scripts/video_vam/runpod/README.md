# RunPod RTX 5090 Workflow (Reusable Network Volume)

## 1. Check live stock & balance

```bash
python3 scripts/video_vam/runpod/provision_5090.py stock
python3 scripts/video_vam/runpod/provision_5090.py status
```

## 2. Create reusable 200 GB Network Volume + launch RTX 5090 pod

```bash
python3 scripts/video_vam/runpod/provision_5090.py up \
  --name lerobot-vam-5090 \
  --volume lerobot-vam-vol \
  --volume-size 200 \
  --datacenter EU-RO-1
```

This automatically attaches `/workspace` (persisting backbones, datasets, `.venv`, and checkpoints across pod stops/restarts) and injects `HF_TOKEN`, `WANDB_API_KEY`, `HF_AUTO_UPLOAD=1`, and your SSH public key.

## 3. Bootstrap the pod (idempotent)

```bash
ssh root@<IP> -p <PORT> 'bash -s' < scripts/video_vam/runpod/bootstrap_pod.sh
```

## 4. Launch numbered experiments (auto-uploads finished models to Hugging Face Hub)

```bash
ssh root@<IP> -p <PORT>
tmux new -s vam_sweep
cd /workspace/lerobot
bash scripts/video_vam/experiments/exp001_c3_edge_joint_lora_bs64.sh
```

## 5. Stop or terminate pod (keeps Network Volume intact)

```bash
python3 scripts/video_vam/runpod/provision_5090.py stop --pod-id <POD_ID>
# or permanently delete pod while preserving /workspace volume:
python3 scripts/video_vam/runpod/provision_5090.py terminate --pod-id <POD_ID>
```
