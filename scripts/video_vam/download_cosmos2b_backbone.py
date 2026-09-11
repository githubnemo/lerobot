from pathlib import Path

from huggingface_hub import hf_hub_download

target_dir = Path("/home/anton/lerobot-video-vam/outputs/models/cosmos2b")
target_dir.mkdir(parents=True, exist_ok=True)

repo_id = "jonpai/mimic-video"
files = ["video_backbone/v2w_pretrained_cosmos.pt", "video_backbone/tokenizer/tokenizer.pth"]

print(f"Downloading Cosmos 2B backbone files to permanent directory: {target_dir}...", flush=True)

for rel_path in files:
    filename = Path(rel_path).name
    dest_path = target_dir / filename
    if dest_path.exists() and dest_path.stat().st_size > 1000000:
        print(f"File already exists: {dest_path} ({dest_path.stat().st_size / 1e6:.1f} MB)", flush=True)
        continue
    print(f"Downloading {rel_path}...", flush=True)
    downloaded = hf_hub_download(
        repo_id=repo_id, filename=rel_path, local_dir=str(target_dir), local_dir_use_symlinks=False
    )
    print(f"Downloaded to {downloaded}", flush=True)

# Also link to legacy path if code expects it
legacy_dir = Path("/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone")
legacy_dir.mkdir(parents=True, exist_ok=True)
legacy_pt = legacy_dir / "v2w_pretrained_cosmos.pt"
primary_pt = target_dir / "video_backbone" / "v2w_pretrained_cosmos.pt"
if not primary_pt.exists():
    primary_pt = target_dir / "v2w_pretrained_cosmos.pt"

if primary_pt.exists() and not legacy_pt.exists():
    legacy_pt.symlink_to(primary_pt)
    print(f"Created symlink: {legacy_pt} -> {primary_pt}", flush=True)

print("Cosmos 2B backbone download process complete!", flush=True)
