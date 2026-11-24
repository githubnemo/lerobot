#!/usr/bin/env python3
"""Manage a reusable RunPod Network Volume and RTX 5090 pod for Video-VAM / FLUX 3 / RL training.

Reads RUNPOD_API_KEY, HF_TOKEN, and WANDB_API_KEY from the environment (with fallback to
`./.env` and `~/.env.ml`). Never prints secret values.

Examples:
    python3 scripts/video_vam/runpod/provision_5090.py stock
    python3 scripts/video_vam/runpod/provision_5090.py status
    python3 scripts/video_vam/runpod/provision_5090.py create-volume --name lerobot-vam-vol --size 200 --datacenter EU-RO-1
    python3 scripts/video_vam/runpod/provision_5090.py up --name lerobot-vam-5090 --volume lerobot-vam-vol --datacenter EU-RO-1
    python3 scripts/video_vam/runpod/provision_5090.py stop --pod-id <POD_ID>
    python3 scripts/video_vam/runpod/provision_5090.py terminate --pod-id <POD_ID>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_URL = "https://api.runpod.io/graphql"
USER_AGENT = "lerobot-runpod-provisioner/1.0"
DEFAULT_GPU = "NVIDIA GeForce RTX 5090"
# Blackwell (sm_120) requires PyTorch built with CUDA 12.8+
DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
DEFAULT_VOLUME_NAME = "lerobot-vam-vol"
DEFAULT_VOLUME_SIZE_GB = 200
DEFAULT_CONTAINER_DISK_GB = 50
DEFAULT_DATACENTER = "EU-RO-1"


def _load_env_fallback(key: str) -> str | None:
    val = os.environ.get(key, "").strip()
    if val:
        return val
    for candidate in (Path(".env"), Path.home() / ".env.ml"):
        if not candidate.is_file():
            continue
        for line in candidate.read_text().splitlines():
            s = line.strip()
            if s.startswith("export "):
                s = s[len("export ") :].strip()
            if s.startswith(f"{key}="):
                raw = s.split("=", 1)[1].strip().strip("\"'")
                if raw:
                    return raw
    return None


def _require_api_key() -> str:
    key = _load_env_fallback("RUNPOD_API_KEY")
    if not key:
        raise SystemExit("ERROR: RUNPOD_API_KEY not found in env, ./.env, or ~/.env.ml")
    return key


def run_graphql(query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    api_key = _require_api_key()
    body: dict[str, Any] = {"query": query}
    if variables is not None:
        body["variables"] = variables
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"RunPod HTTP {err.code}: {detail}") from err
    payload = json.loads(raw)
    if payload.get("errors"):
        raise RuntimeError("RunPod GraphQL error:\n" + json.dumps(payload["errors"], indent=2))
    return payload["data"]


def _detect_ssh_pubkey(explicit_path: Path | None = None) -> str | None:
    candidates = (
        [explicit_path]
        if explicit_path is not None
        else [Path.home() / ".ssh/id_ed25519.pub", Path.home() / ".ssh/id_rsa.pub"]
    )
    for p in candidates:
        if p is not None and p.expanduser().is_file():
            return p.expanduser().read_text().strip()
    return None


def _extract_ssh_endpoint(pod: dict[str, Any]) -> tuple[str, int] | None:
    runtime = pod.get("runtime") or {}
    for port in runtime.get("ports") or []:
        if (
            port.get("privatePort") == 22
            and port.get("isIpPublic")
            and port.get("ip")
            and port.get("publicPort")
        ):
            return str(port["ip"]), int(port["publicPort"])
    return None


def cmd_stock(args: argparse.Namespace) -> int:
    q = """
    {
      dataCenters {
        id
        name
        location
        storageSupport
        gpuAvailability(input: { gpuCount: 1 }) {
          gpuTypeId
          gpuTypeDisplayName
          stockStatus
        }
      }
      gpuTypes {
        id
        displayName
        memoryInGb
        securePrice
        communityPrice
      }
    }
    """
    data = run_graphql(q)
    prices = {g["id"]: g for g in data.get("gpuTypes") or []}
    gpu_filter = (args.gpu or "").lower()

    rows = []
    for dc in data.get("dataCenters") or []:
        if not dc.get("storageSupport"):
            continue
        for g in dc.get("gpuAvailability") or []:
            gid = g.get("gpuTypeId") or ""
            gdisp = g.get("gpuTypeDisplayName") or gid
            stock = g.get("stockStatus")
            if not stock:
                continue
            if gpu_filter and gpu_filter not in gid.lower() and gpu_filter not in gdisp.lower():
                continue
            meta = prices.get(gid, {})
            is_eu = dc["id"].startswith("EU")
            rows.append((0 if is_eu else 1, dc["id"], dc["location"], gid, gdisp, meta, stock))

    rows.sort(key=lambda r: (r[0], r[1], r[3]))
    print(f"Secure Cloud + Network Volume stock matching {args.gpu!r}:")
    if not rows:
        print("  (no in-stock datacenters right now — try --gpu '' to list all GPUs)")
        return 0
    for _, dc_id, loc, gid, gdisp, meta, stock in rows:
        mem = meta.get("memoryInGb", "?")
        sec = meta.get("securePrice", "?")
        print(f"  {dc_id:10s} ({loc:22s})  {gdisp:22s} ({mem}GB)  ${sec}/hr  stock={stock}  [{gid}]")
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    q = """
    {
      myself {
        email
        clientBalance
        currentSpendPerHr
        networkVolumes { id name size dataCenterId }
        pods {
          id
          name
          desiredStatus
          costPerHr
          networkVolumeId
          machine { gpuDisplayName dataCenterId }
          runtime {
            uptimeInSeconds
            ports { ip isIpPublic privatePort publicPort type }
          }
        }
      }
    }
    """
    me = run_graphql(q)["myself"]
    print(
        f"Account: {me['email']} | Balance: ${me['clientBalance']:.2f} | "
        f"Current spend: ${me['currentSpendPerHr']:.3f}/hr"
    )
    print("\nNetwork Volumes:")
    vols = me.get("networkVolumes") or []
    if not vols:
        print("  (none)")
    for v in vols:
        print(f"  {v['id']}  name={v['name']}  size={v['size']}GB  dc={v['dataCenterId']}")

    print("\nPods:")
    pods = me.get("pods") or []
    if not pods:
        print("  (none)")
    for p in pods:
        mach = p.get("machine") or {}
        ssh_ep = _extract_ssh_endpoint(p)
        ssh_str = f"  ssh root@{ssh_ep[0]} -p {ssh_ep[1]}" if ssh_ep else ""
        print(
            f"  {p['id']}  name={p['name']}  status={p['desiredStatus']}  "
            f"gpu={mach.get('gpuDisplayName')}  dc={mach.get('dataCenterId')}  "
            f"vol={p.get('networkVolumeId')}  ${p.get('costPerHr')}/hr{ssh_str}"
        )
    return 0


def ensure_network_volume(name: str, size_gb: int, datacenter: str) -> dict[str, Any]:
    vols = run_graphql("{ myself { networkVolumes { id name size dataCenterId } } }")["myself"][
        "networkVolumes"
    ]
    for v in vols or []:
        if v["name"] == name:
            print(
                f"Reusing existing network volume {v['id']} (name={v['name']}, {v['size']}GB, dc={v['dataCenterId']})"
            )
            return v

    mutation = """
    mutation ($input: CreateNetworkVolumeInput!) {
      createNetworkVolume(input: $input) {
        id
        name
        size
        dataCenterId
      }
    }
    """
    vol = run_graphql(
        mutation,
        {"input": {"name": name, "size": int(size_gb), "dataCenterId": datacenter}},
    )["createNetworkVolume"]
    print(
        f"Created network volume {vol['id']} (name={vol['name']}, {vol['size']}GB, dc={vol['dataCenterId']})"
    )
    return vol


def cmd_create_volume(args: argparse.Namespace) -> int:
    ensure_network_volume(args.name, args.size, args.datacenter)
    return 0


def cmd_up(args: argparse.Namespace) -> int:
    if args.dry_run:
        vol = {"id": f"<reuse-or-create:{args.volume}>", "dataCenterId": args.datacenter}
    else:
        vol = ensure_network_volume(args.volume, args.volume_size, args.datacenter)
    dc_id = vol["dataCenterId"]

    env_list = [
        {"key": "VAM_CACHE_DIR", "value": "/workspace/.cache/video-vam"},
        {"key": "HF_HOME", "value": "/workspace/.cache/huggingface"},
        {"key": "HF_LEROBOT_HOME", "value": "/workspace/.cache/huggingface/lerobot"},
        {"key": "UV_CACHE_DIR", "value": "/workspace/.cache/uv"},
        {"key": "HF_AUTO_UPLOAD", "value": "1"},
        {"key": "HF_UPLOAD_OWNER", "value": "Orellius"},
    ]
    for secret_key in ("HF_TOKEN", "WANDB_API_KEY"):
        val = _load_env_fallback(secret_key)
        if val:
            env_list.append({"key": secret_key, "value": val})
        else:
            print(f"WARNING: {secret_key} not found locally; pod will not receive it.", file=sys.stderr)

    pubkey = _detect_ssh_pubkey(args.ssh_pubkey)
    if pubkey:
        env_list.append({"key": "PUBLIC_KEY", "value": pubkey})
    else:
        print(
            "WARNING: No SSH public key found in ~/.ssh/*.pub; SSH key auth may need RunPod account keys.",
            file=sys.stderr,
        )

    pod_input: dict[str, Any] = {
        "cloudType": "SECURE",
        "gpuCount": 1,
        "gpuTypeId": args.gpu,
        "name": args.name,
        "imageName": args.image,
        "containerDiskInGb": int(args.container_disk),
        "networkVolumeId": vol["id"],
        "volumeMountPath": "/workspace",
        "dataCenterId": dc_id,
        "ports": "22/tcp,8765/tcp,8888/http",
        "startSsh": True,
        "env": env_list,
    }

    if args.dry_run:
        redacted = {
            **pod_input,
            "env": [
                {
                    "key": e["key"],
                    "value": "<REDACTED>" if "TOKEN" in e["key"] or "KEY" in e["key"] else e["value"],
                }
                for e in env_list
            ],
        }
        print("DRY RUN podFindAndDeployOnDemand input:")
        print(json.dumps(redacted, indent=2))
        return 0

    mutation = """
    mutation ($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
        name
        desiredStatus
        costPerHr
        machine { gpuDisplayName dataCenterId }
      }
    }
    """
    pod = run_graphql(mutation, {"input": pod_input})["podFindAndDeployOnDemand"]
    pod_id = pod["id"]
    print(
        f"Deployed pod {pod_id} ({pod['name']}) in {dc_id} at ${pod.get('costPerHr')}/hr. Waiting for SSH..."
    )

    deadline = time.time() + args.wait_seconds
    ssh_ep = None
    while time.time() < deadline:
        q = """
        query ($id: String!) {
          pod(input: { podId: $id }) {
            id
            desiredStatus
            runtime {
              uptimeInSeconds
              ports { ip isIpPublic privatePort publicPort type }
            }
          }
        }
        """
        cur = run_graphql(q, {"id": pod_id})["pod"]
        ssh_ep = _extract_ssh_endpoint(cur)
        if ssh_ep:
            break
        time.sleep(4)

    if ssh_ep:
        ip, port = ssh_ep
        print("\nPod READY! Connect via:")
        print(f"  ssh root@{ip} -p {port}")
        print("\nSuggested ~/.ssh/config entry:")
        print(f"Host runpod-5090\n  HostName {ip}\n  User root\n  Port {port}\n  ForwardAgent yes")
        print("\nBootstrap command:")
        print(f"  ssh root@{ip} -p {port} 'bash -s' < scripts/video_vam/runpod/bootstrap_pod.sh")
    else:
        print(
            f"Pod {pod_id} deployed, but SSH port did not appear within {args.wait_seconds}s. Run `status` to check."
        )
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    q = "mutation ($id: String!) { podStop(input: { podId: $id }) { id desiredStatus } }"
    res = run_graphql(q, {"id": args.pod_id})["podStop"]
    print(f"Stopped pod {res['id']} (status={res['desiredStatus']}). Network volume preserved.")
    return 0


def cmd_terminate(args: argparse.Namespace) -> int:
    q = "mutation ($id: String!) { podTerminate(input: { podId: $id }) }"
    run_graphql(q, {"id": args.pod_id})
    print(f"Terminated pod {args.pod_id}. Network volume preserved.")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    q = "mutation ($id: String!) { podResume(input: { podId: $id, gpuCount: 1 }) { id desiredStatus } }"
    res = run_graphql(q, {"id": args.pod_id})["podResume"]
    print(f"Resumed pod {res['id']} (status={res['desiredStatus']}). Run `status` to get the new SSH port.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_stock = sub.add_parser("stock", help="List Secure Cloud + Network Volume availability for a GPU.")
    p_stock.add_argument("--gpu", default=DEFAULT_GPU, help=f"RunPod GPU type ID (default: {DEFAULT_GPU}).")
    p_stock.set_defaults(func=cmd_stock)

    p_status = sub.add_parser("status", help="Show RunPod balance, network volumes, and pods.")
    p_status.set_defaults(func=cmd_status)

    p_vol = sub.add_parser("create-volume", help="Create or reuse a persistent RunPod Network Volume.")
    p_vol.add_argument("--name", default=DEFAULT_VOLUME_NAME)
    p_vol.add_argument("--size", type=int, default=DEFAULT_VOLUME_SIZE_GB)
    p_vol.add_argument("--datacenter", default=DEFAULT_DATACENTER)
    p_vol.set_defaults(func=cmd_create_volume)

    p_up = sub.add_parser("up", help="Deploy an RTX 5090 pod attached to a reusable Network Volume.")
    p_up.add_argument("--name", default="lerobot-vam-5090")
    p_up.add_argument("--gpu", default=DEFAULT_GPU)
    p_up.add_argument("--image", default=DEFAULT_IMAGE)
    p_up.add_argument("--volume", default=DEFAULT_VOLUME_NAME)
    p_up.add_argument("--volume-size", type=int, default=DEFAULT_VOLUME_SIZE_GB)
    p_up.add_argument("--container-disk", type=int, default=DEFAULT_CONTAINER_DISK_GB)
    p_up.add_argument("--datacenter", default=DEFAULT_DATACENTER)
    p_up.add_argument("--ssh-pubkey", type=Path, default=None)
    p_up.add_argument("--wait-seconds", type=int, default=180)
    p_up.add_argument("--dry-run", action="store_true")
    p_up.set_defaults(func=cmd_up)

    p_stop = sub.add_parser("stop", help="Stop a running pod (preserves network volume).")
    p_stop.add_argument("--pod-id", required=True)
    p_stop.set_defaults(func=cmd_stop)

    p_term = sub.add_parser("terminate", help="Terminate a pod permanently (preserves network volume).")
    p_term.add_argument("--pod-id", required=True)
    p_term.set_defaults(func=cmd_terminate)

    p_res = sub.add_parser("resume", help="Resume a stopped pod.")
    p_res.add_argument("--pod-id", required=True)
    p_res.set_defaults(func=cmd_resume)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
