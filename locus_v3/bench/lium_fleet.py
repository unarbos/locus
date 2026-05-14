"""Session-safe Lium fleet helper for Locus v3.

No credentials are embedded. Set LIUM_API_KEY via Doppler or the environment.
Only pods recorded in SESSION_FILE may be terminated by this script.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

SESSION_FILE = Path(os.environ.get("LOCUS_V3_LIUM_SESSION", "/tmp/locus_v3_lium_session.json"))


def client():
    if not os.environ.get("LIUM_API_KEY"):
        raise SystemExit("LIUM_API_KEY is required; load it from Doppler/env")
    import lium
    return lium.Lium()


def load_session() -> dict:
    if SESSION_FILE.exists():
        return json.loads(SESSION_FILE.read_text())
    return {"rented": []}


def save_session(session: dict) -> None:
    SESSION_FILE.write_text(json.dumps(session, indent=2, sort_keys=True))


def cmd_available(args) -> int:
    c = client()
    executors = c.ls(gpu_type=args.gpu)
    executors.sort(key=lambda e: e.price_per_gpu)
    for e in executors[: args.top]:
        net = (e.specs or {}).get("network", {}) or {}
        upload = net.get("ema_verifyx_upload_speed") or net.get("ema_upload_speed", 0)
        print(f"{e.huid} {e.gpu_count}x {e.gpu_type} ${e.price_per_hour:.2f}/hr upload={upload:.0f}Mbps id={e.id}")
    return 0


def cmd_rent(args) -> int:
    c = client()
    session = load_session()
    candidates = []
    for e in c.ls(gpu_type=args.gpu):
        net = (e.specs or {}).get("network", {}) or {}
        upload = net.get("ema_verifyx_upload_speed") or net.get("ema_upload_speed", 0) or 0
        cuda = (e.specs or {}).get("gpu", {}).get("cuda_driver", 0) or 0
        if upload >= args.min_upload_mbps and cuda >= args.min_cuda:
            candidates.append(e)
    candidates.sort(key=lambda e: e.price_per_gpu)
    for e in candidates[: args.n]:
        res = c.up(executor_id=e.id, name=f"locus-v3-{e.huid[:12]}")
        pod_id = res.get("id") or (res.get("pod") or {}).get("id")
        session.setdefault("rented", []).append({
            "pod_id": pod_id,
            "executor_id": e.id,
            "huid": e.huid,
            "gpu_type": e.gpu_type,
            "gpu_count": e.gpu_count,
            "price_per_hour": float(e.price_per_hour or 0),
            "rented_at": time.time(),
        })
        save_session(session)
        print(f"rented {e.huid} pod_id={pod_id}")
    return 0


def cmd_wait_ready(args) -> int:
    c = client()
    wanted = {r["pod_id"] for r in load_session().get("rented", []) if r.get("pod_id")}
    for pod in c.ps():
        if pod.id in wanted:
            c.wait_ready(pod, timeout=args.timeout, poll_interval=10)
            print(f"ready {pod.huid} {pod.ssh_cmd}")
    return 0


def cmd_write_hosts(args) -> int:
    c = client()
    wanted = {r["pod_id"] for r in load_session().get("rented", []) if r.get("pod_id")}
    pods = [p for p in c.ps() if p.id in wanted]
    lines = ["# Locus v3 session-only Lium hosts", "# tag user host port n_workers gpu_class price_per_hour"]
    for i, pod in enumerate(sorted(pods, key=lambda p: p.huid)):
        parts = (pod.ssh_cmd or "").split()
        host = parts[1].split("@")[-1] if len(parts) > 1 else "?"
        port = parts[3] if len(parts) > 3 else "22"
        tag = chr(ord("A") + i)
        e = pod.executor
        lines.append(f"{tag} root {host} {port} {e.gpu_count} {e.gpu_type} {float(e.price_per_hour or 0):.2f}")
    Path(args.output).write_text("\n".join(lines) + "\n")
    print(f"wrote {len(pods)} hosts to {args.output}")
    return 0


def cmd_terminate_mine(args) -> int:
    c = client()
    session = load_session()
    wanted = [r["pod_id"] for r in session.get("rented", []) if r.get("pod_id")]
    pods = {p.id: p for p in c.ps()}
    remaining = []
    for record in session.get("rented", []):
        pod_id = record.get("pod_id")
        if not pod_id:
            continue
        pod = pods.get(pod_id)
        if pod is None:
            continue
        if pod_id not in wanted:
            remaining.append(record)
            continue
        c.down(pod)
        print(f"terminated {pod.huid} ({pod_id})")
    session["rented"] = remaining
    save_session(session)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    av = sub.add_parser("available")
    av.add_argument("--gpu", required=True)
    av.add_argument("--top", type=int, default=10)
    av.set_defaults(fn=cmd_available)
    rent = sub.add_parser("rent")
    rent.add_argument("--gpu", required=True)
    rent.add_argument("--n", type=int, required=True)
    rent.add_argument("--min-upload-mbps", type=float, default=150.0)
    rent.add_argument("--min-cuda", type=int, default=12000)
    rent.set_defaults(fn=cmd_rent)
    wait = sub.add_parser("wait-ready")
    wait.add_argument("--timeout", type=int, default=900)
    wait.set_defaults(fn=cmd_wait_ready)
    hosts = sub.add_parser("write-hosts")
    hosts.add_argument("--output", default="bench/hosts.env")
    hosts.set_defaults(fn=cmd_write_hosts)
    sub.add_parser("terminate-mine").set_defaults(fn=cmd_terminate_mine)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
