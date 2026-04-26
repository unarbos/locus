"""Fan out locoprop_scaling_law.py workers across GPUs sharing one --output."""
import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default="0", help="comma-separated GPU indices")
    p.add_argument("--workers-per-gpu", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--script", default="locoprop_scaling_law.py")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--logs", type=Path, default=None)
    p.add_argument("--stagger", type=float, default=2.0)
    p.add_argument("rest", nargs=argparse.REMAINDER, help="-- followed by sweep-script args")
    return p.parse_args()


def main():
    args = parse_args()
    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    if not gpus:
        raise SystemExit("--gpus must list at least one index")
    if args.workers_per_gpu < 1:
        raise SystemExit("--workers-per-gpu must be >= 1")
    forwarded = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest

    args.output.mkdir(parents=True, exist_ok=True)
    logs_dir = args.logs or (args.output / "logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    procs = []
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_cmd = [args.python, "-u", args.script,
                "--output", str(args.output), *forwarded]
    print(f"launcher base cmd: {' '.join(shlex.quote(c) for c in base_cmd)}", flush=True)

    try:
        for g in gpus:
            for w in range(args.workers_per_gpu):
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(g)
                env.pop("CUDA_DEVICE_ORDER", None)
                tag = f"gpu{g}_w{w}_{ts}"
                log = logs_dir / f"{tag}.log"
                f = open(log, "wb", buffering=0)
                p = subprocess.Popen(base_cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                     start_new_session=True)
                procs.append((tag, p, f))
                print(f"  spawned {tag} pid={p.pid} log={log}", flush=True)
                time.sleep(args.stagger)
    except BaseException:
        for tag, p, f in procs:
            p.send_signal(signal.SIGTERM)
        raise

    failed = 0
    try:
        while procs:
            for entry in list(procs):
                tag, p, f = entry
                rc = p.poll()
                if rc is None:
                    continue
                f.close()
                print(f"[{tag}] exited {'ok' if rc == 0 else f'FAIL rc={rc}'}", flush=True)
                failed += rc != 0
                procs.remove(entry)
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("launcher interrupted; sending SIGTERM to workers", flush=True)
        for tag, p, f in procs:
            p.send_signal(signal.SIGTERM)
        for tag, p, f in procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
            f.close()
        raise SystemExit(130)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
