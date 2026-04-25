"""Scaling-law runner for the validated high-n LocoProp path.

Defaults use `locograd` outer AdamW plus Nesterov inner steps, matching the recent
high-n probes. Pass `--loco-opt sign --inner-beta 0.0` for main.py's older CLI path.
"""

import argparse
import json
import random
import socket
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from main import LocoShuffleMLPBlock, ResidualMLP, train


INNER_STEPS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
BATCH_SIZES = [32, 64, 128, 256, 512, 1024]
LRS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
KEY_FIELDS = (
    "inner_steps", "batch_size", "outer_lr", "inner_lr", "inner_beta", "loco_opt",
    "groups", "block", "dim", "layers", "epochs", "print_every", "seed", "device",
    "eval_samples", "eval_chunk",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inner-steps", nargs="+", type=int, default=INNER_STEPS)
    p.add_argument("--batch-sizes", nargs="+", type=int, default=BATCH_SIZES)
    p.add_argument("--lrs", nargs="+", type=float, default=LRS)
    p.add_argument("--epochs", type=int, default=4096)
    p.add_argument("--print-every", type=int, default=64)
    p.add_argument("--groups", type=int, default=16)
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--device", default="cuda")
    p.add_argument("--inner-lr", type=float, default=1.0)
    p.add_argument("--inner-beta", type=float, default=0.9)
    p.add_argument("--loco-opt", default="locograd")
    p.add_argument("--eval-samples", type=int, default=16384)
    p.add_argument("--eval-chunk", type=int, default=4096)
    p.add_argument("--output", type=Path)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def default_output_path():
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("findings") / f"locoprop_scaling_law_{ts}.jsonl"


def finite_or_none(x):
    x = float(x)
    return x if np.isfinite(x) else None


def cuda_index(device):
    idx = torch.device(device).index
    return torch.cuda.current_device() if idx is None else idx


def key_value(cfg, field):
    return cfg.get("lr") if field == "outer_lr" and "outer_lr" not in cfg else cfg.get(field)


def run_key(cfg):
    return tuple(key_value(cfg, field) for field in KEY_FIELDS)


def done_keys(path):
    if not path.exists():
        return set()
    keys = set()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind", "run") != "run":
                continue
            cfg = row.get("config", {})
            keys.add(run_key(cfg))
    return keys


def repair_jsonl_tail(path):
    if not path.exists() or path.stat().st_size == 0:
        return
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return
    last_nl = data.rfind(b"\n")
    tail = data[last_nl + 1:]
    try:
        json.loads(tail.decode())
    except (json.JSONDecodeError, UnicodeDecodeError):
        with path.open("ab") as f:
            f.truncate(0 if last_nl < 0 else last_nl + 1)
    else:
        with path.open("ab") as f:
            f.write(b"\n")


@torch.no_grad()
def eval_clean(model, teacher, dim, device, samples, chunk):
    if samples <= 0:
        return None
    model.eval()
    mse = t2 = n = 0.0
    for _ in range(samples // chunk):
        x = torch.randn(chunk, dim, device=device)
        t = teacher(x)
        y = model(x)
        mse += F.mse_loss(y, t).item() * chunk
        t2 += t.pow(2).mean().item() * chunk
        n += chunk
    model.train()
    if n == 0:
        return None
    mse /= n
    t2 /= n
    return {"samples": int(n), "mse": finite_or_none(mse), "target_m2": finite_or_none(t2),
            "nmse": finite_or_none(mse / t2)}


def build_model(args, inner_steps, dim):
    return ResidualMLP(
        LocoShuffleMLPBlock,
        args.layers,
        groups=args.groups,
        block=args.block,
        loco_steps=inner_steps,
        lr=args.inner_lr,
        wd=0.0,
        inner_beta=args.inner_beta,
        autograd_targets=True,
    ).to(args.device)


def summarize(losses, print_every):
    if len(losses) == 0:
        return {"recorded_steps": 0, "final": None, "best": None, "tail_mean": None,
                "argmin_step": None}
    tail = losses[-print_every:] if len(losses) >= print_every else losses
    finite = np.isfinite(losses)
    return {
        "recorded_steps": int(len(losses)),
        "final": finite_or_none(losses[-1]),
        "best": finite_or_none(np.min(losses[finite])) if finite.any() else None,
        "tail_mean": finite_or_none(np.mean(tail[np.isfinite(tail)])) if np.isfinite(tail).any() else None,
        "argmin_step": int(np.nanargmin(losses)) if finite.any() else None,
    }


def final_window_diverged(losses, print_every):
    n = len(losses) // print_every
    if n < 2:
        return False
    windows = losses[:n * print_every].reshape(n, print_every)
    with np.errstate(divide="ignore", invalid="ignore"):
        geomeans = np.exp(np.log(windows).mean(axis=1))
    prev = geomeans[:-1][np.isfinite(geomeans[:-1])]
    return len(prev) > 0 and np.isfinite(geomeans[-1]) and geomeans[-1] > 2 * np.min(prev)


def classify_status(losses, expected_steps, print_every):
    if len(losses) == 0:
        return "early_stop"
    if not np.isfinite(losses).all() or final_window_diverged(losses, print_every):
        return "diverged"
    return "ok" if len(losses) >= expected_steps else "early_stop"


def run_config(args, inner_steps, batch_size, lr, dim):
    return {
        "inner_steps": inner_steps,
        "batch_size": batch_size,
        "lr": lr,
        "outer_lr": lr,
        "inner_lr": args.inner_lr,
        "inner_beta": args.inner_beta,
        "loco_opt": args.loco_opt,
        "groups": args.groups,
        "block": args.block,
        "dim": dim,
        "layers": args.layers,
        "epochs": args.epochs,
        "print_every": args.print_every,
        "seed": args.seed,
        "device": args.device,
        "eval_samples": args.eval_samples,
        "eval_chunk": args.eval_chunk,
    }


def trajectory_path(traj_dir, inner_steps, batch_size, lr):
    return traj_dir / f"n{inner_steps}_B{batch_size}_lr{lr:g}.npy"


def run_one(args, inner_steps, batch_size, lr, index, total, traj_dir):
    dim = args.groups * args.block
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(cuda_index(args.device))

    start = time.perf_counter()
    model = None
    traj_file = trajectory_path(traj_dir, inner_steps, batch_size, lr)
    row = {
        "schema_version": 2,
        "kind": "run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "index": index,
        "total": total,
        "config": run_config(args, inner_steps, batch_size, lr, dim),
        "trajectory_file": str(traj_file),
    }

    try:
        model = build_model(args, inner_steps, dim)
        out = train(model, args.epochs, batch_size, dim, lr, args.device, args.print_every,
                    loco_opt=args.loco_opt, return_full_trajectory=True)
        losses = np.asarray(out["train_losses"], dtype=np.float64)
        np.save(traj_file, losses)
        row.update(
            status=classify_status(losses, args.epochs + 1, args.print_every),
            elapsed_sec=finite_or_none(time.perf_counter() - start),
            summary=summarize(losses, args.print_every),
        )
        try:
            row["eval"] = eval_clean(model, out["teacher"], dim, args.device, args.eval_samples, args.eval_chunk)
            row["eval_model_timing"] = "after_final_optimizer_step"
        except Exception as e:
            row["eval"] = None
            row["eval_error"] = {"message": str(e), "traceback": traceback.format_exc()}
    except Exception as e:
        row.update(
            status=f"error:{type(e).__name__}",
            elapsed_sec=finite_or_none(time.perf_counter() - start),
            error={"message": str(e), "traceback": traceback.format_exc()},
            summary=summarize(np.asarray([], dtype=np.float64), args.print_every),
            eval=None,
        )
    finally:
        row["cuda"] = {
            "device_name": torch.cuda.get_device_name(cuda_index(args.device)),
            "max_memory_allocated": int(torch.cuda.max_memory_allocated(cuda_index(args.device))),
        }
        if model is not None:
            del model
        torch.cuda.empty_cache()
    return row


def metadata(args, total):
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "metadata",
        "host": socket.gethostname(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "total_runs": total,
        "target_grid": {
            "inner_steps": args.inner_steps,
            "batch_sizes": args.batch_sizes,
            "lrs": args.lrs,
            "outer_lrs": args.lrs,
        },
        "defaults": {
            "epochs": args.epochs,
            "print_every": args.print_every,
            "inner_lr": args.inner_lr,
            "inner_beta": args.inner_beta,
            "loco_opt": args.loco_opt,
            "return_full_trajectory": True,
        },
        "resume_key_fields": KEY_FIELDS,
    }


def main():
    args = parse_args()
    if args.print_every < 1:
        raise SystemExit("--print-every must be positive")
    if torch.device(args.device).type != "cuda":
        raise SystemExit("main.train() is CUDA-only in this repo; pass --device cuda or cuda:<index>")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if args.eval_samples < 0 or args.eval_chunk <= 0:
        raise SystemExit("--eval-samples must be non-negative and --eval-chunk must be positive")
    if args.eval_samples % args.eval_chunk != 0:
        raise SystemExit("--eval-samples must be divisible by --eval-chunk")

    output = args.output or default_output_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    traj_dir = output.parent / "trajectories" / output.stem
    traj_dir.mkdir(parents=True, exist_ok=True)

    grid = sorted(
        [(n, b, lr) for n in args.inner_steps for b in args.batch_sizes for lr in args.lrs],
        key=lambda t: (t[1], t[0], t[2]),
    )
    if args.shuffle:
        random.Random(args.seed).shuffle(grid)
    if args.limit is not None:
        grid = grid[:args.limit]

    print(f"writing {len(grid)} runs to {output}", flush=True)
    print(f"inner_steps={args.inner_steps}", flush=True)
    print(f"batch_sizes={args.batch_sizes}", flush=True)
    print(f"lrs={args.lrs}", flush=True)
    if args.dry_run:
        return

    total = len(grid)
    repair_jsonl_tail(output)
    completed = done_keys(output)
    with output.open("a") as f:
        if output.stat().st_size == 0:
            f.write(json.dumps(metadata(args, total), allow_nan=False) + "\n")
            f.flush()
        for i, (inner_steps, batch_size, lr) in enumerate(grid, 1):
            key = run_key(run_config(args, inner_steps, batch_size, lr, args.groups * args.block))
            if key in completed:
                print(f"[{i}/{total}] skip n={inner_steps} B={batch_size} lr={lr:g}", flush=True)
                continue
            print(f"[{i}/{total}] n={inner_steps} B={batch_size} lr={lr:g}", flush=True)
            row = run_one(args, inner_steps, batch_size, lr, i, total, traj_dir)
            f.write(json.dumps(row, allow_nan=False) + "\n")
            f.flush()
            completed.add(key)
            s = row["summary"]
            print(f"    {row['status']} tail={s['tail_mean']} best={s['best']} "
                  f"steps={s['recorded_steps']} time={row['elapsed_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
