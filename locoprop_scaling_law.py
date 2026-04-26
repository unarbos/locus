"""LocoProp scaling-law sweep. Per-run npz at <out>/runs/<id>.npz, O_EXCL claims at
<out>/claims/<id>.claim. Workers race a shared heavy-first grid; error rows persist."""

import argparse
import hashlib
import json
import os
import socket
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from main import DenseMLPBlock, LocoShuffleMLPBlock, ResidualMLP, ShuffleMLPBlock, train


INNER_STEPS = [4, 16, 64, 256, 1024, 4096]
BATCH_SIZES = [16, 64, 256, 1024, 4096, 16384, 65536]
LRS_LOCO = [1e-5, 4e-5, 1.6e-4, 6.4e-4, 2.5e-3, 1e-2, 4e-2, 1.6e-1]
LRS_BP = [1e-5, 2e-5, 4e-5, 8e-5, 1.6e-4, 3.2e-4, 6.4e-4, 1.28e-3,
          2.5e-3, 5e-3, 1e-2, 2e-2, 4e-2, 8e-2, 1.6e-1]
WDS = [0.0, 0.01, 0.1]

ID_FIELDS = (
    "block_kind", "inner_steps", "batch_size", "outer_lr", "weight_decay",
    "inner_lr", "inner_beta", "loco_opt",
    "groups", "block", "layers", "epochs", "print_every", "seed",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("loco", "bp", "bp_dense"), default="loco")
    p.add_argument("--inner-steps", nargs="+", type=int, default=None)
    p.add_argument("--batch-sizes", nargs="+", type=int, default=BATCH_SIZES)
    p.add_argument("--lrs", nargs="+", type=float, default=None)
    p.add_argument("--wds", nargs="+", type=float, default=WDS)
    p.add_argument("--epochs", type=int, default=65536)
    p.add_argument("--print-every", type=int, default=256)
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
    p.add_argument("--limit", type=int)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def default_output_path(mode):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("findings") / f"scaling_{mode}_{ts}"


def finite_or_none(x):
    x = float(x)
    return x if np.isfinite(x) else None


def cuda_index(device):
    idx = torch.device(device).index
    return torch.cuda.current_device() if idx is None else idx


def task_id(cfg):
    canonical = json.dumps({k: cfg[k] for k in ID_FIELDS}, sort_keys=True, allow_nan=False)
    return hashlib.sha1(canonical.encode()).hexdigest()[:16]


def try_claim(claim_path):
    try:
        fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, f"{os.getpid()} {socket.gethostname()} "
                     f"{datetime.now(timezone.utc).isoformat()}\n".encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def sweep_stale_claims(claims_dir, runs_dir, host):
    dropped = 0
    for cp in claims_dir.glob("*.claim"):
        if (runs_dir / f"{cp.stem}.npz").exists():
            cp.unlink(missing_ok=True); dropped += 1; continue
        try:
            pid, claim_host, *_ = cp.read_text().split("\n", 1)[0].split()
            pid = int(pid)
        except (OSError, ValueError):
            continue
        if claim_host != host:
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            cp.unlink(missing_ok=True); dropped += 1
        except PermissionError:
            pass
    return dropped


@torch.no_grad()
def eval_clean(model, teacher, dim, device, samples, chunk, seed):
    if samples <= 0:
        return None
    g = torch.Generator(device=device).manual_seed(seed ^ 0xE7A1)
    model.eval()
    mse = t2 = n = 0.0
    for _ in range(samples // chunk):
        x = torch.randn(chunk, dim, device=device, generator=g)
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


def build_model(args, inner_steps):
    if args.mode == "bp":
        return ResidualMLP(ShuffleMLPBlock, args.layers,
                           groups=args.groups, block=args.block).to(args.device)
    if args.mode == "bp_dense":  # hidden=block matches shuffle matmul flops (2*G*B^2)
        return ResidualMLP(DenseMLPBlock, args.layers,
                           dim=args.groups * args.block, hidden=args.block).to(args.device)
    return ResidualMLP(LocoShuffleMLPBlock, args.layers,
                       groups=args.groups, block=args.block,
                       loco_steps=inner_steps, lr=args.inner_lr, wd=0.0,
                       inner_beta=args.inner_beta, autograd_targets=True).to(args.device)


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


def run_config(args, inner_steps, batch_size, lr, wd):
    bp = args.mode in ("bp", "bp_dense")
    return {
        "block_kind": args.mode,
        "inner_steps": 1 if bp else inner_steps,
        "batch_size": batch_size,
        "outer_lr": lr,
        "weight_decay": wd,
        "inner_lr": None if bp else args.inner_lr,
        "inner_beta": None if bp else args.inner_beta,
        "loco_opt": None if bp else args.loco_opt,
        "groups": args.groups,
        "block": args.block,
        "dim": args.groups * args.block,
        "layers": args.layers,
        "epochs": args.epochs,
        "print_every": args.print_every,
        "seed": args.seed,
        "eval_samples": args.eval_samples,
        "eval_chunk": args.eval_chunk,
    }


def run_one(args, cfg):
    dim = cfg["dim"]
    torch.manual_seed(cfg["seed"])
    torch.cuda.reset_peak_memory_stats(cuda_index(args.device))
    start = time.perf_counter()
    losses = np.empty(0, dtype=np.float64)
    model = None
    meta = {
        "config": cfg,
        "task_id": task_id(cfg),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "torch_version": torch.__version__,
    }
    try:
        model = build_model(args, cfg["inner_steps"])
        torch.manual_seed(cfg["seed"])  # Teacher() is built inside train() after model; reseed for invariance
        kwargs = dict(return_full_trajectory=True, weight_decay=cfg["weight_decay"])
        if args.mode == "loco":
            kwargs["loco_opt"] = args.loco_opt
        out = train(model, cfg["epochs"], cfg["batch_size"], dim, cfg["outer_lr"],
                    args.device, cfg["print_every"], **kwargs)
        losses = np.asarray(out["train_losses"], dtype=np.float64)
        meta["status"] = classify_status(losses, cfg["epochs"], cfg["print_every"])
        meta["summary"] = summarize(losses, cfg["print_every"])
        try:
            meta["eval"] = eval_clean(model, out["teacher"], dim, args.device,
                                      cfg["eval_samples"], cfg["eval_chunk"], cfg["seed"])
        except Exception as e:
            meta["eval"] = None
            meta["eval_error"] = {"message": str(e), "traceback": traceback.format_exc()}
    except Exception as e:
        meta["status"] = f"error:{type(e).__name__}"
        meta["summary"] = summarize(losses, cfg["print_every"])
        meta["eval"] = None
        meta["error"] = {"message": str(e), "traceback": traceback.format_exc()}
    finally:
        meta["elapsed_sec"] = finite_or_none(time.perf_counter() - start)
        meta["cuda"] = {
            "device_name": torch.cuda.get_device_name(cuda_index(args.device)),
            "max_memory_allocated": int(torch.cuda.max_memory_allocated(cuda_index(args.device))),
        }
        if model is not None:
            del model
        torch.cuda.empty_cache()
    return losses, meta


def save_run(npz_path, losses, meta):
    tmp = npz_path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, losses=losses, meta=np.array(json.dumps(meta, allow_nan=False)))
    os.replace(tmp, npz_path)


def main():
    args = parse_args()
    if args.mode in ("bp", "bp_dense"):
        args.inner_steps = [1]
        if args.lrs is None:
            args.lrs = LRS_BP
    else:
        if args.inner_steps is None:
            args.inner_steps = INNER_STEPS
        if args.lrs is None:
            args.lrs = LRS_LOCO
    if args.print_every < 1:
        raise SystemExit("--print-every must be positive")
    if torch.device(args.device).type != "cuda":
        raise SystemExit("main.train() is CUDA-only; pass --device cuda or cuda:<index>")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if args.eval_samples < 0 or args.eval_chunk <= 0:
        raise SystemExit("--eval-samples must be non-negative and --eval-chunk must be positive")
    if args.eval_samples and args.eval_samples % args.eval_chunk != 0:
        raise SystemExit("--eval-samples must be divisible by --eval-chunk")

    out = args.output or default_output_path(args.mode)
    runs_dir = out / "runs"; runs_dir.mkdir(parents=True, exist_ok=True)
    claims_dir = out / "claims"; claims_dir.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname()
    dropped = sweep_stale_claims(claims_dir, runs_dir, host)
    if dropped:
        print(f"[pid {os.getpid()}] swept {dropped} stale claims on {host}", flush=True)

    grid = sorted(
        [(n, b, lr, wd) for n in args.inner_steps for b in args.batch_sizes
         for lr in args.lrs for wd in args.wds],
        key=lambda t: (-(t[0] + 1) * t[1], t[2], t[3]),
    )
    if args.limit is not None:
        grid = grid[:args.limit]

    print(f"[pid {os.getpid()}] mode={args.mode} grid={len(grid)} out={out}", flush=True)
    print(f"  inner_steps={args.inner_steps}", flush=True)
    print(f"  batch_sizes={args.batch_sizes}", flush=True)
    print(f"  lrs={args.lrs}", flush=True)
    print(f"  wds={args.wds}", flush=True)
    if args.dry_run:
        return

    completed = skipped = 0
    for i, (n, b, lr, wd) in enumerate(grid, 1):
        cfg = run_config(args, n, b, lr, wd)
        tid = task_id(cfg)
        npz_path = runs_dir / f"{tid}.npz"
        claim_path = claims_dir / f"{tid}.claim"
        if npz_path.exists():
            skipped += 1
            continue
        if not try_claim(claim_path):
            skipped += 1
            continue
        if npz_path.exists():  # raced: another worker finished and unlinked between our exists() and try_claim()
            claim_path.unlink(missing_ok=True)
            skipped += 1
            continue
        print(f"[{i}/{len(grid)}] {tid} mode={args.mode} n={cfg['inner_steps']} "
              f"B={b} lr={lr:g} wd={wd:g}", flush=True)
        try:
            losses, meta = run_one(args, cfg)
            save_run(npz_path, losses, meta)
            completed += 1
            s = meta["summary"]
            print(f"    {meta['status']} best={s['best']} tail={s['tail_mean']} "
                  f"steps={s['recorded_steps']} time={meta['elapsed_sec']:.1f}s", flush=True)
        finally:
            claim_path.unlink(missing_ok=True)

    print(f"[pid {os.getpid()}] done. completed={completed} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
