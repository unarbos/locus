"""Build, compute, and plot the LocoProp scaling ladder.

Reads sweep jsonl(s), extracts (n, B, lr) -> loss, and produces:
  1. ladder envelope: per-n best loss across (B, lr), as a function of n and as a function of compute
  2. Pareto: loss vs compute (bwd-passes) colored by n
  3. heatmap: loss(B, lr) per n
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_runs(paths):
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("kind") != "run" or str(d.get("status", "")).startswith("error"):
                continue
            c = d["config"]; s = d.get("summary") or {}
            ev = d.get("eval") or {}
            best = s.get("best"); tail = s.get("tail_mean")
            if best is None or tail is None or not np.isfinite(best) or not np.isfinite(tail):
                continue
            rows.append(dict(
                n=c["inner_steps"], B=c["batch_size"], lr=c["outer_lr"],
                epochs=c["epochs"], best=best, tail=tail,
                eval=ev.get("nmse"),
                bwd=c["epochs"] * (c["inner_steps"] + 1) * c["batch_size"],
                samples=c["epochs"] * c["batch_size"],
                status=d.get("status"),
            ))
    return rows


def best_by(rows, group, key):
    out = {}
    for r in rows:
        k = tuple(r[g] for g in group) if isinstance(group, (list, tuple)) else r[group]
        cur = out.get(k)
        if cur is None or r[key] < cur[key]:
            out[k] = r
    return out


def pareto_lower(xs, ys):
    order = np.argsort(xs)
    xs, ys = np.asarray(xs)[order], np.asarray(ys)[order]
    keep, best = [], np.inf
    for i, y in enumerate(ys):
        if y < best:
            keep.append(i); best = y
    return xs[keep], ys[keep]


def plot_ladder(rows, out_path, key="best"):
    rows = [r for r in rows if r[key] is not None and np.isfinite(r[key])]
    ns = sorted({r["n"] for r in rows})
    Bs = sorted({r["B"] for r in rows})
    lrs = sorted({r["lr"] for r in rows})
    cmap_n = plt.get_cmap("viridis")
    cmap_b = plt.get_cmap("plasma")
    color_n = {n: cmap_n(i / max(1, len(ns) - 1)) for i, n in enumerate(ns)}
    color_b = {b: cmap_b(i / max(1, len(Bs) - 1)) for i, b in enumerate(Bs)}

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # --- (0,0) n-ladder: loss vs n, one line per B (lr-optimal), envelope ---
    ax = axes[0, 0]
    nb = best_by(rows, ("n", "B"), key)
    for B in Bs:
        xy = sorted([(n, r[key]) for (n, b), r in nb.items() if b == B])
        if xy:
            xs, ys = zip(*xy)
            ax.loglog(xs, ys, "o-", color=color_b[B], ms=4, lw=1, alpha=0.7, label=f"B={B}")
    env_n = best_by(rows, "n", key)
    xs = sorted(env_n)
    ys = [env_n[n][key] for n in xs]
    ax.loglog(xs, ys, "k-", lw=2.2, ms=8, marker="o", label="envelope")
    ax.set_xlabel("inner steps n"); ax.set_ylabel(f"{key} loss")
    ax.set_title("n-ladder (lr-optimal at each (n, B))")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8, ncol=2)

    # --- (0,1) B-ladder: loss vs B, one line per n (lr-optimal), envelope ---
    ax = axes[0, 1]
    for n in ns:
        xy = sorted([(B, r[key]) for (nn, B), r in nb.items() if nn == n])
        if xy:
            xs, ys = zip(*xy)
            ax.loglog(xs, ys, "o-", color=color_n[n], ms=4, lw=1, alpha=0.7, label=f"n={n}")
    env_b = best_by(rows, "B", key)
    xs = sorted(env_b)
    ys = [env_b[B][key] for B in xs]
    ax.loglog(xs, ys, "k-", lw=2.2, ms=8, marker="o", label="envelope")
    ax.set_xlabel("batch B"); ax.set_ylabel(f"{key} loss")
    ax.set_title("B-ladder (lr-optimal at each (n, B))")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8, ncol=2)

    # --- (1,0) interaction heatmap: n × B, lr-optimal best loss ---
    ax = axes[1, 0]
    grid = np.full((len(ns), len(Bs)), np.nan)
    lr_grid = np.full((len(ns), len(Bs)), np.nan)
    for (n, B), r in nb.items():
        grid[ns.index(n), Bs.index(B)] = r[key]
        lr_grid[ns.index(n), Bs.index(B)] = r["lr"]
    im = ax.imshow(np.log10(grid), aspect="auto", origin="lower", cmap="viridis_r")
    ax.set_xticks(range(len(Bs))); ax.set_xticklabels(Bs)
    ax.set_yticks(range(len(ns))); ax.set_yticklabels(ns)
    ax.set_xlabel("batch B"); ax.set_ylabel("inner steps n")
    ax.set_title(f"interaction: log10({key}) over (n, B)  [lr-optimal per cell, lr printed]")
    med = np.nanmedian(np.log10(grid))
    for i in range(len(ns)):
        for j in range(len(Bs)):
            v = grid[i, j]
            if np.isfinite(v):
                tc = "white" if np.log10(v) > med else "black"
                ax.text(j, i, f"{v:.3f}\n{lr_grid[i,j]:g}", ha="center", va="center",
                        fontsize=6.5, color=tc)
    plt.colorbar(im, ax=ax, label=f"log10({key})")

    # --- (1,1) Pareto: loss vs compute, scatter colored by n ---
    ax = axes[1, 1]
    for n in ns:
        sub = [r for r in rows if r["n"] == n]
        ax.scatter([r["bwd"] for r in sub], [r[key] for r in sub],
                   color=color_n[n], s=18, alpha=0.55, label=f"n={n}")
    px, py = pareto_lower([r["bwd"] for r in rows], [r[key] for r in rows])
    ax.loglog(px, py, "k-", lw=2, label="Pareto")
    ax.set_xlabel("compute = epochs * (n+1) * B  (bwd-pass-samples)")
    ax.set_ylabel(f"{key} loss")
    ax.set_title("Pareto: loss vs compute")
    ax.grid(True, which="both", alpha=0.3); ax.legend(ncol=2, fontsize=7, loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


def print_table(rows, key="best"):
    print(f"\n=== n-ladder ({len(rows)} configs) ===")
    env_n = best_by(rows, "n", key)
    print(f"{'n':>4s}  {'B':>5s}  {'lr':>8s}  {'best':>8s}  {'tail':>8s}  {'eval':>8s}  {'bwd (M)':>10s}")
    for n in sorted(env_n):
        r = env_n[n]
        ev = f"{r['eval']:.4f}" if r['eval'] is not None and np.isfinite(r['eval']) else "    -   "
        print(f"{n:>4d}  {r['B']:>5d}  {r['lr']:>8.0e}  {r['best']:>8.4f}  {r['tail']:>8.4f}  {ev}  {r['bwd']/1e6:>10.1f}")

    print(f"\n=== B-ladder ===")
    env_b = best_by(rows, "B", key)
    print(f"{'B':>5s}  {'n':>4s}  {'lr':>8s}  {'best':>8s}  {'tail':>8s}  {'eval':>8s}  {'bwd (M)':>10s}")
    for B in sorted(env_b):
        r = env_b[B]
        ev = f"{r['eval']:.4f}" if r['eval'] is not None and np.isfinite(r['eval']) else "    -   "
        print(f"{B:>5d}  {r['n']:>4d}  {r['lr']:>8.0e}  {r['best']:>8.4f}  {r['tail']:>8.4f}  {ev}  {r['bwd']/1e6:>10.1f}")

    print(f"\n=== interaction (best {key} per (n, B); '-' = unsampled) ===")
    nb = best_by(rows, ("n", "B"), key)
    Bs = sorted({r["B"] for r in rows})
    ns = sorted({r["n"] for r in rows})
    print("    n \\ B  " + "  ".join(f"{B:>7d}" for B in Bs))
    for n in ns:
        cells = []
        for B in Bs:
            r = nb.get((n, B))
            cells.append(f"{r[key]:>7.4f}" if r else "      -")
        print(f"    {n:>5d}  " + "  ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=["findings/locoprop_scaling_full.jsonl"])
    ap.add_argument("--key", default="best", choices=["best", "tail", "eval"])
    ap.add_argument("--out", default="findings/scaling_ladder.png")
    args = ap.parse_args()

    rows = load_runs(args.paths)
    if not rows:
        raise SystemExit("no usable runs")
    print_table(rows, args.key)
    plot_ladder(rows, args.out, args.key)


if __name__ == "__main__":
    main()
