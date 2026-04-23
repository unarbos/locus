"""Ablate claims: SGD outer at proper LR? BP floor real? n scaling saturated?"""
import numpy as np
import torch
from torch.nn import functional as F
from main import LocoShuffleMLPBlock, ShuffleMLPBlock, ResidualMLP, train

DIM, GROUPS, BLOCK, LAYERS = 1024, 16, 64, 8
BATCH, PRINT_EVERY = 16384, 256
SEED = 43


@torch.no_grad()
def eval_clean(model, teacher, n=16384, chunk=4096):
    model.eval()
    mse = t2 = 0.0
    for _ in range(n // chunk):
        x = torch.randn(chunk, DIM, device='cuda')
        t = teacher(x); y = model(x)
        mse += F.mse_loss(y, t).item() * chunk
        t2 += t.pow(2).mean().item() * chunk
    model.train()
    return mse / n, t2 / n


def run(cfg, epochs):
    torch.manual_seed(SEED)
    if cfg['kind'] == 'bp_adam':
        m = ResidualMLP(ShuffleMLPBlock, LAYERS, groups=GROUPS, block=BLOCK).cuda()
        out = train(m, epochs, BATCH, DIM, cfg['outer_lr'], 'cuda', PRINT_EVERY)
    else:
        m = ResidualMLP(LocoShuffleMLPBlock, LAYERS, groups=GROUPS, block=BLOCK,
                        loco_steps=cfg['n_steps'], lr=cfg['inner_lr'], wd=0.0,
                        inner_beta=cfg.get('inner_beta', 0.0),
                        autograd_targets=True).cuda()
        out = train(m, epochs, BATCH, DIM, cfg['outer_lr'], 'cuda', PRINT_EVERY,
                    loco_opt=cfg['loco_opt'])
    L = np.asarray(out['train_losses'])
    tail = L[-PRINT_EVERY:].mean() if len(L) >= PRINT_EVERY else float('nan')
    best = L.min() if len(L) else float('nan')
    try:
        mse, t2 = eval_clean(m, out['teacher'])
    except Exception:
        mse, t2 = float('nan'), 1.0
    del m; torch.cuda.empty_cache()
    return tail, best, mse, mse / t2


# Ablation 1: SGD outer LR scan (n=64 loco) — is my "AdamW essential" claim real?
# loco_as_grad magnitude is comparable to a full parameter update, so LR ~1 is natural.
sgd_scan = [
    dict(label='n=64 sgd out=0.01 nes i=1.0   ', kind='loco', n_steps=64, inner_lr=1.0, outer_lr=0.01, inner_beta=0.9, loco_opt='loco_sgd'),
    dict(label='n=64 sgd out=0.1  nes i=1.0   ', kind='loco', n_steps=64, inner_lr=1.0, outer_lr=0.1,  inner_beta=0.9, loco_opt='loco_sgd'),
    dict(label='n=64 sgd out=0.3  nes i=1.0   ', kind='loco', n_steps=64, inner_lr=1.0, outer_lr=0.3,  inner_beta=0.9, loco_opt='loco_sgd'),
    dict(label='n=64 sgd out=1.0  nes i=1.0   ', kind='loco', n_steps=64, inner_lr=1.0, outer_lr=1.0,  inner_beta=0.9, loco_opt='loco_sgd'),
    dict(label='n=64 sgd out=3.0  nes i=1.0   ', kind='loco', n_steps=64, inner_lr=1.0, outer_lr=3.0,  inner_beta=0.9, loco_opt='loco_sgd'),
]
# And SGD-nesterov outer variant
sgd_scan += [
    dict(label='n=64 snes out=0.3  nes i=1.0  ', kind='loco', n_steps=64, inner_lr=1.0, outer_lr=0.3,  inner_beta=0.9, loco_opt='loco_nesterov'),
    dict(label='n=64 snes out=1.0  nes i=1.0  ', kind='loco', n_steps=64, inner_lr=1.0, outer_lr=1.0,  inner_beta=0.9, loco_opt='loco_nesterov'),
]

# Ablation 2: BP at higher LR / longer — is the 0.06 floor real or just stuck?
bp_scan = [
    dict(label='bp+adamw 1e-2                 ', kind='bp_adam', outer_lr=1e-2),
    dict(label='bp+adamw 3e-3 (anchor)        ', kind='bp_adam', outer_lr=3e-3),
]

# Ablation 3: n scaling — did n=256 saturate?
push_scan = [
    dict(label='n=512 adam 3e-3 nes i=1.0    ', kind='loco', n_steps=512, inner_lr=1.0, outer_lr=3e-3, inner_beta=0.9, loco_opt='locograd'),
]

print("=" * 70, flush=True)
print("Ablation 1: SGD/Nesterov outer LR scan (n=64, 1024 epochs)", flush=True)
print("=" * 70, flush=True)
rows = []
for cfg in sgd_scan:
    print(f"\n=== {cfg['label']} ===", flush=True)
    try:
        tail, best, mse, nmse = run(cfg, 1024)
    except Exception as e:
        tail = best = mse = nmse = float('nan')
        print(f"  FAILED: {type(e).__name__}: {e}")
    rows.append((cfg['label'], tail, best, mse, nmse))

print("\n=" * 70, flush=True)
print("Ablation 2: BP higher LR / longer (4096 epochs)", flush=True)
print("=" * 70, flush=True)
for cfg in bp_scan:
    print(f"\n=== {cfg['label']} ===", flush=True)
    try:
        tail, best, mse, nmse = run(cfg, 4096)
    except Exception as e:
        tail = best = mse = nmse = float('nan')
        print(f"  FAILED: {type(e).__name__}: {e}")
    rows.append((cfg['label'], tail, best, mse, nmse))

print("\n=" * 70, flush=True)
print("Ablation 3: n=512 loco, 2048 epochs (saturation?)", flush=True)
print("=" * 70, flush=True)
for cfg in push_scan:
    print(f"\n=== {cfg['label']} ===", flush=True)
    try:
        tail, best, mse, nmse = run(cfg, 2048)
    except Exception as e:
        tail = best = mse = nmse = float('nan')
        print(f"  FAILED: {type(e).__name__}: {e}")
    rows.append((cfg['label'], tail, best, mse, nmse))

print(f"\n=== summary ===", flush=True)
print(f"{'config':>32s}  {'tail':>8s}  {'best':>8s}  {'eval':>8s}  {'nMSE':>7s}", flush=True)
for tag, tail, best, mse, nmse in rows:
    t_s = f"{tail:8.4f}" if np.isfinite(tail) else "     nan"
    b_s = f"{best:8.4f}" if np.isfinite(best) else "     nan"
    e_s = f"{mse:8.4f}" if np.isfinite(mse) else "     nan"
    n_s = f"{nmse:7.3f}" if np.isfinite(nmse) else "    nan"
    print(f"{tag:>32s}  {t_s}  {b_s}  {e_s}  {n_s}", flush=True)
