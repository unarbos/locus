"""Control: is LocoProp's 3x loss gap from GN preconditioning, or just 'more compute per batch'?

BP with K Adam steps per sampled batch, K in {1, 4, 16, 64}. K=64 matches total Adam-step
count of n=64 LocoProp at 2048 outer batches. If BP K=64 closes the gap to LocoProp,
the advantage is batch-reuse, not layer-local GN."""
import time
import numpy as np
import torch
from torch.nn import functional as F
from main import ShuffleMLPBlock, ResidualMLP, Teacher, data

DIM, GROUPS, BLOCK, LAYERS = 1024, 16, 64, 8
BATCH, EPOCHS, PRINT_EVERY = 16384, 2048, 256
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


def train_bp_k(k, lr, epochs=EPOCHS, batch=BATCH, dim=DIM, print_every=PRINT_EVERY):
    m = ResidualMLP(ShuffleMLPBlock, LAYERS, groups=GROUPS, block=BLOCK).cuda()
    m = torch.compile(m, mode='max-autotune-no-cudagraphs')
    teacher = Teacher(dim, dim).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, fused=True, weight_decay=0.1)
    losses = []
    buf = torch.zeros((print_every,), device='cuda', dtype=torch.float64)
    best = float('inf')
    t0 = time.perf_counter()
    for step in range(1, epochs + 1):
        with torch.no_grad():
            src, tgt = data(teacher, batch, dim)
        for _ in range(k):
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(m(src), tgt)
            loss.backward()
            opt.step()
        buf[(step - 1) % print_every] = loss.detach()
        if step % print_every == 0:
            avg = buf.cpu().log().mean().exp().item()
            print(f"  {step:5d} | train {avg:.6f}", flush=True)
            losses.extend(buf.cpu().tolist())
            if not np.isfinite(avg) or avg > 2 * best:
                break
            best = min(best, avg)
    torch.cuda.synchronize()
    return m, teacher, np.asarray(losses), time.perf_counter() - t0


def run(k, lr):
    torch.manual_seed(SEED)
    m, teacher, L, dt = train_bp_k(k, lr)
    tail = L[-PRINT_EVERY:].mean() if len(L) >= PRINT_EVERY else float('nan')
    best = L.min() if len(L) else float('nan')
    try:
        mse, t2 = eval_clean(m, teacher)
    except Exception:
        mse, t2 = float('nan'), 1.0
    del m; torch.cuda.empty_cache()
    return tail, best, mse, mse / t2, dt


configs = [
    dict(label='BP K=1  lr=3e-3 ', k=1,  lr=3e-3),
    dict(label='BP K=4  lr=3e-3 ', k=4,  lr=3e-3),
    dict(label='BP K=16 lr=3e-3 ', k=16, lr=3e-3),
    dict(label='BP K=64 lr=3e-3 ', k=64, lr=3e-3),
    dict(label='BP K=4  lr=1e-3 ', k=4,  lr=1e-3),
    dict(label='BP K=16 lr=1e-3 ', k=16, lr=1e-3),
    dict(label='BP K=64 lr=1e-3 ', k=64, lr=1e-3),
]

rows = []
for cfg in configs:
    print(f"\n=== {cfg['label']} ===", flush=True)
    try:
        tail, best, mse, nmse, dt = run(cfg['k'], cfg['lr'])
    except Exception as e:
        tail = best = mse = nmse = float('nan'); dt = 0.0
        print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
    rows.append((cfg['label'], tail, best, mse, nmse, dt))

print(f"\n=== summary (ep={EPOCHS}, batch={BATCH}) ===", flush=True)
print(f"{'config':>20s}  {'tail':>8s}  {'best':>8s}  {'eval':>8s}  {'nMSE':>7s}  {'sec':>7s}", flush=True)
for tag, tail, best, mse, nmse, dt in rows:
    t_s = f"{tail:8.4f}" if np.isfinite(tail) else "     nan"
    b_s = f"{best:8.4f}" if np.isfinite(best) else "     nan"
    e_s = f"{mse:8.4f}" if np.isfinite(mse) else "     nan"
    n_s = f"{nmse:7.3f}" if np.isfinite(nmse) else "    nan"
    print(f"{tag:>20s}  {t_s}  {b_s}  {e_s}  {n_s}  {dt:7.1f}", flush=True)
