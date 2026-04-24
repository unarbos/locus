"""Fresh-data control: does BP K=1 at 16x/64x horizon match BP K=16/K=64 batch-reuse?

If yes: Adam-step count is all that matters.
If no (K>1 wins): batch reuse itself is beneficial (extract more from each batch before moving on).
"""
import time
import numpy as np
import torch
from torch.nn import functional as F
from main import ShuffleMLPBlock, ResidualMLP, Teacher, data

DIM, GROUPS, BLOCK, LAYERS = 1024, 16, 64, 8
BATCH, PRINT_EVERY = 16384, 1024
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


def train_bp_k(k, lr, epochs, print_every=PRINT_EVERY):
    m = ResidualMLP(ShuffleMLPBlock, LAYERS, groups=GROUPS, block=BLOCK).cuda()
    m = torch.compile(m, mode='max-autotune-no-cudagraphs')
    teacher = Teacher(DIM, DIM).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, fused=True, weight_decay=0.1)
    losses = []
    buf = torch.zeros((print_every,), device='cuda', dtype=torch.float64)
    best = float('inf')
    t0 = time.perf_counter()
    for step in range(1, epochs + 1):
        with torch.no_grad():
            src, tgt = data(teacher, BATCH, DIM)
        for _ in range(k):
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(m(src), tgt)
            loss.backward()
            opt.step()
        buf[(step - 1) % print_every] = loss.detach()
        if step % print_every == 0:
            avg = buf.cpu().log().mean().exp().item()
            print(f"  {step:6d} | train {avg:.6f}", flush=True)
            losses.extend(buf.cpu().tolist())
            if not np.isfinite(avg) or avg > 2 * best:
                break
            best = min(best, avg)
    torch.cuda.synchronize()
    return m, teacher, np.asarray(losses), time.perf_counter() - t0


def run(k, lr, epochs):
    torch.manual_seed(SEED)
    m, teacher, L, dt = train_bp_k(k, lr, epochs)
    tail = L[-PRINT_EVERY:].mean() if len(L) >= PRINT_EVERY else float('nan')
    best = L.min() if len(L) else float('nan')
    try:
        mse, t2 = eval_clean(m, teacher)
    except Exception:
        mse, t2 = float('nan'), 1.0
    del m; torch.cuda.empty_cache()
    return tail, best, mse, mse / t2, dt


# Fresh-data K=1 matched to total Adam steps of K=16 ep=2048 (= 32768) and K=64 ep=2048 (= 131072).
configs = [
    dict(label='BP K=1 lr=1e-3 ep=32768 ',  k=1, lr=1e-3, epochs=32768),
    dict(label='BP K=1 lr=3e-4 ep=32768 ',  k=1, lr=3e-4, epochs=32768),
    dict(label='BP K=1 lr=1e-3 ep=131072',  k=1, lr=1e-3, epochs=131072),
    dict(label='BP K=1 lr=3e-4 ep=131072',  k=1, lr=3e-4, epochs=131072),
]

rows = []
for cfg in configs:
    print(f"\n=== {cfg['label']} ===", flush=True)
    try:
        tail, best, mse, nmse, dt = run(cfg['k'], cfg['lr'], cfg['epochs'])
    except Exception as e:
        tail = best = mse = nmse = float('nan'); dt = 0.0
        print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
    rows.append((cfg['label'], tail, best, mse, nmse, dt))

print(f"\n=== summary ===", flush=True)
print(f"{'config':>28s}  {'tail':>8s}  {'best':>8s}  {'eval':>8s}  {'nMSE':>7s}  {'sec':>7s}", flush=True)
for tag, tail, best, mse, nmse, dt in rows:
    t_s = f"{tail:8.4f}" if np.isfinite(tail) else "     nan"
    b_s = f"{best:8.4f}" if np.isfinite(best) else "     nan"
    e_s = f"{mse:8.4f}" if np.isfinite(mse) else "     nan"
    n_s = f"{nmse:7.3f}" if np.isfinite(nmse) else "    nan"
    print(f"{tag:>28s}  {t_s}  {b_s}  {e_s}  {n_s}  {dt:7.1f}", flush=True)
