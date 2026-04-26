"""One run, orthogonal-W2 init (reverted). Verify baseline LocoProp converges (nMSE < 1)."""
import numpy as np
import torch
from torch.nn import functional as F
from main import LocoShuffleMLPBlock, ResidualMLP, train

GROUPS, BLOCK, LAYERS, LOCO_STEPS = 16, 64, 8, 16
DIM = GROUPS * BLOCK
BATCH, EPOCHS, PRINT_EVERY = 512, 256, 32


@torch.no_grad()
def eval_clean(model, teacher, n=16384, chunk=4096):
    s = dict(mse=0.0, t2=0.0, n=0)
    for _ in range(n // chunk):
        x = torch.randn(chunk, DIM, device='cuda')
        t = teacher(x)
        y = model(x)
        s['mse'] += F.mse_loss(y, t).item() * chunk
        s['t2'] += t.pow(2).mean().item() * chunk
        s['n'] += chunk
    return s['mse'] / s['n'], s['t2'] / s['n']


for lr in [0.01, 0.1, 0.3, 1.0]:
    torch.manual_seed(42)
    model = ResidualMLP(LocoShuffleMLPBlock, LAYERS, groups=GROUPS, block=BLOCK,
                        loco_steps=LOCO_STEPS, lr=lr, wd=0.0, autograd_targets=True).cuda()
    out = train(model, EPOCHS, BATCH, DIM, lr, 'cuda', PRINT_EVERY)
    losses = np.asarray(out['train_losses'], dtype=np.float64)
    mse, t2 = eval_clean(model, out['teacher'])
    tail = losses[-PRINT_EVERY:].mean() if len(losses) >= PRINT_EVERY else float('nan')
    best = losses.min() if len(losses) else float('nan')
    print(f"LR={lr:<5}  tail={tail:.4f}  best={best:.4f}  eval_mse={mse:.4f}  |t|^2={t2:.4f}  nMSE={mse/t2:.3f}")
    del model
    torch.cuda.empty_cache()
