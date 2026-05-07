import os, sys, time, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import heavyball
heavyball.utils.set_torch()

rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])
local_rank = int(os.environ['LOCAL_RANK'])
device = torch.device('cuda', local_rank)
torch.cuda.set_device(device)
dist.init_process_group(backend='nccl', device_id=device)
torch.manual_seed(int(os.environ.get('SEED', '42')))
master = (rank == 0)

DIM = 768
HDIM = 3072
HEAD_DIM = 128
N_HEADS = 6
N_LAYERS = 11
VOCAB = 50304
SEQ_LEN = int(os.environ.get('SEQ_LEN', '1024'))
BATCH_SEQ = int(os.environ.get('BATCH_SEQ', '8'))
LR = float(os.environ.get('LR', '0.001'))
WD = float(os.environ.get('WD', '0.01'))
N_STEPS = int(os.environ.get('N_STEPS', '20'))
LOCO_STEPS = int(os.environ.get('LOCO_STEPS', '0'))
LOCO_LR = float(os.environ.get('LOCO_LR', '0.0001'))


def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))


INNER_CHUNK = 64


@torch.compile(mode='max-autotune-no-cudagraphs', dynamic=False)
def _inner_sgd_first(x, go, W1, W2, lr, n, K):
    target = F.relu(x @ W1.T).pow(2) @ W2 - go
    W1k = W1.clone()
    W2k = W2.clone()
    for _ in range(K):
        pre = x @ W1k.T
        relu = pre.relu()
        post = relu.pow(2)
        dout = (post @ W2k - target) / n
        dpre = (dout @ W2k.T) * 2 * relu
        W1k = W1k - lr * (dpre.T @ x)
        W2k = W2k - lr * (post.T @ dout)
    return W1k, W2k, target


@torch.compile(mode='max-autotune-no-cudagraphs', dynamic=False)
def _inner_sgd_continue(x, target, W1k, W2k, lr, n, K):
    for _ in range(K):
        pre = x @ W1k.T
        relu = pre.relu()
        post = relu.pow(2)
        dout = (post @ W2k - target) / n
        dpre = (dout @ W2k.T) * 2 * relu
        W1k = W1k - lr * (dpre.T @ x)
        W2k = W2k - lr * (post.T @ dout)
    return W1k, W2k


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        b = 0.5 * DIM ** -0.5 * 3 ** 0.5
        self.qkv = nn.Parameter(torch.empty(3 * N_HEADS * HEAD_DIM, DIM).uniform_(-b, b))
        self.proj = nn.Parameter(torch.zeros(DIM, N_HEADS * HEAD_DIM))

    def forward(self, x):
        B, T, _ = x.shape
        qkv = rms_norm(x) @ self.qkv.T
        q, k, v = qkv.view(B, T, 3, N_HEADS, HEAD_DIM).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return o.transpose(1, 2).reshape(B, T, -1) @ self.proj.T


class LocoMLP(nn.Module):
    def __init__(self):
        super().__init__()
        b = 0.5 * DIM ** -0.5 * 3 ** 0.5
        self.c_fc = nn.Parameter(torch.empty(HDIM, DIM).uniform_(-b, b))
        self.c_proj = nn.Parameter(torch.empty(HDIM, DIM).normal_(std=0.02 / N_LAYERS ** 0.5))
        self._x = None
        self._g = None

    def forward(self, x):
        xn = rms_norm(x)
        h = xn @ self.c_fc.T
        y = F.relu(h).pow(2) @ self.c_proj
        self._x = None
        self._g = None
        if self.training and LOCO_STEPS > 0:
            self._x = xn.detach()
            y.register_hook(self._cap)
        return y

    def _cap(self, g):
        self._g = g.detach()

    @torch.no_grad()
    def loco_step(self):
        if self._x is None or self._g is None:
            return
        x = self._x.reshape(-1, DIM)
        go = self._g.reshape(-1, DIM)
        n = float(go.numel())
        W1, W2 = self.c_fc, self.c_proj
        K = min(LOCO_STEPS, INNER_CHUNK)
        W1k, W2k, target = _inner_sgd_first(x, go, W1, W2, LOCO_LR, n, K)
        remaining = LOCO_STEPS - K
        while remaining > 0:
            K2 = min(remaining, INNER_CHUNK)
            W1k, W2k = _inner_sgd_continue(x, target, W1k, W2k, LOCO_LR, n, K2)
            remaining -= K2
        W1.loco_grad = W1 - W1k
        W2.loco_grad = W2 - W2k
        self._x = None
        self._g = None


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
        self.attns = nn.ModuleList([Attention() for _ in range(N_LAYERS)])
        self.mlps = nn.ModuleList([LocoMLP() for _ in range(N_LAYERS)])
        self.lm_head = nn.Parameter(torch.empty(VOCAB, DIM))
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.lm_head, std=0.02)

    def forward(self, tokens, targets):
        x = self.embed(tokens)
        for attn, mlp in zip(self.attns, self.mlps):
            x = x + attn(x)
            x = x + mlp(x)
        logits = rms_norm(x) @ self.lm_head.T
        return F.cross_entropy(logits.view(-1, VOCAB).float(), targets.view(-1))

    def loco_step(self):
        for m in self.mlps:
            m.loco_step()


class AdamWLoco(torch.optim.Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.95), eps=1e-6, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            lr, (b1, b2), eps, wd = grp['lr'], grp['betas'], grp['eps'], grp['weight_decay']
            for p in grp['params']:
                if p.grad is None:
                    continue
                g = p.grad.float()
                st = self.state[p]
                if not st:
                    st['step'] = 0
                    st['m'] = torch.zeros_like(p, dtype=torch.float32)
                    st['v'] = torch.zeros_like(p, dtype=torch.float32)
                st['step'] += 1
                t = st['step']
                m, v = st['m'], st['v']
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                bc1 = 1 - b1 ** t
                bc2 = math.sqrt(1 - b2 ** t)
                denom = v.sqrt().div_(bc2).add_(eps)
                mag = m.abs().div_(denom).mul_(lr / bc1)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                lg = getattr(p, 'loco_grad', None)
                direction = lg.sign().neg_() if lg is not None else m.sign().neg_()
                p.addcmul_(direction.to(p.dtype), mag.to(p.dtype))
                if lg is not None:
                    p.loco_grad = None


def load_shard(path):
    header = torch.from_file(str(path), False, 256, dtype=torch.int32)
    n = int(header[2])
    with open(path, 'rb') as f:
        toks = torch.empty(n, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        f.readinto(toks.numpy())
    return toks


def loader(shards, batch_seq, seq_len):
    chunk = batch_seq * seq_len * world_size
    while True:
        for shard in shards:
            toks = load_shard(shard)
            n = len(toks) // chunk
            for s in range(n):
                start = s * chunk + rank * batch_seq * seq_len
                inp = toks[start : start + batch_seq * seq_len].to(device, dtype=torch.int64).view(batch_seq, seq_len)
                tgt = toks[start + 1 : start + 1 + batch_seq * seq_len].to(device, dtype=torch.int64).view(batch_seq, seq_len)
                yield inp, tgt


def main():
    model = GPT().to(device).float()
    for p in model.parameters():
        if world_size > 1:
            dist.broadcast(p, 0)
    opt = AdamWLoco(model.parameters(), lr=LR, weight_decay=WD)

    data = Path(os.environ.get('DATA_PATH', 'data/fineweb10B'))
    train_shards = sorted(data.glob('fineweb_train_*.bin'))
    val_shards = sorted(data.glob('fineweb_val_*.bin'))
    assert train_shards and val_shards, f"no shards at {data}"
    train_it = loader(train_shards, BATCH_SEQ, SEQ_LEN)
    val_it = loader(val_shards, BATCH_SEQ, SEQ_LEN)

    if master:
        print(f"loco_steps={LOCO_STEPS} loco_lr={LOCO_LR} lr={LR} batch_seq={BATCH_SEQ} seq_len={SEQ_LEN}", flush=True)

    # FedAvg semantics: per-rank n in inner SGD + AVG-reduce of loco_grad averages
    # local solver outputs, not data-parallel solver on the global batch. K=1 is
    # equivalent to data-parallel; K>1 diverges from a single-GPU run on the same tokens.
    def reduce_grads():
        if world_size <= 1:
            return
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            lg = getattr(p, 'loco_grad', None)
            if lg is not None:
                dist.all_reduce(lg, op=dist.ReduceOp.AVG)

    model.train()
    t0 = time.perf_counter()
    for step in range(N_STEPS):
        x, y = next(train_it)
        loss = model(x, y)
        loss.backward()
        if LOCO_STEPS > 0:
            model.loco_step()
        reduce_grads()
        opt.step()
        opt.zero_grad(set_to_none=True)
        l = loss.detach().clone()
        if world_size > 1:
            dist.all_reduce(l, op=dist.ReduceOp.AVG)
        log_every = int(os.environ.get('LOG_EVERY', '1'))
        if master and ((step + 1) % log_every == 0 or step == 0 or step == N_STEPS - 1):
            print(f"step {step+1}/{N_STEPS} loss {l.item():.4f} t {(time.perf_counter()-t0):.1f}s", flush=True)

    model.eval()
    val = torch.zeros((), device=device)
    n_val = 8
    with torch.no_grad():
        for _ in range(n_val):
            x, y = next(val_it)
            val += model(x, y)
    val /= n_val
    if world_size > 1:
        dist.all_reduce(val, op=dist.ReduceOp.AVG)
    if master:
        print(f"val_loss {val.item():.4f}", flush=True)


if __name__ == '__main__':
    main()
