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
        self.c_proj = nn.Parameter(torch.zeros(HDIM, DIM))
        self._x = None
        self._g = None

    def forward(self, x):
        xn = rms_norm(x)
        h = xn @ self.c_fc.T
        y = F.relu(h).pow(2) @ self.c_proj
        if self.training and LOCO_STEPS > 0:
            self._x = xn.detach()
            self._g = None
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
        target = F.relu(x @ W1.T).pow(2) @ W2 - go
        W1k = W1.clone()
        W2k = W2.clone()
        acc1 = torch.zeros_like(W1)
        acc2 = torch.zeros_like(W2)
        for _ in range(LOCO_STEPS):
            pre = x @ W1k.T
            relu = pre.relu()
            post = relu.pow(2)
            dout = (post @ W2k - target) / n
            dpre = (dout @ W2k.T) * 2 * relu
            u1 = LOCO_LR * (dpre.T @ x)
            u2 = LOCO_LR * (post.T @ dout)
            W1k -= u1
            W2k -= u2
            acc1 += u1
            acc2 += u2
        W1.loco_grad = acc1
        W2.loco_grad = acc2
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
    def __init__(self, params, lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, sign=True):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.sign = sign

    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            lr, (b1, b2), eps, wd = grp['lr'], grp['betas'], grp['eps'], grp['weight_decay']
            for p in grp['params']:
                if p.grad is None and getattr(p, 'loco_grad', None) is None:
                    continue
                g = p.grad.float() if p.grad is not None else torch.zeros_like(p, dtype=torch.float32)
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
                if self.sign:
                    mag = m.abs().div_(denom).mul_(lr / bc1)
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    lg = getattr(p, 'loco_grad', None)
                    direction = lg.sign().neg_() if lg is not None else m.sign().neg_()
                    p.addcmul_(direction.to(p.dtype), mag.to(p.dtype))
                    if lg is not None:
                        p.loco_grad = None
                else:
                    lg = getattr(p, 'loco_grad', None)
                    src = lg.float() if lg is not None else m
                    update = src.div(denom).mul(lr / bc1)
                    if wd != 0:
                        update.add_(p, alpha=lr * wd)
                    p.sub_(update.to(p.dtype))
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
    opt = AdamWLoco(model.parameters(), lr=LR, weight_decay=WD,
                    sign=int(os.environ.get('SIGN', '1')))

    data = Path(os.environ.get('DATA_PATH', 'data/fineweb10B'))
    train_shards = sorted(data.glob('fineweb_train_*.bin'))
    val_shards = sorted(data.glob('fineweb_val_*.bin'))
    assert train_shards and val_shards, f"no shards at {data}"
    train_it = loader(train_shards, BATCH_SEQ, SEQ_LEN)
    val_it = loader(val_shards, BATCH_SEQ, SEQ_LEN)

    if master:
        print(f"loco_steps={LOCO_STEPS} loco_lr={LOCO_LR} lr={LR} batch_seq={BATCH_SEQ} seq_len={SEQ_LEN}", flush=True)

    staged = int(os.environ.get('STAGED', '0'))
    block_params = []
    for i in range(N_LAYERS):
        bp = set(model.attns[i].parameters()) | set(model.mlps[i].parameters())
        block_params.append(bp)
    shared_params = set(model.parameters())
    for bp in block_params:
        shared_params -= bp

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
        if LOCO_STEPS > 0 and staged:
            saved = {p: (p.grad.detach().clone() if p.grad is not None else None)
                     for p in model.parameters()}
            for i in range(N_LAYERS):
                for p in model.parameters():
                    p.grad = (saved[p].clone() if (p in block_params[i] and saved[p] is not None) else None)
                model.mlps[i].loco_step()
                reduce_grads()
                opt.step()
                opt.zero_grad(set_to_none=True)
            for p in model.parameters():
                p.grad = (saved[p].clone() if (p in shared_params and saved[p] is not None) else None)
            reduce_grads()
            opt.step()
            opt.zero_grad(set_to_none=True)
        else:
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
