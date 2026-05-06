import copy
import math
import pickle
import time

import heavyball
import numpy as np
import torch
import typer
from torch import Tensor
from torch import nn
from torch.nn import functional as F

heavyball.utils.set_torch()
torch._inductor.config.fx_graph_cache = True
torch._functorch.config.enable_autograd_cache = True
torch._inductor.config.autotune_local_cache = True
app = typer.Typer(pretty_exceptions_enable=False)


def shuffle(x: Tensor, groups: int) -> Tensor:
    B, D = x.shape
    block = D // groups
    return x.reshape(B, block, groups).transpose(-1, -2).contiguous().reshape(B, D)


@torch.compile(mode='max-autotune-no-cudagraphs')
def loco_fwd(x, ln_w, ln_b, W1, b1, W2, b2, groups, block, eps=1e-5):
    x_s = shuffle(x, groups)
    x_s = F.layer_norm(x_s, (x.size(-1),), weight=ln_w, bias=ln_b, eps=eps)
    x_s = x_s.reshape(-1, groups, block)
    h = torch.einsum('bgi,gij->bgj', x_s, W1) + b1
    a = F.relu(h)
    out = torch.einsum('bgi,gij->bgj', a, W2) + b2
    return out.reshape(x_s.size(0), -1) + x

def _bwd_core(x, ln_w, ln_b, W1, b1, W2, b2, groups, grad_on_output_hidden, compute_target, eps=1e-5):
    B, D = x.shape
    x_s = shuffle(x, groups)
    mu = x_s.mean(-1, keepdim=True)
    var = x_s.var(-1, unbiased=False, keepdim=True)
    x_hat = (x_s - mu) / (var + eps).sqrt()
    x_ln = x_hat * ln_w + ln_b
    x_r = x_ln.reshape(B, groups, -1)
    h = torch.einsum('bgi,gij->bgj', x_r, W1) + b1
    mask = h > 0
    a = h * mask
    y = torch.einsum('bgi,gij->bgj', a, W2) + b2
    if compute_target:
        target = y.double() - grad_on_output_hidden.reshape(B, groups, -1).double()
    else:
        target = grad_on_output_hidden
    dout = (y.double() - target).mul(1 / y.numel()).to(y.dtype)

    db2 = dout.sum(0)
    dW2 = torch.einsum('bgi,bgj->gij', a, dout)
    da = torch.einsum('bgj,gij->bgi', dout, W2)
    dh = da * mask
    db1 = dh.sum(0)
    dW1 = torch.einsum('bgi,bgj->gij', x_r, dh)

    dx_ln = torch.einsum('bgj,gij->bgi', dh, W1).reshape(B, D)
    dln_b = dx_ln.sum(0)
    dln_w = (dx_ln * x_hat).sum(0)

    return target, dln_w, dln_b, dW1, db1, dW2, db2


INNER_CHUNK = 64


@torch.compile(mode='max-autotune-no-cudagraphs', dynamic=False)
def _inner_sgd_first(train_x, target, ln_w, ln_b, W1, b1, W2, b2, lr, groups, K):
    p = tuple(t.clone() for t in (ln_w, ln_b, W1, b1, W2, b2))
    target, *real_grads = _bwd_core(train_x, *p, groups, target, True)
    real_grads = tuple(real_grads)
    p = tuple(pj - lr * g for pj, g in zip(p, real_grads))
    for _ in range(K - 1):
        _, *grads = _bwd_core(train_x, *p, groups, target, False)
        p = tuple(pj - lr * g for pj, g in zip(p, grads))
    return p, target, real_grads


@torch.compile(mode='max-autotune-no-cudagraphs', dynamic=False)
def _inner_sgd_continue(train_x, target, ln_w, ln_b, W1, b1, W2, b2, lr, groups, K):
    p = (ln_w, ln_b, W1, b1, W2, b2)
    for _ in range(K):
        _, *grads = _bwd_core(train_x, *p, groups, target, False)
        p = tuple(pj - lr * g for pj, g in zip(p, grads))
    return p


@torch.compile(mode='max-autotune-no-cudagraphs', dynamic=False)
def _inner_nesterov_first(train_x, target, ln_w, ln_b, W1, b1, W2, b2, lr, beta, groups, K):
    p = tuple(t.clone() for t in (ln_w, ln_b, W1, b1, W2, b2))
    target, *real_grads = _bwd_core(train_x, *p, groups, target, True)
    real_grads = tuple(real_grads)
    mom = tuple(-lr * g for g in real_grads)
    p = tuple(pj + mj for pj, mj in zip(p, mom))
    for _ in range(K - 1):
        la = tuple(pj + beta * mj for pj, mj in zip(p, mom))
        _, *grads = _bwd_core(train_x, *la, groups, target, False)
        mom = tuple(beta * mj - lr * g for mj, g in zip(mom, grads))
        p = tuple(pj + mj for pj, mj in zip(p, mom))
    return p, mom, target, real_grads


@torch.compile(mode='max-autotune-no-cudagraphs', dynamic=False)
def _inner_nesterov_continue(train_x, target, ln_w, ln_b, W1, b1, W2, b2,
                              mln_w, mln_b, mW1, mb1, mW2, mb2, lr, beta, groups, K):
    p = (ln_w, ln_b, W1, b1, W2, b2)
    mom = (mln_w, mln_b, mW1, mb1, mW2, mb2)
    for _ in range(K):
        la = tuple(pj + beta * mj for pj, mj in zip(p, mom))
        _, *grads = _bwd_core(train_x, *la, groups, target, False)
        mom = tuple(beta * mj - lr * g for mj, g in zip(mom, grads))
        p = tuple(pj + mj for pj, mj in zip(p, mom))
    return p, mom


def locoprop_step(train_x, grad_on_output_hidden, ln_w, ln_b, W1, b1, W2, b2, lr, n_steps, GROUPS,
                   beta=0.0):
    """Chunked-unrolled inner SGD/Nesterov. INNER_CHUNK caps unroll size so compile time
    is bounded; large n is run as multiple chunks. n=1 ≡ BP equivalence preserved (real_grads
    comes from iteration 0 of the first chunk, bit-identical to a single bwd call)."""
    params_orig = (ln_w, ln_b, W1, b1, W2, b2)
    K = min(n_steps, INNER_CHUNK)
    if beta > 0:
        param_copy, mom, target, real_grads = _inner_nesterov_first(
            train_x, grad_on_output_hidden, *params_orig, lr, beta, GROUPS, K)
        remaining = n_steps - K
        while remaining > 0:
            K2 = min(remaining, INNER_CHUNK)
            param_copy, mom = _inner_nesterov_continue(
                train_x, target, *param_copy, *mom, lr, beta, GROUPS, K2)
            remaining -= K2
    else:
        param_copy, target, real_grads = _inner_sgd_first(
            train_x, grad_on_output_hidden, *params_orig, lr, GROUPS, K)
        remaining = n_steps - K
        while remaining > 0:
            K2 = min(remaining, INNER_CHUNK)
            param_copy = _inner_sgd_continue(
                train_x, target, *param_copy, lr, GROUPS, K2)
            remaining -= K2
    for p, c, g in zip(params_orig, param_copy, real_grads):
        p.grad = g
        p.loco_grad = p - c


@torch.compile(mode='max-autotune-no-cudagraphs')
def locoprop_backward(x, grad_out, grad_in, ln_w, ln_b, W1, b1, W2, b2, BLOCK, GROUPS, EPS=1e-5):
    B, C = x.shape
    x_s = shuffle(x, GROUPS)
    mu = x_s.mean(-1, keepdim=True)
    sigma = (x_s.var(-1, unbiased=False, keepdim=True) + EPS).sqrt()
    x_hat = (x_s - mu) / sigma
    ln_out = (x_hat * ln_w + ln_b).reshape(B, GROUPS, BLOCK)
    mask = (torch.einsum('bgi,gij->bgj', ln_out, W1) + b1) > 0

    d_out = grad_out.reshape(B, GROUPS, -1)
    dh = torch.einsum('bgj,gij->bgi', d_out, W2) * mask
    d_ln = torch.einsum('bgj,gij->bgi', dh, W1).reshape(B, C)

    dx_hat = d_ln * ln_w
    dx_s = (C * dx_hat - dx_hat.sum(-1, keepdim=True)
            - x_hat * (dx_hat * x_hat).sum(-1, keepdim=True)) / (C * sigma)
    grad_in.copy_(shuffle(dx_s, C // GROUPS) + grad_out)


class ShuffleMLPBlock(nn.Module):
    def __init__(self, groups: int, block: int):
        super().__init__()
        self.groups, self.block = groups, block
        dim = groups * block
        self.ln = nn.LayerNorm(dim)
        self.W1 = nn.Parameter(torch.empty(groups, block, block))
        self.b1 = nn.Parameter(torch.zeros(groups, block))
        self.W2 = nn.Parameter(torch.empty(groups, block, block))
        self.b2 = nn.Parameter(torch.zeros(groups, block))

        for g in range(groups):
            torch.nn.init.orthogonal_(self.W1[g])
            torch.nn.init.orthogonal_(self.W2[g])

    def forward(self, x: Tensor) -> Tensor:
        return loco_fwd(x, self.ln.weight, self.ln.bias, self.W1, self.b1, self.W2, self.b2, self.groups, self.block)


class LocoShuffleMLPBlock(ShuffleMLPBlock):
    def __init__(self, groups: int, block: int, loco_steps: int, lr: float, wd: float = 0.0,
                 inner_beta: float = 0.0):
        super().__init__(groups, block)
        self.loco_steps, self.lr, self.wd, self.inner_beta = loco_steps, lr, wd, inner_beta

    def loco_step(self, train_x: Tensor, target: Tensor):
        locoprop_step(train_x, target, self.ln.weight, self.ln.bias, self.W1, self.b1, self.W2, self.b2, self.lr,
                      self.loco_steps, GROUPS=self.groups, beta=self.inner_beta)

    def backward_step(self, x: Tensor, grad_out: Tensor, grad_in: Tensor):
        """Fused backward: recompute forward from x, backprop grad_out -> grad_in."""
        locoprop_backward(x, grad_out, grad_in, self.ln.weight, self.ln.bias, self.W1, self.b1, self.W2, self.b2,
                          BLOCK=self.block, GROUPS=self.groups)


class DenseMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.W1 = nn.Parameter(torch.empty(dim, hidden))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.W2 = nn.Parameter(torch.empty(hidden, dim))
        self.b2 = nn.Parameter(torch.zeros(dim))
        torch.nn.init.orthogonal_(self.W1)
        torch.nn.init.orthogonal_(self.W2)

    def forward(self, x: Tensor) -> Tensor:
        y = self.ln(x)
        h = y @ self.W1 + self.b1
        a = F.relu(h)
        return x + a @ self.W2 + self.b2


class ResidualMLP(nn.Module):
    def __init__(self, block_cls, n_layers: int, autograd_targets: bool = False, **kw):
        super().__init__()
        self.layers = nn.ModuleList([block_cls(**kw) for _ in range(n_layers)])
        self.is_loco = block_cls == LocoShuffleMLPBlock
        self.autograd_targets = autograd_targets

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    @torch.no_grad()
    def loco_forward(self, train_x: Tensor, target: Tensor, opt, target_lr: float = 1.0) -> Tensor:
        """Staged O(n²) LocoProp: for each layer i, forward to end, backprop to layer i, take a step."""
        depth = len(self.layers)
        init_loss = None
        for i in range(depth):
            y = self.layers[i](train_x)
            xs = []
            for j in range(i + 1, depth):
                xs.append(y)
                y = self.layers[j](y)
            d_y = y - target
            if i == 0:
                init_loss = d_y.square().mean()
            grad_in = torch.empty_like(d_y)
            for j in range(depth - 1, i, -1):
                self.layers[j].backward_step(xs.pop(), d_y, grad_in)
                grad_in, d_y = d_y, grad_in
            self.layers[i].loco_step(train_x, target_lr * d_y.double())
            opt.step()
            opt.zero_grad(set_to_none=True)
            train_x = self.layers[i](train_x)
        return init_loss

    @torch.no_grad()
    def loco_forward_autograd(self, train_x: Tensor, target: Tensor, target_lr: float = 1.0) -> Tensor:
        """O(n) baseline: single backward sweep to get per-layer targets, then local optimization."""
        x = train_x
        ys = [x := layer(x) for layer in self.layers]
        d_y = x - target
        loss = d_y.square().mean()

        grad_storage = torch.empty_like(d_y)
        for inp, layer in zip(reversed([train_x] + ys[:-1]), reversed(self.layers)):
            layer.loco_step(inp, d_y.double() * target_lr)
            layer.backward_step(inp, d_y, grad_storage)
            d_y, grad_storage = grad_storage, d_y
        return loss


class AdamWLocoSign(torch.optim.Optimizer):
    """AdamW magnitude (on real gradients) with locoprop direction.

    `p.grad` holds the true backprop gradient — Adam's moments and per-element step
    magnitude are computed from it as usual. `p.loco_grad = current - proposed` is the
    gradient-aligned locoprop update; we descend along its negated sign with |adam_step|.

    Calibration knobs (matter at scale, mostly harmless on small toys):
      `eps`         default 1e-6 (was 1e-8). For bf16 params with `state_fp32=False`,
                    1e-8 is below `v.sqrt()` precision when grads are small, and the
                    `mag = |m| / (sqrt(v) + eps)` ratio explodes on those elements.
                    1e-6 is the standard transformers/HF default for bf16 training.
      `state_fp32`  default True. Keeps `m, v` in fp32 even when params are bf16.
                    The bf16 `addcmul_(g, g)` for `v` rounds away contributions when
                    the existing `v` is much larger than `(1-β2)g²`, so `v` stalls at
                    a stale value and `mag` drifts. fp32 state costs 2× param memory
                    (~64GB extra at 8B); usually worth it.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0,
                 state_fp32=True):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self._state_fp32 = state_fp32

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group['lr'], group['betas'], group['eps'], group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st['step'] = 0
                    state_dtype = torch.float32 if self._state_fp32 else p.dtype
                    st['m'] = torch.zeros_like(p, dtype=state_dtype)
                    st['v'] = torch.zeros_like(p, dtype=state_dtype)
                st['step'] += 1
                t = st['step']
                m, v = st['m'], st['v']
                gf = g.float() if m.dtype == torch.float32 and g.dtype != torch.float32 else g
                m.mul_(b1).add_(gf, alpha=1 - b1)
                v.mul_(b2).addcmul_(gf, gf, value=1 - b2)
                bc1 = 1 - b1 ** t
                bc2_sqrt = math.sqrt(1 - b2 ** t)
                denom = v.sqrt().div_(bc2_sqrt).add_(eps)
                mag = m.abs().div_(denom).mul_(lr / bc1)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                lg = getattr(p, 'loco_grad', None)
                direction = lg.sign().neg_() if lg is not None else m.sign().neg_()
                p.addcmul_(direction.to(p.dtype), mag.to(p.dtype))


class Teacher(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.inproj = nn.Linear(dim, hidden)
        self.outproj = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.outproj(F.relu(self.inproj(x)))

@torch.compile(mode='max-autotune-no-cudagraphs')
def data(teacher, batch, dim):
    src = torch.randn((batch, dim), device=teacher.inproj.weight.device)
    return src, teacher(src)


def train(model: ResidualMLP, train_steps: int, batch: int, dim: int, lr: float, device: str, print_every: int,
          loco_opt: str = 'sign', return_full_trajectory: bool = False, weight_decay: float = 0.1):
    target_lr = 1 if model.is_loco else None
    if model.is_loco and loco_opt == 'sign':
        opt = AdamWLocoSign(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif model.is_loco and loco_opt in ('polyak', 'loco_nesterov'):
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    elif model.is_loco and loco_opt in ('graft', 'loco_sgd'):
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, fused=True, weight_decay=weight_decay)
    use_loco_grad = model.is_loco and loco_opt in ('locograd', 'loco_nesterov', 'loco_sgd')
    use_graft = model.is_loco and loco_opt == 'graft'
    train_losses = []
    full_losses = torch.empty((train_steps,), device=device, dtype=torch.float64) if return_full_trajectory else None
    last_recorded = -1
    loss_buffer = torch.zeros((print_every,), device=device, dtype=torch.float64)

    teacher = Teacher(dim, dim).to(device)
    model: ResidualMLP = torch.compile(model, mode='max-autotune-no-cudagraphs')
    params = [p for p in model.parameters() if p.requires_grad]

    def compute_grads(src_, tgt_):
        if model.is_loco:
            if model.autograd_targets:
                return model.loco_forward_autograd(src_, tgt_, target_lr=target_lr)
            with torch.no_grad():
                return model.loco_forward(src_, tgt_, opt, target_lr=target_lr)
        y_pred_ = model(src_)
        loss_ = F.mse_loss(y_pred_, tgt_)
        loss_.backward()
        return loss_

    # Untimed warmup: trigger compile of inner kernels. Always use autograd path
    # (same compiled kernels as staged) since it leaves model state untouched.
    with torch.no_grad():
        src, tgt = data(teacher, batch, dim)
    if model.is_loco:
        model.loco_forward_autograd(src, tgt, target_lr=target_lr)
    else:
        F.mse_loss(model(src), tgt).backward()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    start = time.perf_counter()

    best_loss = float('inf')
    for step in range(train_steps):
        with torch.no_grad():
            src, tgt = data(teacher, batch, dim)
        loss = compute_grads(src, tgt)
        if full_losses is not None:
            full_losses[step] = loss.detach()
            last_recorded = step
        if use_loco_grad:
            for p in params:
                lg = getattr(p, 'loco_grad', None)
                if lg is not None:
                    p.grad = lg
        elif use_graft:
            for p in params:
                lg = getattr(p, 'loco_grad', None)
                if lg is not None:
                    p.grad = lg.sign() * (lg.norm() / lg.numel() ** 0.5)
        opt.step()
        opt.zero_grad()
        loss_buffer[step % print_every] = loss.detach()
        if (step + 1) % print_every == 0:
            with torch.no_grad():
                buf = loss_buffer.cpu()
                avg = buf.log().mean().exp().item()
                print(f"  {step + 1:5d} | train {avg:.6f}")
                train_losses.extend(buf.tolist())
                if not np.isfinite(avg) or avg > 2 * best_loss:
                    break
                best_loss = min(best_loss, avg)

    torch.cuda.synchronize()
    if full_losses is not None:
        train_losses = full_losses[:last_recorded + 1].cpu().numpy()
    else:
        train_losses = np.array(train_losses)
    return {"time": time.perf_counter() - start, "train_losses": train_losses, "teacher": teacher}


def run_varying_lr(lr_range, model, epochs, batch, dim, device, print_every):
    return {lr: train(copy.deepcopy(model), epochs, batch, dim, lr, device, print_every) for lr in lr_range}


@app.command()
def main(groups: int = typer.Option(16, help="Number of parallel groups"),
         block: int = typer.Option(64, help="Block size per group"), batch: int = typer.Option(2 ** 18, help="Batch size"),
         layers: int = typer.Option(8, help="Number of layers"),
         epochs: int = typer.Option(4096, help="Training epochs"), lr: float = typer.Option(0.0001, help="Learning rate"),
         wd: float = typer.Option(0.0, help="Weight decay toward original weights in inner loop"),
         steps: list[int] = typer.Option([1, 16, 256], help="LocoProp steps"),
         seed: int = typer.Option(42, help="Random seed"), print_every: int = typer.Option(256, help="Print interval"),
         device: str = typer.Option("cuda", help="Device"), ):
    config = locals()
    name = ''.join(f'{k}={v}' for k, v in config.items() if isinstance(v, int))

    torch.manual_seed(seed)
    dim = groups * block

    print(f"ShuffleNet LocoProp | groups={groups} block={block} dim={dim} layers={layers}")
    print("=" * 70)

    results = {}
    lrs = np.logspace(-3, -0, 2)

    for s in steps:
        print(f"\n[LocoProp {s} - Autograd O(n)]")
        torch.manual_seed(seed + 1)
        model = ResidualMLP(LocoShuffleMLPBlock, layers, groups=groups, block=block, loco_steps=s, lr=lr, wd=wd,
                            autograd_targets=True).to(device)
        results[f"loco_{s}_autograd"] = run_varying_lr(lrs, model, epochs, batch, dim, device, print_every)
    with open(f'results/{name}.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    app()
