import copy
import functools
import math
import pickle
import time
from typing import List

import heavyball
import numpy as np
import torch
import typer
from sympy import prevprime
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


def unshuffle(x: Tensor, groups: int) -> Tensor:
    B, D = x.shape
    block = D // groups
    return x.reshape(B, groups, block).transpose(-1, -2).contiguous().reshape(B, D)

@torch.compile(mode='max-autotune-no-cudagraphs')
def loco_fwd(x, ln_w, ln_b, W1, b1, W2, b2, groups, block, eps=1e-5, residual=True):
    x_s = shuffle(x, groups)
    x_s = F.layer_norm(x_s, (x.size(-1),), weight=ln_w, bias=ln_b, eps=eps)
    x_s = x_s.reshape(-1, groups, block)
    h = torch.einsum('bgi,gij->bgj', x_s, W1) + b1
    a = F.relu(h)
    out = torch.einsum('bgi,gij->bgj', a, W2) + b2
    return out.reshape(x_s.size(0), -1) + (x if residual else 0)

@torch.compile(mode='max-autotune-no-cudagraphs')
def bwd(x, ln_w, ln_b, W1, b1, W2, b2, groups, grad_on_output_hidden, compute_target, eps=1e-5):
    B, D = x.shape
    x_s = shuffle(x, groups)
    mu = x_s.mean(-1, keepdim=True)
    var = x_s.var(-1, unbiased=False, keepdim=True)
    x_hat = (x_s - mu) / (var + eps).sqrt()
    x_ln = x_hat * ln_w + ln_b
    x_r = x_ln.reshape(B, groups, -1)
    h = torch.einsum('bgi,gij->bgj', x_r, W1) + b1
    mask = (h > 0).float()
    a = h * mask
    y = torch.einsum('bgi,gij->bgj', a, W2) + b2
    if compute_target:
        target = y.double() - grad_on_output_hidden.reshape(B, groups, -1).double()
    else:
        target = grad_on_output_hidden
    dout = (y.double() - target).mul(1 / y.numel()).to(y.dtype)

    db2 = dout.sum(0)
    dW2 = torch.einsum('bgi,bgj->gij', a, dout)
    da = torch.einsum('bgj,gji->bgi', dout, W2.transpose(-1, -2))
    dh = da * mask
    db1 = dh.sum(0)
    dW1 = torch.einsum('bgi,bgj->gij', x_r, dh)

    dx_ln = torch.einsum('bgj,gji->bgi', dh, W1.transpose(-1, -2)).reshape(B, D)
    dln_b = dx_ln.sum(0)
    dln_w = (dx_ln * x_hat).sum(0)

    return dout, target, dln_w, dln_b, dW1, db1, dW2, db2


@torch.compile(mode='max-autotune-no-cudagraphs')
def optim(params, grads, lr):
    for p, g in zip(params, grads):
        p -= lr * g


@torch.compile(mode='max-autotune-no-cudagraphs')
def nesterov_update(params, grads, mom, lr, beta):
    for p, g, m in zip(params, grads, mom):
        m.mul_(beta).sub_(g, alpha=lr)
        p.add_(m)


@torch.compile(mode='max-autotune-no-cudagraphs')
def nesterov_lookahead(out, params, mom, beta):
    for o, p, m in zip(out, params, mom):
        torch.add(p, m, alpha=beta, out=o)


def locoprop_step(train_x, grad_on_output_hidden, ln_w, ln_b, W1, b1, W2, b2, lr, n_steps, GROUPS,
                   beta=0.0):
    """Inner SGD on local regression from current params. beta>0 enables Nesterov acceleration.
    At n=1: (p-c) = lr · ∇L_local = lr/2 · g_BP (mom is zero at step 0, so Nesterov ≡ SGD there)."""
    params_orig = params = ln_w, ln_b, W1, b1, W2, b2
    real_grads = None
    param_copy = [p.clone() for p in params]
    if beta > 0:
        mom = [torch.zeros_like(p) for p in param_copy]
        lookahead = [torch.empty_like(p) for p in param_copy]
        for i in range(n_steps):
            if i > 0:
                nesterov_lookahead(lookahead, param_copy, mom, beta)
                la = lookahead
            else:
                la = param_copy
            dout, grad_on_output_hidden, *grads = bwd(train_x, *la, GROUPS, grad_on_output_hidden, i == 0)
            if i == 0:
                real_grads = grads
            nesterov_update(param_copy, grads, mom, lr, beta)
    else:
        for i in range(n_steps):
            dout, grad_on_output_hidden, *grads = bwd(train_x, *param_copy, GROUPS, grad_on_output_hidden, i == 0)
            if i == 0:
                real_grads = grads
            optim(param_copy, grads, lr)
    for p, c, g in zip(params_orig, param_copy, real_grads):
        p.grad = g
        p.loco_dir = c - p
        p.loco_as_grad = p - c


@torch.compile(mode='max-autotune-no-cudagraphs')
def locoprop_backward(x, grad_out, grad_in, ln_w, ln_b, W1, b1, W2, b2, BLOCK, GROUPS, EPS=1e-5):
    B, C = x.shape

    # recompute forward
    x_s = shuffle(x, GROUPS)
    mu = x_s.mean(-1, keepdim=True)
    var = x_s.var(-1, unbiased=False, keepdim=True)
    sigma = (var + EPS).sqrt()
    x_hat = (x_s - mu) / sigma
    ln_out = (x_hat * ln_w + ln_b).reshape(B, GROUPS, BLOCK)
    h = torch.einsum('bgi,gij->bgj', ln_out, W1) + b1
    mask = h > 0

    # backward through W2
    d_out = grad_out.reshape(B, GROUPS, -1)
    da = torch.einsum('bgj,gij->bgi', d_out, W2)

    # backward through relu + W1
    dh = da * mask
    d_ln = torch.einsum('bgj,gij->bgi', dh, W1).reshape(B, C)

    # backward through layer norm
    dx_hat = d_ln * ln_w
    dx_s = (1.0 / (C * sigma)) * (
            C * dx_hat - dx_hat.sum(-1, keepdim=True) - x_hat * (dx_hat * x_hat).sum(-1, keepdim=True))

    # backward through shuffle (inverse permutation)
    dx = shuffle(dx_s, C // GROUPS)
    grad_in.copy_(dx + grad_out)  # residual


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

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass using fused Triton kernel.
        Computes y = f(x) + x where f is LayerNorm + MLP.
        """
        y = torch.empty_like(x)
        self.forward_step(x, y)
        return y

    def forward_step(self, x: Tensor, output_tensor: Tensor):
        output_tensor.copy_(super().forward(x))

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
            x = layer(x)  # layer.forward() includes residual
        return x

    @torch.no_grad()
    def loco_forward(self, train_x: Tensor, target: Tensor, opt, target_lr: float = 1.0) -> tuple[Tensor, Tensor]:
        """
        Layer-by-layer LocoProp training.
        For each layer i:
          1. Forward through layers i to end to get final prediction
          2. Compute gradient at layer i output via backprop
          3. Set target = current_output - target_lr * gradient (move towards lower loss)
          4. Run loco_step which trains layer i weights and updates train_x in-place

        Returns:
            (train_x, init_loss): train_x is the final output, init_loss is the
                                  loss from the first forward pass (before training).
        """
        train_x = train_x.clone()
        depth = len(self.layers)
        init_loss = None

        for i in range(depth):
            y_i = y = self.layers[i](train_x)
            xs = []
            for j in range(i + 1, depth):
                xs.append(y)
                new = torch.empty_like(y)
                self.layers[j].forward_step(y, new)
                y = new

            d_y = y - target

            # Compute initial loss before any training
            if i == 0:
                init_loss = (d_y ** 2).mean()

            grad_in = torch.empty_like(d_y)
            for j in range(depth - 1, i, -1):
                self.layers[j].backward_step(xs.pop(-1), d_y, grad_in)
                grad_in, d_y = d_y, grad_in

            self.layers[i].loco_step(train_x, target_lr * d_y.double())
            opt.step()
            opt.zero_grad(set_to_none=True)
            train_x = self.layers[i](train_x)

        return None, init_loss

    @torch.no_grad()
    def loco_forward_autograd(self, train_x: Tensor, target: Tensor, target_lr: float = 1.0) -> tuple[Tensor, Tensor]:
        """O(n) baseline: single autograd backward to get all targets, then local optimization."""
        x = train_x.clone().requires_grad_(True)
        ys = [x := layer(x) for layer in self.layers]

        d_y = x - target
        loss = d_y.square().mean()

        xs = [train_x.detach()] + ys[:-1]
        grad_storage = torch.empty_like(d_y)
        for layer in self.layers[::-1]:
            x = xs.pop(-1).detach()
            layer.loco_step(x, d_y.double() * target_lr)
            layer.backward_step(x, d_y, grad_storage)
            d_y, grad_storage = grad_storage, d_y

        return None, loss.detach()


class AdamWLocoSign(torch.optim.Optimizer):
    """AdamW magnitude (on real gradients) with locoprop direction.

    `p.grad` holds the true backprop gradient — Adam's moments and per-element step
    magnitude are computed from it as usual. `p.loco_dir = proposed - current` is the
    locoprop-proposed update direction; we keep its element-wise sign and scale by |adam_step|.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

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
                    st['m'] = torch.zeros_like(p)
                    st['v'] = torch.zeros_like(p)
                st['step'] += 1
                t = st['step']
                m, v = st['m'], st['v']
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                bc1 = 1 - b1 ** t
                bc2_sqrt = math.sqrt(1 - b2 ** t)
                denom = v.sqrt().div_(bc2_sqrt).add_(eps)
                mag = m.abs().div_(denom).mul_(lr / bc1)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                loco_dir = getattr(p, 'loco_dir', None)
                direction = loco_dir.sign() if loco_dir is not None else m.sign().neg_()
                p.addcmul_(direction, mag)


@functools.lru_cache()
def cached_prevprime(n):
    return prevprime(n)


@torch.compile(mode='max-autotune-no-cudagraphs', fullgraph=True)
def solve_loss(template: Tensor, fn, steps: int):  # SGD without momentum finds the optimum very fast and consistently
    yp = torch.zeros_like(template)
    for i in range(steps):
        loss, vjp_fn = torch.func.vjp(fn, yp)
        d_yp, = vjp_fn(torch.ones_like(loss))

        lr = min(i + 1, steps - i) / steps
        yp -= d_yp * lr
    return yp.detach()


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
    tgt = teacher(src).detach()
    src = src.detach()
    return src, tgt


def train(model: ResidualMLP, train_steps: int, batch: int, dim: int, lr: float, device: str, print_every: int,
          numbers: int | None = None, loss_fn=F.mse_loss, loss_steps: int = 32,
          noise_std: float = 0.0, sam_rho: float = 0.0,
          loco_opt: str = 'sign', return_full_trajectory: bool = False):
    if model.is_loco:
        target_lr = 1
    else:
        target_lr = None
    if model.is_loco and loco_opt == 'sign':
        opt = AdamWLocoSign(model.parameters(), lr=lr, weight_decay=0.1)
    elif model.is_loco and loco_opt == 'graft':
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    elif model.is_loco and loco_opt == 'polyak':
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    elif model.is_loco and loco_opt == 'loco_nesterov':
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    elif model.is_loco and loco_opt == 'loco_sgd':
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, fused=True, weight_decay=0.1)
    use_loco_grad = model.is_loco and loco_opt in ('locograd', 'loco_nesterov', 'loco_sgd')
    use_graft = model.is_loco and loco_opt == 'graft'
    use_polyak = model.is_loco and loco_opt == 'polyak'
    train_losses = []
    full_losses = torch.empty((train_steps + 1,), device=device, dtype=torch.float64) if return_full_trajectory else None
    last_recorded = -1

    loss_buffer = torch.zeros((print_every,), device=device, dtype=torch.float64)

    teacher = Teacher(dim, dim).to(device)
    model: ResidualMLP = torch.compile(model, mode='max-autotune-no-cudagraphs')

    params = [p for p in model.parameters() if p.requires_grad]

    def compute_grads(src_, tgt_):
        if model.is_loco:
            if model.autograd_targets:
                _, loss_ = model.loco_forward_autograd(src_, tgt_, target_lr=target_lr)
            else:
                with torch.no_grad():
                    _, loss_ = model.loco_forward(src_, tgt_, opt, target_lr=target_lr)
        else:
            y_pred_ = model(src_)
            loss_ = loss_fn(y_pred_, tgt_)
            loss_.backward()
        return loss_

    start = 0

    best_loss = float('inf')

    for step in range(train_steps + 1):
        with torch.no_grad():
            src, tgt = data(teacher, batch, dim)
            if noise_std > 0:
                src = src + noise_std * torch.randn_like(src)

        loss = compute_grads(src, tgt)
        if full_losses is not None:
            full_losses[step] = loss.detach()
            last_recorded = step

        if use_loco_grad:
            for p in params:
                lg = getattr(p, 'loco_as_grad', None)
                if lg is not None:
                    p.grad = lg
        elif use_graft:
            for p in params:
                lg = getattr(p, 'loco_as_grad', None)
                if lg is not None:
                    p.grad = lg.sign() * (lg.norm() / lg.numel() ** 0.5)

        if sam_rho > 0:
            # Param-space SAM: ascend along current gradient, recompute at θ', restore θ, apply step using θ' direction
            orig = [p.data.clone() for p in params]
            with torch.no_grad():
                gnorm_sq = sum((p.grad.pow(2).sum() for p in params if p.grad is not None), start=torch.zeros((), device=device))
                scale = sam_rho / (gnorm_sq.sqrt() + 1e-12)
                for p in params:
                    if p.grad is not None:
                        p.data.add_(p.grad, alpha=scale.item())
                        p.grad = None
                    if getattr(p, 'loco_dir', None) is not None:
                        p.loco_dir = None
            opt.zero_grad(set_to_none=True)
            loss = compute_grads(src, tgt)
            if full_losses is not None:
                full_losses[step] = loss.detach()
            with torch.no_grad():
                for p, o in zip(params, orig):
                    p.data.copy_(o)
        opt.step()
        opt.zero_grad()

        loss_buffer[step % print_every] = loss.detach()

        if step == 0:
            torch.cuda.synchronize()
            start = time.perf_counter()
            continue

        step += 1  # we have now done the step and should log the first datapoint as 1st not 0th -> 10/20 not 9/19
        if step % print_every == 0:
            with torch.no_grad():
                buf = loss_buffer.cpu()
                avg = buf.log().mean().exp().item()
                print(f"  {step:5d} | train {avg:.6f}")
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
         steps: List[int] = typer.Option([1, 16, 256], help="LocoProp steps"),
         seed: int = typer.Option(42, help="Random seed"), print_every: int = typer.Option(256, help="Print interval"),
         device: str = typer.Option("cuda", help="Device"), ):
    config = locals()
    name = ''.join(f'{k}={v}' for k, v in config.items() if isinstance(v, int))

    torch.manual_seed(seed)
    dim = groups * block

    print(f"ShuffleNet LocoProp | groups={groups} block={block} dim={dim} layers={layers}")
    print("=" * 70)

    results = {}
    lrs = np.logspace(-3, -0, 2)  # 10 ** -8 to 10 ** -2

    # print("\n[Backprop - Shuffle]")
    # torch.manual_seed(seed + 1)
    # model = ResidualMLP(ShuffleMLPBlock, layers, groups=groups, block=block).to(device)
    # results["backprop_shuffle"] = run_varying_lr(lrs, model, epochs, batch, dim, device, print_every)

    # sqrt_scale = 2 ** round((groups.bit_length() - 1) / 2)  # nearest pow2 to sqrt(groups)
    # hidden = sqrt_scale * block
    # print(f"\n[Backprop - Dense (hidden={hidden}, {dim / hidden:.1f}x bottleneck)]")
    # torch.manual_seed(seed + 1)
    # model = ResidualMLP(DenseMLPBlock, layers, dim=dim, hidden=hidden).to(device)
    # results["backprop_dense"] = run_varying_lr(lrs, model, epochs, batch, dim, device, print_every)

    for s in steps:
        # print(f"\n[LocoProp {s} - Staged O(n²)]")
        # torch.manual_seed(seed + 1)
        # model = ResidualMLP(LocoShuffleMLPBlock, layers, groups=groups, block=block, loco_steps=s, lr=lr, wd=wd,
        #                     autograd_targets=False).to(device)
        # results[f"loco_{s}_staged"] = run_varying_lr(lrs, model, epochs, batch, dim, device, print_every)

        print(f"\n[LocoProp {s} - Autograd O(n)]")
        torch.manual_seed(seed + 1)
        model = ResidualMLP(LocoShuffleMLPBlock, layers, groups=groups, block=block, loco_steps=s, lr=lr, wd=wd,
                            autograd_targets=True).to(device)
        results[f"loco_{s}_autograd"] = run_varying_lr(lrs, model, epochs, batch, dim, device, print_every)
    with open(f'results/{name}.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    app()
