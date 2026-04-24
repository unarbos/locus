# LocoProp `n=1` equivalence: derivation, bug, and fix

## What LocoProp actually computes

For each layer `i` with parameters θ_i, input x_i (activation from previous layer), and a **target**
a_i (a vector in the output space of layer i), LocoProp solves

    θ_i ← argmin_θ ‖f_i(x_i; θ) − a_i‖²                     (local regression)

by running a few steps of an inner optimizer starting from the current θ_i. The trained-toward
parameters c_i are not used directly; instead the outer optimizer sees the *proposed update*

    Δ_i = c_i − θ_i          (`p.loco_dir`)
    g̃_i = θ_i − c_i          (`p.loco_as_grad`, acts as a pseudo-gradient)

### Where the target comes from

The standard LocoProp target is

    a_i = y_i − η · ∂L/∂y_i                                (y_i = layer i's output)

— "move this layer's output a little in the descent direction." Plugging into the local regression,
the **first** inner gradient is

    ∇_θ ½‖f_i(x_i; θ) − a_i‖²  at θ = θ_i
       = J_i^T · (y_i − a_i)
       = J_i^T · (η · ∂L/∂y_i)
       = η · ∂L/∂θ_i                                       (exact BP gradient, up to η)

So **with `n_steps=1`, plain SGD inner, and the right scaling, LocoProp is literally BP.** The
equivalence is not an approximation, it's an identity. Any empirical divergence between
`loco_as_grad` and `autograd.grad` at n=1 is a bug.

### Exact scaling in our code

- `F.mse_loss` uses factor `2/N`: `∂L/∂y = (2/N)(y − tgt)`.
- `bwd` in `main.py` uses factor `1/N` and is called with `grad_on_output_hidden = y − tgt` (no 2/N),
  so what `bwd` returns as param-grad equals `(1/N) · J^T · (y − tgt)`, which is `½ · ∂L/∂θ` from `F.mse_loss`.
- One inner SGD step: `c = p − lr · bwd_grad`. So `loco_as_grad = p − c = lr · bwd_grad = (lr/2) · ∂L/∂θ`.

→ **`inner_lr = 2.0` makes `loco_as_grad` bit-equal to the BP gradient** at n=1.

The outer optimizer then does `p -= outer_lr · loco_as_grad`, which is BP+(your-outer) at
`effective_lr = outer_lr`. Nothing exotic.

## The bug

In `ResidualMLP.loco_forward_autograd`, the reverse sweep looked like:

```python
grad_storage = torch.empty_like(d_y)
for layer in self.layers[::-1]:
    x = xs.pop(-1)
    layer.backward_step(x.clone(), d_y.clone(), grad_storage)   # writes grad_storage
    layer.loco_step(x.clone(), d_y.double() * target_lr)        # reads d_y
    d_y = grad_storage                                          # aliases next iter's target
```

`backward_step` ultimately calls `locoprop_backward`, whose last line is

    grad_in.copy_(dx + grad_out)     # in-place into `grad_storage`

After iteration `k` finishes, `d_y` and `grad_storage` point to the **same tensor**. Iteration
`k+1` begins with `backward_step(..., d_y.clone(), grad_storage)` — the `.clone()` snapshots the
current (correct) `d_y` for `backward_step`'s own use, but the `copy_` at the end of
`backward_step` overwrites `grad_storage` in place, and because `d_y` aliases it, `d_y`'s content
silently becomes the *input* gradient of the current layer before `loco_step` runs.

So `loco_step` on layer `k` was being fed the d_y that belonged to layer `k−1`. Off by one layer.

### Why the drift pattern matched the bug

At layer `N−1` (deepest), `d_y` is still the freshly computed `y − target` (not aliased yet):
loco_step sees the right thing → ratio 1.0000, cos 1.0000.

Every layer earlier in the sweep sees a d_y that has already been advanced one step through
`locoprop_backward`. Because consecutive layer-output gradients are *similar* (residual network,
each `J_i^T` is close to identity + perturbation), the error is small per layer but accumulates.
Observed drift at layer 6: ratio 1.04, cos 0.97; at layer 0: ratio ≈ 1.10, cos ≈ 0.90 — classic
off-by-one-layer-in-a-deep-chain signature.

## The fix

```python
for layer in self.layers[::-1]:
    x = xs.pop(-1)
    layer.loco_step(x.clone(), d_y.double() * target_lr)        # read d_y first
    layer.backward_step(x.clone(), d_y, grad_storage)           # then overwrite grad_storage
    d_y, grad_storage = grad_storage, d_y                       # swap buffers — no aliasing
```

Two changes:

1. **Order**: `loco_step` runs before `backward_step`. At this point `d_y` still holds the
   output gradient of the current layer, which is exactly what the local regression target needs.

2. **Swap, not alias**: after `backward_step` writes `grad_storage`, we *swap* `d_y ↔ grad_storage`.
   The new `d_y` is the buffer that just got written (it holds the input gradient of the current
   layer = output gradient of the next layer). The new `grad_storage` is the *old* `d_y` buffer,
   which is free to be overwritten. Neither aliases the other — the next iteration's `backward_step`
   will read `d_y` and write `grad_storage`, which are now different tensors.

Re-read the `loco_step` isolation argument: `locoprop_step` clones parameters (`param_copy = [p.clone() for p in params]`) and only writes attributes `p.grad`, `p.loco_dir`, `p.loco_as_grad` on the originals. No layer weight changes during the reverse sweep, so `backward_step` after `loco_step` sees the same weights autograd sees. Weight updates are deferred to `opt.step()` after the sweep completes.

## Verification

- **`debug_scale.py`**: compares `p.loco_as_grad` (n=1, inner_lr=2) against `torch.autograd.grad` per
  parameter of each of 8 layers. Post-fix: `ratio = 1.0000`, `cos = 1.0000` for every parameter of
  every layer. Pre-fix: layer 7 clean, drift begins at layer 6.
- **Training curves**: BP+AdamW vs n=1-LocoProp+AdamW with `inner_lr=2.0` are numerically identical
  through 512 epochs at two outer LRs (1e-3 and 3e-3). Tail losses match to 4 decimals.
- **`debug_chain.py`**: independent test that `locoprop_backward` composed over 8 layers (no
  `loco_step` in between) matches autograd-through-the-full-forward to cos=1.0000 at every layer.
  Rules out any systematic error in the manual backward kernel — confirms the bug was purely the
  alias in the outer loop.

## What this says about higher n

With `n=1 ≡ BP` established, the question of whether n > 1 helps becomes well-posed. At n inner
steps, `c_i` approaches the argmin of the local regression — the full Gauss-Newton step for
layer i in isolation (with fixed target). The outer update is then a GN-preconditioned direction
per layer, not a gradient.

### Why plain-SGD inner hides the benefit

The local regression `½‖f_i(x_i; θ) − a_i‖²` is a quadratic-like objective with Hessian `H ≈
J_i^T J_i`. Plain SGD on this objective converges at rate `(1 − 1/κ)^n` where κ is the condition
number of H. For a grouped-linear-plus-ReLU-plus-LayerNorm block, κ is in the tens, meaning SGD
takes on the order of κ steps (tens to hundreds) to actually get close to the minimum.

At n=16 plain-SGD, `c_i` has moved maybe 5% of the way toward the GN solution. Most of the
"high-n improvement" budget is being spent waiting for first-order inner convergence.

**Nesterov acceleration** improves the rate to `(1 − 1/√κ)^n` — sqrt of the condition number —
which in practice means n=16 Nesterov ≈ n=100+ plain SGD for the same local residual. The
update rule:

```
v_{t+1} = β·v_t − lr·∇L_local(θ_t + β·v_t)
θ_{t+1} = θ_t + v_{t+1}
```

At step 0 `v=0`, so the lookahead point equals `θ_0` and the first update is `−lr·∇L_local(θ_0)`
— identical to plain SGD. **The n=1 ≡ BP identity is preserved for any β.** The sqrt-speedup
kicks in at n ≥ 2.

### Why AdamW outer is still required

With loco_as_grad already Gauss-Newton-preconditioned per layer, a natural hypothesis is
"SGD outer should now suffice." Empirically it doesn't. Scanning SGD outer LR at n=64 (1024
epochs, Nesterov inner β=0.9 lr=1.0):

| SGD outer LR | tail | eval | nMSE |
|---|---|---|---|
| 0.01 | 0.982 | 0.967 | 17.4 |
| 0.1 | NaN | — | — |
| 0.3, 1.0, 3.0 | NaN | — | — |
| SGD-Nesterov 0.3, 1.0 | NaN | — | — |

There is no stable regime. At LR=0.01 the update is too small to escape identity (note: eval
nMSE 17× means model output has blown up in magnitude relative to teacher — not literally
at identity, but producing a near-random-scale residual stack). At LR ≥ 0.1, divergence.

The reason is *per-parameter scale variance*, not just per-layer. `loco_as_grad` components
span many orders of magnitude within a single parameter tensor (LayerNorm γ/β vs projection
weights vs biases, with GN preconditioning amplifying the spread). A single scalar LR cannot
simultaneously be (a) large enough for small-magnitude entries to move and (b) small enough
for large-magnitude entries to stay stable. Adam's per-element `√v` normalization is
load-bearing here, not ornamental.

Loco's GN per layer + Adam's per-parameter scaling compose: one provides *within-layer*
curvature adaptation, the other *across-parameter* scale equalization.

### Empirical scaling

At 2048 epochs, batch 16384, 8 layers × 1024 dim, AdamW outer 3e-3, Nesterov inner β=0.9 lr=1.0:

| config | tail | eval | nMSE |
|---|---|---|---|
| BP+AdamW 3e-3 (4096 ep) | 0.0524 | 0.0524 | 0.943 |
| BP+AdamW 3e-3 | 0.0594 | 0.0603 | 1.084 |
| BP+AdamW 1e-3 | 0.0673 | 0.0637 | 1.146 |
| BP+AdamW 1e-2 (4096 ep) | 0.315 | 0.307 | 5.52 |
| n=64 Nesterov | 0.0339 | 0.0348 | 0.626 |
| n=128 Nesterov | 0.0224 | 0.0250 | 0.449 |
| n=256 Nesterov | 0.0193 | 0.0191 | 0.343 |
| n=512 Nesterov | 0.0190 | 0.0187 | 0.337 |

Two things the ablation settled:

1. **BP floor at ~0.052 is real, not optimization stall.** Doubling BP+AdamW 3e-3 from 2048 →
   4096 epochs moves tail from 0.0594 → 0.0524 (12% relative, 2× compute). BP+AdamW 1e-2 at
   4096 epochs is still at 0.31, much worse — higher LR trades speed-early for divergent-late,
   not a better asymptote. The floor is whatever ceiling BP hits on this task (architectural
   expressivity + gradient-descent noise under MSE on Gaussian inputs).

2. **n-scaling saturates between 256 and 512.** n=256 → n=512 at 2048 epochs: 0.0193 → 0.0190
   tail, 0.0191 → 0.0187 eval. 2× compute, ~1.5% improvement. The asymptotic LocoProp floor
   on this problem is ~0.019 eval, 3.2× lower than BP's 0.052. The earlier extrapolation
   ("still halving at n=256") was wrong — it was already curving over and n=512 flattened it.

The cost is O(n) per outer step. n=256 is ~256× the bwd-kernel work per outer step.

### Decomposing the gap: how much of the 3× is GN specifically?

The naive comparison ("BP K=1 at 2048 epochs = 0.059 vs LocoProp n=256 = 0.019") conflates
two things: LocoProp at n=256 does 256 bwd-passes per outer step, BP K=1 does 1. BP might
just be under-trained. Two controls:

**BP with K Adam steps per batch (same data, K updates).** At 2048 outer batches:

| K | lr | tail | eval |
|---|---|---|---|
| 1 | 3e-3 | 0.0591 | 0.0563 |
| 4 | 3e-3 | 0.0444 | 0.0444 |
| 16 | 3e-3 | 0.0608 | 0.0427 |
| 64 | 3e-3 | 0.0380 | 0.0337 |
| 4 | 1e-3 | 0.0406 | 0.0401 |
| 16 | 1e-3 | 0.0322 | 0.0330 |
| 64 | 1e-3 | 0.0304 | 0.0316 |
| 256 | 1e-3 (partial) | ~0.0296 | — |

**BP K=1 with fresh data at long horizon (matched Adam-step budget).**

| ep | lr | tail | eval |
|---|---|---|---|
| 32768 | 1e-3 | 0.0320 | 0.0320 |
| 32768 | 3e-4 | 0.0340 | 0.0341 |
| 131072 | 1e-3 | 0.0279 | 0.0281 |
| 131072 | 3e-4 | 0.0283 | 0.0285 |

Two observations:

1. **Batch reuse is not load-bearing.** At matched Adam-step count, K=16 on 2048 batches
   (32k Adam steps) and K=1 on 32768 batches (also 32k Adam steps) both land at ~0.032 eval.
   What BP needs isn't more of the same batch, it's more Adam updates.

2. **BP saturates near 0.028 eval at the 131k-Adam-step budget** (either K=256 reuse or
   K=1 fresh). Doubling the Adam budget barely moves it. Call this BP's long-horizon floor.

So the decomposition is:

- From 0.059 (BP K=1 at 2048 ep) to 0.028 (BP at 131k Adam steps): **"BP was under-trained"**
  — accounts for ~0.48 of the log-gap.
- From 0.028 (BP floor) to 0.019 (LocoProp n=256): **GN-specific** — ~0.52 of log-gap,
  or 1.5× in linear loss.

The algorithmic gain from per-layer Gauss-Newton preconditioning is ~1.5×, not 3×. What looked
like a "3× win" was partly GN and partly "well-tuned BP at long horizon is much better than
2048-epoch BP at its default-ish LR." Both lessons matter; they are different lessons.

### Wall-clock picture

At matched bwd-pass count, LocoProp still wins (BP K=1 at 131k bwd = 0.028; LocoProp n=256
at 2048 outer × 256 inner ≈ 524k bwd = 0.019). But BP's bwd pass is cheaper than LocoProp's
inner-regression bwd (no intermediate Adam state, no autograd-target forward sweep), and the
per-sample advantage translates to a less dramatic per-wall-clock advantage. A careful
wall-clock comparison at matched compute hasn't been run yet.

### What's still open

- At matched bwd-pass budget (524k), does BP K=1 ep=524288 close further toward 0.019, or
  does it truly plateau near 0.027? Half a day of compute to check.
- Is there a regime (batch size, architecture depth, task non-convexity) where loco dominates
  wall-clock? GN preconditioning should matter more when per-step BP progress saturates for
  first-order reasons (bad conditioning, not noise). Larger batch or harder task is the natural
  sweep.
- Shampoo/Muon-style inner (orthogonalized) may further reduce the inner iteration budget by
  handling within-matrix anisotropy that Nesterov can't.

## Lessons reinforced

- A derivation that predicts `n=1 ≡ BP` doesn't predict `n > 1 >> BP`. The benefit is only
  accessible once (a) the inner iteration is efficient enough to approach the argmin, and
  (b) the outer handles the scale/variance structure that per-layer GN introduces.
- "Higher n didn't help much" was a *symptom*, not an *algorithmic verdict*. Diagnosing it
  required separating the inner-convergence question from the outer-scaling question.

## Lessons

- When a clean mathematical identity (n=1 LocoProp ≡ BP) fails empirically, assume the bug is in
  the implementation, not the theory. Bisect by instrumenting the identity directly (per-layer
  ratio and cosine), not by replacing the algorithm.
- In-place buffer reuse is fine — but only if you notice which variables alias the buffer. `d_y =
  grad_storage` after an in-place write quietly mutates `d_y`.
- A deep residual chain is an error-amplifier for off-by-one bugs in the reverse sweep. Any per-
  layer systematic error grows roughly as the operator norm of the composed Jacobian. If you see
  "last layer perfect, drift increasing with depth," the bug is almost certainly in how state is
  passed between layers, not in the single-layer math.
