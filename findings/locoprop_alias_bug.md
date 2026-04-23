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
"SGD outer should now suffice." Empirically it doesn't — SGD outer plateaus near identity
(tail ≈ 1.0) regardless of step size. The reason is *scale*, not direction. Per-layer
magnitudes of `loco_as_grad` vary by factors of 5–10× across depth (deeper layers accumulate the
GN chain; shallower layers start from a smaller target residual). AdamW's per-parameter
`√v` normalization fixes this automatically; a single global SGD learning rate cannot.

Loco's GN per layer + Adam's variance normalization across layers compose: one provides
*within-layer* curvature adaptation, the other *across-layer* scale balancing. Neither is
redundant with the other on this architecture.

### Empirical scaling

At 2048 epochs, batch 16384, 8 layers × 1024 dim, AdamW outer 3e-3, Nesterov inner β=0.9 lr=1.0:

| config | tail | eval | nMSE |
|---|---|---|---|
| BP+AdamW 3e-3 | 0.0594 | 0.0603 | 1.084 |
| BP+AdamW 1e-3 | 0.0673 | 0.0637 | 1.146 |
| n=64 Nesterov | 0.0339 | 0.0348 | 0.626 |
| n=128 Nesterov | 0.0224 | 0.0250 | 0.449 |
| n=256 Nesterov | 0.0193 | 0.0191 | 0.343 |

Doubling n roughly halves the residual gap to the architecture's capacity floor. Not yet
saturated at n=256. Training curves show the LocoProp runs reach BP+AdamW's final loss within
the first ~10% of training and then continue to improve for another order of magnitude while
BP plateaus.

The cost is O(n) per outer step — n=256 is ~256× the bwd-kernel work. Per-sample, not
per-wall-clock, the GN preconditioning is a real algorithmic gain, not a compute-shift.
Whether it's worthwhile in practice depends on whether one can tolerate the compute multiplier
to cut the final loss by 3×.

### What's still open

- Where does the n → ∞ floor actually lie? n=256 is still decreasing. A plateau below 0.015
  would suggest the GN step has residual bias (linearized targets become stale when the outer
  step is large).
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
