# Gradient Clipping & “Norms” (Mini‑Course)

This document explains:

- What a **clamp** is
- What **gradient clipping** is (and why it exists)
- The difference between **value clipping** and **norm clipping**
- The difference between **global** vs **per‑tensor/per‑layer** norm clipping
- How this differs from **normalization layers** (LayerNorm/RMSNorm) in the model architecture
- Why these details matter for **parity testing** between PyTorch and PCP

---

## 1) The basic training pipeline (where gradients live)

A standard training step has three phases:

1. **Forward pass**
   - The model runs on inputs and produces a scalar **loss**:
     - `loss = L(theta; x, y)`

2. **Backward pass**
   - Automatic differentiation computes **gradients** of the loss with respect to parameters:
     - `g = d(loss)/d(theta)`
   - In PyTorch these are `param.grad` tensors.

3. **Optimizer update**
   - The optimizer uses gradients to update parameters:
     - `theta = Update(theta, g)`

**Gradient clipping happens between (2) and (3).**  
It modifies gradients before the optimizer uses them.

This is different from **normalization layers** (LayerNorm/RMSNorm), which live in the **forward pass** and act on activations.

---

## 2) What is a “clamp”?

**Clamp** means “cap values to a range”.

For a scalar value `v` clamped to `[a, b]`:

```
clamp(v, a, b) = min(max(v, a), b)
```

For tensors, clamp is **elementwise**: each element is clamped independently.

In PyTorch:

```python
torch.clamp(tensor, min=a, max=b)
```

---

## 3) What is “gradient clipping”?

Sometimes gradients become very large (“exploding gradients”). This can cause:

- Huge parameter updates (training diverges)
- `NaN`/`Inf` in activations or parameters
- Instability (loss spikes, training collapses)

**Gradient clipping** prevents catastrophic updates by limiting gradient magnitude.

There are two common families:

---

## 4) Gradient clipping by value (elementwise clipping)

**Value clipping** clamps each element of the gradient tensor:

```
g[i] = clamp(g[i], -c, c)
```

In PyTorch:

```python
loss.backward()
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=c)
optimizer.step()
```

Pros:
- Simple.
- Prevents single entries from exploding.

Cons:
- Can change the gradient *direction* a lot if many values saturate to ±c.

---

## 5) Gradient clipping by norm (scale the whole gradient)

**Norm clipping** rescales a gradient tensor if its norm is too large.

For a gradient tensor `g` and max norm `c`:

```
scale = min(1, c / ||g||)
g = g * scale
```

In PyTorch (global norm; see below):

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=c)
optimizer.step()
```

Pros:
- Preserves direction: you only scale magnitude.
- Very common and usually “gentler” than value clipping.

Cons:
- Needs a norm computation (extra ops).
- You must decide what “||g||” means over which set of parameters (global vs per‑tensor).

---

## 6) Global vs per‑tensor vs per‑layer norm clipping

When you clip “by norm”, you still need to define **what group of gradients share one norm**.

### A) Global norm clipping (most common)

Compute a single norm across **all gradients of all parameters** (conceptually concatenate them).

This is what `torch.nn.utils.clip_grad_norm_(model.parameters(), ...)` does by default.

Effect:
- One giant safety belt.
- If one layer spikes, it can scale down all gradients.

### B) Per‑tensor norm clipping

Compute a norm for each parameter tensor separately, and clip each tensor independently:

```
for each param tensor p:
  g_p = grad(p)
  g_p = g_p * min(1, c / ||g_p||)
```

Effect:
- One spiky tensor does not shrink all other tensors.
- Different from PyTorch’s usual global norm clipping.

### C) Per‑layer / per‑group norm clipping

Group parameters by “layer” or blocks and clip norm per group.

Effect:
- Middle ground between global and per‑tensor.
- Useful when you want consistent scaling per module.

---

## 7) “Norm” in the model architecture (LayerNorm/RMSNorm) is different

This is a common confusion.

- **LayerNorm / RMSNorm** are *forward-pass layers* that normalize activations.
  - They live in the model definition.
  - They change the forward computation and therefore the loss.

- **Gradient norm clipping** is a *training-time procedure*.
  - It does not change the model architecture.
  - It modifies gradients after backprop but before parameter updates.

They target different problems:

- Norm layers stabilize **activations / signal flow**.
- Gradient clipping stabilizes **updates** when gradients spike.

You often use both in real training.

---

## 8) How PCP’s training recipe applies clipping

PCP builds one compiled “training step” graph. That graph includes:

- A forward loss computation
- A gradient computation (autodiff)
- An optimizer update (AdamW)

Because it is “all-in-one”, gradient clipping can be implemented inside the compiled graph.

In this repo, PCP currently uses **two** gradient-limiting mechanisms:

### A) Elementwise gradient clamp (value clipping)

During autodiff graph construction, PCP clamps gradient values to `[clip_min, clip_max]`.

Conceptually:

```python
grad = torch.clamp(grad, min=clip_min, max=clip_max)
```

### B) Per‑parameter‑tensor L2 norm clipping

Inside the AdamW update, PCP computes each gradient tensor’s norm and scales it down if above `max_grad_norm`.

Conceptually:

```python
norm = grad.pow(2).sum().sqrt() + eps
scale = max_norm / max(norm, max_norm)
grad = grad * scale
```

This is **per tensor**, not the usual PyTorch global norm over all parameters.

---

## 9) Why these details matter for “PyTorch vs PCP parity”

If you compare:

- PCP inner loop: `value clip + per‑tensor norm clip + AdamW`
vs
- PyTorch default: `AdamW` only (no clipping)

…then you are **not running the same training recipe**, so you should expect parameter updates to differ even if PCP is correct.

To do an apples-to-apples comparison, you must either:

1. Apply the same clipping policy in PyTorch (same order + same per‑tensor vs global), or
2. Disable clipping in PCP (make it configurable and set it off), then compare against PyTorch default AdamW.

---

## 10) PyTorch code snippets (common patterns)

### A) AdamW with no clipping

```python
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

### B) AdamW with value clipping + global norm clipping

```python
loss.backward()
torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # global norm
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

### C) AdamW with per‑tensor norm clipping (not the default)

```python
loss.backward()
max_norm = 1.0
for p in model.parameters():
    if p.grad is None:
        continue
    g = p.grad
    norm = g.norm(2)
    if norm > max_norm:
        g.mul_(max_norm / (norm + 1e-8))
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

---

## 11) Mental model summary

- **Clamp**: cap each number to a range.
- **Gradient clipping by value**: clamp each gradient element.
- **Gradient clipping by norm**: scale a whole gradient tensor/group so its norm is under a threshold.
- **Global norm**: one norm across all params; **per‑tensor**: one norm per parameter tensor; **per‑layer**: one norm per group.
- **Norm layers (RMSNorm/LayerNorm)** are forward‑pass architecture; **gradient norm clipping** is training‑time.

If you want strict parity between two training systems, you must match:

- model forward math
- gradient math (autodiff)
- optimizer math (AdamW details)
- clipping policy (value vs norm, global vs per‑tensor, order)
- any other “training recipe” rules (weight decay form, eps placement, etc.)
