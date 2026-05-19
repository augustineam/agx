# Latent Interpolation Strategies (Fake Path)

## Context

In Step 2 (Decoder Fake Path), the decoder generates a "fake" image from a latent code `z_fake` that differs from the real encoding `z_real`. The quality of `z_fake` determines what the decoder learns:

- Too far from the manifold → decoder learns to generate noise → useless adversarial signal
- Too close to `z_real` → decoder learns near-identity → weak discrimination pressure
- On the manifold but different from `z_real` → decoder learns the normal manifold's structure

The `z_fake_interp` configuration controls how `z_fake` is sampled.

---

## Modes

### 1. Manifold Interpolation (Default: Recommended)

```python
z_fake_interp = dict(mode="manifold", manifold_op="roll")
```

**Mechanism:**

```
z_a = z_real[i]           # First endpoint (current sample)
z_b = z_real[(i+1) % B]   # Second endpoint (rolled neighbor)
t ~ Uniform(0, 1)          # Random interpolation weight
z_fake = (1 - t) * z_a + t * z_b   # Linear interpolation
```

**Why it works:** Both `z_a` and `z_b` are encodings of real images — they lie ON the learned manifold. Linear interpolation between two on-manifold points stays approximately on-manifold (for well-structured latent spaces), producing latent codes that decode to plausible-looking images that are nonetheless "fake" (not encoding any specific real input).

**Why linear (not spherical):** The latent space is regularized by KLD toward a Gaussian distribution $\mathcal{N}(0, I)$. Gaussian space has no intrinsic curvature — geodesics are straight lines. Slerp (spherical interpolation) is appropriate for hypersphere-distributed latents (e.g., normalized embeddings), not Gaussian-distributed ones.

**Pairing strategy — `roll` vs `shuffle`:**

| Operation        | Behavior                                                           | Trade-off                                                                                       |
| ---------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| `roll` (default) | Each sample pairs with its batch neighbor: `z[i]` ↔ `z[(i+1) % B]` | Deterministic pairing. Every sample gets a unique partner. Safe for small batches.              |
| `shuffle`        | Random permutation of the batch                                    | May accidentally pair a sample with itself in small batches. Better diversity in large batches. |

`roll` is preferred because:

- Guarantees no self-pairing (every sample pairs with a different sample)
- Deterministic given batch order — easier to debug
- In stratified batches (same product), roll pairs same-product samples → interpolation stays within product manifold

---

### 2. Perturbed (Gaussian Noise)

```python
z_fake_interp = dict(mode="perturbed", perturbed_sigma=0.2)
```

**Mechanism:**

```
z_fake = z_real + ε,   where ε ~ N(0, σ²I)
```

**When useful:** Never, in practice. The reparameterization trick already adds noise: `z = μ + σ · ε`. Adding more noise on top is redundant — it's equivalent to increasing the effective `σ` in the latent space without the principled KLD regularization.

**Why deprecated in favor of manifold:**

- Perturbed samples may leave the manifold (noise in arbitrary directions)
- No guarantee the perturbed point decodes to anything meaningful
- Redundant with the stochastic encoder
- Doesn't provide the "interpolation between two real structures" signal that manifold mode provides

---

### 3. Spherical Interpolation (Slerp)

```python
z_fake_interp = dict(mode="slerp", manifold_op="roll")
```

**Mechanism:**

```
z_a, z_b = z_real, rolled(z_real)
θ = arccos(z_a · z_b / (||z_a|| · ||z_b||))
z_fake = sin((1-t)θ) / sin(θ) · z_a + sin(tθ) / sin(θ) · z_b
```

**When appropriate:** Only if the latent space lies on a hypersphere (normalized embeddings, VQ-VAE codebook, etc.).

**Why NOT appropriate here:**

- Our latent space is regularized toward $\mathcal{N}(0, I)$ via KLD — it's a Gaussian cloud, not a hypersphere
- Slerp interpolation is numerically unstable near $\theta \approx 0$ (parallel vectors) and $\theta \approx \pi$ (antipodal)
- For Gaussian-distributed latents, slerp and lerp produce nearly identical results (the manifold is flat)
- Additional computational cost (arccos, sin) with no benefit

---

## Interaction with Multi-Product Training

### Stratified Batches (Same Product per Batch)

When using `StratifiedProductSampler`, all samples in a batch are from the same product. Manifold interpolation (roll) pairs same-product samples:

```
Batch: [bread_1, bread_2, bread_3, ..., bread_B]
Roll:  [bread_2, bread_3, bread_4, ..., bread_1]
Interp: lerp(bread_i, bread_(i+1))  → stays on bread manifold
```

This is ideal for early training — the decoder learns per-product manifold structure without cross-product confusion.

### Mixed Batches (Multiple Products per Batch)

With standard `DataLoader(shuffle=True)`, batches contain mixed products:

```
Batch: [bread_1, tuna_3, beer_2, glass_5, ...]
Roll:  [tuna_3, beer_2, glass_5, ..., bread_1]
Interp: lerp(bread_1, tuna_3)  → cross-product "chimera"
```

Cross-product interpolation creates latent codes that decode to **hybrid images** — part bread, part tuna. These are structurally anomalous and serve as **hard negative augmentation** for the encoder critic (step 4). The encoder must learn to reject these chimeras, sharpening its per-product discrimination.

### Recommended Curriculum

| Training Phase | Batch Strategy | Interpolation Effect                                                                 |
| -------------- | -------------- | ------------------------------------------------------------------------------------ |
| Early (0–30%)  | Stratified     | Same-product interpolation. Decoder learns clean product manifolds.                  |
| Mid (30–70%)   | Mixed          | Cross-product chimeras. Encoder hardens discrimination boundaries.                   |
| Late (70–100%) | Mixed          | Continued hardening. Cross-product KLD becomes high, providing strong critic signal. |

### Condition Alignment

The condition vector `c` used in the fake path is always aligned with the **first endpoint** `z_a` (the original sample). When interpolating across products, the decoded fake has mixed-product structure but the condition says "this should be product A" — creating exactly the type of mismatch the encoder should learn to detect.

---

## Implementation Details

```python
def sample_z_fake(self, z: keras.KerasTensor):
    mode = self.z_fake_interp.get("mode", "perturbed")
    if mode == "manifold":
        z_fake = self._sample_z_manifold(z)
    elif mode == "perturbed":
        z_fake = self._sample_z_perturbed(z)
    elif mode == "slerp":
        z_fake = self._sample_z_slerp(z)
    return z_fake

def _sample_manifold(self, z: keras.KerasTensor):
    """Get the second interpolation endpoint."""
    manifold_op = self.z_fake_interp.get("manifold_op", "roll")
    if manifold_op == "roll":
        z_b = ops.roll(z, shift=1, axis=0)
    else:
        z_b = keras.random.shuffle(z, axis=0)
    return ops.stop_gradient(z_b)

def _sample_z_manifold(self, z: keras.KerasTensor):
    """Linear interpolation between z and a rolled/shuffled partner."""
    z_b = self._sample_manifold(z)
    batch_size = ops.shape(z)[0]
    t = keras.random.uniform((batch_size, 1, 1, 1))
    z_interp = (1 - t) * z + t * z_b
    return ops.stop_gradient(z_interp)
```

**Critical:** `z_fake` is always `stop_gradient`'d. The decoder's adversarial signal flows through the encoder→decoder cycle, NOT through the input `z_fake` itself. The fake starting point is treated as a fixed "seed" for generation.

---

## Hyperparameters

| Parameter         | Default      | Notes                                                                    |
| ----------------- | ------------ | ------------------------------------------------------------------------ |
| `mode`            | `"manifold"` | Recommended. Alternatives: `"perturbed"`, `"slerp"`                      |
| `manifold_op`     | `"roll"`     | Safe for all batch sizes. Alternative: `"shuffle"`                       |
| `perturbed_sigma` | `0.2`        | Only relevant if `mode="perturbed"`. Noise scale relative to latent std. |

---

## Design Decisions

| Considered                      | Verdict        | Reason                                                                                          |
| ------------------------------- | -------------- | ----------------------------------------------------------------------------------------------- |
| Pure noise `z ~ N(0, I)`        | ❌             | Too far from manifold early in training. Decoder wastes capacity on out-of-distribution inputs. |
| Perturbed (add noise to z_real) | ❌ Deprecated  | Redundant with reparameterization trick. No manifold guarantee.                                 |
| Slerp                           | ❌             | Latent space is Gaussian, not hypersphere. Unnecessary complexity.                              |
| Manifold + roll                 | ✅ Default     | On-manifold, diverse, deterministic pairing, safe for small batches.                            |
| Manifold + shuffle              | ⚠️ Optional    | Better for large batches; risk of self-pairing in small batches.                                |
| Curriculum (perturbed→manifold) | ❌ Unnecessary | Manifold interpolation works from epoch 0 (both endpoints are real encodings).                  |
