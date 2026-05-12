# Decoder Adversarial Loss (Steps 2 & 3)

## Objective

The decoder is trained to produce outputs that the encoder considers normal (low KLD when re-encoded) and that survive round-tripping (cycle consistency). These steps provide the **adversarial generation signal** that was missing in the old design.

## Key Innovation: Frozen-but-Differentiable Encoder

In both steps 2 and 3, the encoder is set to `trainable=False` but the computation graph is stored through it. This means:

- **Encoder parameters don't update** (no gradient applied to E)
- **Decoder parameters DO receive gradient through E's forward pass** (E acts as a differentiable function, not a wall)

This gives the decoder two gradient paths it never had before:

1. **KLD → E → D₁**: "Produce outputs that encode with low KLD"
2. **Reconstruction loss → D₂ → z → reparam → E → D₁**: "Produce outputs that survive encode→decode round-trip"

## Step 2: Fake Path

### Graph

```
noise → [D₁] → fake → [E_frozen] → (μ, σ) → reparam → z_fake → [D₂] → rec_fake
```

### Loss

$$\mathcal{L}_{\text{fake}} = \exp(-\tau_d \cdot \text{ELBO}_{\text{fake}})$$

$$\text{ELBO}_{\text{fake}} = \underbrace{-\text{mean}_{h,w}[\mathcal{L}_{\text{rec}}(\text{sg}(\text{fake}), \text{rec\_fake})]}_{\text{logP(rec\_fake|z\_fake)}} - \beta \cdot \underbrace{\text{mean}_{h,w}[\text{KLD}(\mu, \sigma)]}_{\text{regularization}}$$

where $\mathcal{L}_{\text{rec}}$ is the blended reconstruction loss (MSE + SSIM, see [02-loss-foundations.md](./02-loss-foundations.md#reconstruction-loss)).

### Gradient Decomposition for D₁

D₁ receives gradient from two sources within the same backward pass:

**Path A — KLD through E:**
$$\nabla_{\theta_{D_1}} \text{KLD}(\mu, \sigma) = \frac{\partial \text{KLD}}{\partial (\mu, \sigma)} \cdot \frac{\partial (\mu, \sigma)}{\partial \text{fake}} \cdot \frac{\partial \text{fake}}{\partial \theta_{D_1}}$$

Signal: "Change your output so that when the encoder processes it, the resulting (μ, σ) are closer to N(0, I)."

**Path B — Reconstruction loss through D₂, z, reparam, E:**
$$\nabla_{\theta_{D_1}} \mathcal{L}_{\text{rec}}(\text{sg(fake)}, \text{rec\_fake}) = \frac{\partial \mathcal{L}_{\text{rec}}}{\partial \text{rec\_fake}} \cdot \frac{\partial \text{rec\_fake}}{\partial z} \cdot \frac{\partial z}{\partial (\mu, \sigma)} \cdot \frac{\partial (\mu, \sigma)}{\partial \text{fake}} \cdot \frac{\partial \text{fake}}{\partial \theta_{D_1}}$$

Signal: "Change your output so that when encoded to z and re-decoded by D₂, the result matches your original output." This is the **cycle consistency** signal — D₁ learns to produce images on the manifold where encode→decode is approximately identity. The SSIM component provides gradient specifically for structural preservation through the round-trip.

### Why `fake` Is Stop-Gradiented as Reconstruction Target

Without stop_gradient: $\mathcal{L}_{\text{rec}}(\text{fake}, \text{rec\_fake})$ has gradient flowing to BOTH sides (D₁ controls fake, D₁+D₂ control rec_fake). D₁ could minimize the loss by making fake converge to whatever D₂ outputs — a degenerate fixed point unrelated to the data distribution.

With stop_gradient: D₁ can only reduce the reconstruction loss by improving the prediction (rec_fake) through the z path. The target (fake) is treated as a fixed reference within each step.

### Why `z_fake` Is NOT Stop-Gradiented

If z_fake were detached, D₂ would get reconstruction gradient but D₁ would lose path B entirely. D₁ would only receive the KLD signal — "produce images that encode normally." But low KLD ≠ reconstructable. D₁ could produce high-frequency noise that projects to low-KLD regions without being on D's generation manifold.

The live z provides the explicit constraint: "not only must your output encode normally, it must also be RECONSTRUCTABLE from that encoding." This directly optimizes the anomaly detection inference metric.

Memory cost of keeping z live: **zero additional memory** — activations are stored during forward regardless of stop_gradient. The only difference is compute during backward (gradient traverses E→D₁ from reconstruction loss path in addition to KLD path).

---

## Step 3: Reconstruction Path

### Graph

```
z_real(const) → [D₁] → rec_real → [E_frozen] → (μ, σ, embeds_rec) → reparam → z_rec → [D₂] → rec_rec
```

### Loss

$$\mathcal{L}_{\text{step3}} = \exp(-\tau_d \cdot \text{ELBO}_{\text{rec}}) + \lambda_{\text{embed}} \cdot \mathcal{L}_{\text{embed}}$$

$$\text{ELBO}_{\text{rec}} = -\text{mean}_{h,w}[\mathcal{L}_{\text{rec}}(\text{sg}(\text{rec\_real}), \text{rec\_rec})] - \beta \cdot \text{KLD}(\mu_{\text{rec}}, \sigma_{\text{rec}})$$

where $\mathcal{L}_{\text{rec}}$ is the blended reconstruction loss (MSE + SSIM).

$$\mathcal{L}_{\text{embed}} = \text{embedding\_loss}(\text{sg}(\text{embeds\_real}), \text{embeds\_rec})$$

### Three Gradient Signals to D₁

1. **Embedding loss through E**: "Produce reconstructions whose encoder features match the original's"
   - Path: embed_loss → embeds_rec → E_forward → rec_real → D₁

2. **KLD through E**: "Produce reconstructions that encode close to the prior"
   - Path: KLD → (μ,σ) → E_forward → rec_real → D₁

3. **Reconstruction loss through D₂, z_rec, E**: "Produce reconstructions that survive re-encoding + re-decoding"
   - Path: $\mathcal{L}_{\text{rec}}$ → rec_rec → D₂ → z_rec → reparam → (μ,σ) → E_forward → rec_real → D₁

### Embedding Loss: Perceptual Critic

The embedding loss compares encoder features from two paths:

- **`embeds_real`** (from step 1, stop-gradiented): What the encoder sees in the original image
- **`embeds_rec`** (live, through frozen E): What the encoder sees in D₁'s reconstruction

This asymmetry means D₁ receives: "Change your output so that when re-encoded, the encoder produces the same intermediate features it produced for the original image."

The encoder acts as a **frozen perceptual critic** — it defines the feature space where reconstruction fidelity is measured, without being modified by that measurement.

### The "Bad Image Early" Concern

At training start, rec_real from D₁ is poor quality. Does the cycle consistency signal (MSE on rec_rec) teach D₂ to reconstruct garbage?

**Self-correcting:** Step 1's cooperative training continuously improves D₁'s reconstruction quality (via direct reconstruction loss on real images). Steps 2/3 operate on whatever D currently produces. Early noisy cycle-consistency is suboptimal signal but not harmful — it provides the correct gradient direction ("survive round-trip better") just with noisier magnitude. As step 1 converges, step 3 becomes increasingly precise.

---

## Exponential Curriculum Weighting

Both steps use $\exp(-\tau_d \cdot \text{ELBO})$ where $\text{ELBO} \in [-\infty, 0]$:

| ELBO   | Decoder quality | exp(-τ_d·ELBO) at τ=1 | Effect                    |
| ------ | --------------- | --------------------- | ------------------------- |
| ≈ -2.0 | Failing badly   | exp(2) ≈ 7.4          | 7× gradient amplification |
| ≈ -1.0 | Moderate        | exp(1) ≈ 2.7          | 3× amplification          |
| ≈ -0.2 | Nearly fooling  | exp(0.2) ≈ 1.2        | Normal gradient           |
| ≈ 0.0  | Fooling encoder | exp(0) = 1.0          | Baseline                  |

The decoder focuses training capacity on its failure modes — samples where the encoder easily discriminates its output.

### β in the ELBO

`beta_kld` appears in all ELBO computations (steps 1, 2, 3) with the same value. This is correct because β defines "what normal means geometrically" — the reconstruction-vs-regularization tradeoff that shapes the latent space. Using inconsistent β across steps would create conflicting definitions of normality.

---

## Hyperparameters

| Parameter                | Default | Effect                                                     |
| ------------------------ | ------- | ---------------------------------------------------------- |
| `dec_expelbo_temp` (τ_d) | 1.0     | Curriculum sharpness. Higher = more focus on failure modes |
| `lambda_embed`           | 1.0     | Weight of embedding loss in step 3                         |
| `beta_kld`               | 0.25    | KLD weight in all ELBO terms                               |
| `spatial_temperature`    | 1.0     | Spatial curriculum in pixel MSE component (all steps)      |
| `alpha_ssim`             | 0.3     | SSIM blend weight in reconstruction loss (all steps)       |

- τ_d = 0: No curriculum; exp(0) = 1 everywhere → linear -ELBO
- τ_d = 1: Moderate (7× ratio between hardest and easiest)
- τ_d = 2: Sharp (55× ratio) — use with caution, may cause instability

If `expelbo_fake` or `expelbo_rec` metrics consistently exceed 20, reduce τ_d or add gradient clipping.
