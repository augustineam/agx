# Encoder Critic Loss (Step 4)

## Objective

The encoder is trained as a **discriminator** that identifies decoder outputs via KL divergence. Real images should encode with low KLD (close to the standard normal prior); decoder outputs should encode with high KLD (far from prior).

## Loss Formulation

$$\mathcal{L}_{\text{critic}} = \exp(-\tau_e \cdot \text{KLD}_{\text{fake}}) + \exp(-\tau_e \cdot \text{KLD}_{\text{rec}})$$

where:
- $\text{KLD}_{\text{fake}} = \text{mean}_{h,w}[D_{\text{KL}}(q_\phi(z|\text{fake}) \| p(z))]$ — per-position KLD divided by latent size (channels), then averaged over spatial dimensions
- $\text{KLD}_{\text{rec}} = \text{mean}_{h,w}[D_{\text{KL}}(q_\phi(z|\text{rec\_real}) \| p(z))]$

The encoder **minimizes** this loss, which means **maximizing** KLD on decoder outputs.

## Curriculum Weighting: `exp(-τ · KLD)`

Since KLD ≥ 0:

| KLD value | Meaning | exp(-τ·KLD) at τ=1 | Gradient contribution |
|---|---|---|---|
| ≈ 0 | Hard to discriminate | exp(0) = 1.0 | Full |
| ≈ 1 | Moderate | exp(-1) = 0.37 | Moderate |
| ≈ 3 | Easy to discriminate | exp(-3) = 0.05 | Nearly ignored |

The encoder concentrates effort on decoder outputs that are hard to tell apart from reals, rather than wasting capacity on obvious fakes.

**Gradient analysis:**

$$\frac{\partial}{\partial \theta}\exp(-\tau \cdot \text{KLD}) = -\tau \cdot \exp(-\tau \cdot \text{KLD}) \cdot \frac{\partial \text{KLD}}{\partial \theta}$$

Minimizing the loss pushes $\text{KLD}$ higher (∂KLD/∂θ > 0 when minimizing exp(-τ·KLD)). The gradient magnitude is proportional to exp(-τ·KLD) — larger for hard cases. ✓

## Why Pure KLD (Not Full ELBO)

In the old design, the encoder's loss included ELBO terms with logP(x|z) components. These were **phantom gradients** — the reconstruction terms involved decoder outputs under `stop_gradient` or `no_grad`, meaning the MSE components contributed zero actual gradient to the encoder. Only KLD terms were ever differentiable.

The new design is honest: the encoder's discrimination mechanism IS KLD. No phantom terms.

## Why No Embedding Loss in the Encoder

See [deprecated/03-encoder-loss-old.md](./deprecated/03-encoder-loss-old.md) for full analysis. Summary:

- **Frozen backbone**: Embeddings come from frozen layers downstream of trainable head → no-op
- **Thawed backbone**: Would corrupt features toward adversarial noise sensitivity
- **Redundant**: KLD already discriminates at the latent level

## Interaction with Step 1

The encoder receives **two updates per training step** with different characters:

1. **Step 1 (collaborative):** "Map reals to structured, low-KLD codes that enable reconstruction"
2. **Step 4 (adversarial):** "Push KLD high on decoder outputs"

These are complementary, not contradictory:
- Step 1 establishes what "normal encoding" looks like (low KLD, good reconstruction)
- Step 4 sharpens the boundary — decoder outputs that slip close to "normal" are actively pushed away

Per-step gradient application ensures each signal has its own Adam momentum trajectory.

## Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `enc_expkld_temp` (τ_e) | 1.0 | Curriculum sharpness. Higher = ignore easy discriminations more |

- τ_e = 0.5: Mild curriculum (exp(-1.5) = 0.22 for KLD=3)
- τ_e = 1.0: Moderate (exp(-3) = 0.05 for KLD=3)
- τ_e = 2.0: Sharp (exp(-6) = 0.002 for KLD=3)

Higher values give the decoder more "head start" early in training by dampening easy rejections, allowing the decoder to improve before the encoder tightens discrimination.
