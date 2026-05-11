> **DEPRECATED**: This section describes the old training orchestration (monolithic 14-pass graphs). See [../03-training-orchestration.md](../03-training-orchestration.md) for the current design.

# Encoder Loss

The encoder plays the role of a **discriminator** in this adversarial framework. It is trained to:

1. **Maximize ELBO on real data** — build a structured latent space for normal samples
2. **Minimize ELBO on reconstructions and fakes** — learn to reject decoder outputs

$$\mathcal{L}_{\text{enc}} = \underbrace{-\text{ELBO}_{\text{real}}}_{\text{Fit normal manifold}} + \frac{1}{2} \left( \underbrace{\exp(\tau_e \cdot \text{ELBO}_{\text{rec}})}_{\text{Reject reconstructions}} + \underbrace{\exp(\tau_e \cdot \text{ELBO}_{\text{fake}})}_{\text{Reject fakes}} \right)$$

### Exponential Curriculum Weighting (Encoder)

The rejection terms use exponential curriculum weighting (`expelbo`):

$$\widetilde{\text{ELBO}} = \exp(\tau_e \cdot \text{ELBO})$$

Since ELBO values are negative (in the $[-2, 0]$ range with mean reductions), this creates adaptive curriculum learning:

- **Bad fakes** (ELBO $\approx -2$): $\exp(-2\tau_e) \approx 0.14$ at $\tau_e=1$ → low contribution. The encoder already rejects these easily.
- **Good fakes** (ELBO $\approx 0$): $\exp(0) \approx 1$ → high contribution. These are hard to discriminate and need the most training.

The `enc_expelbo_temp` ($\tau_e$) parameter controls curriculum sharpness:

- $\tau_e = 1$: Moderate differentiation ($\exp(-2) = 0.14$ vs $\exp(0) = 1$)
- $\tau_e = 2$: Sharp differentiation ($\exp(-4) = 0.02$ vs $\exp(0) = 1$)
- $\tau_e = 3$: Very sharp ($\exp(-6) = 0.002$ vs $\exp(0) = 1$)

Note: the ELBO multiplication previously in the formulation ($-\text{ELBO} \cdot \exp(\tau \cdot \text{ELBO})$) was removed. With ELBO properly in the $[-2, 0]$ range, the exponential alone provides sufficient differentiation. The old formulation had the encoder maximizing ALL ELBOs (including rec/fake) because the $-\text{ELBO}$ factor flipped the sign; now the encoder purely minimizes exp-weighted fake/rec quality — bad fakes contribute less, good fakes contribute more.

### Spatial Curriculum in the Encoder

The encoder's ELBO terms use the same `spatial_temperature` as the decoder. This has two effects that are both beneficial:

1. **On `ELBO_real`**: Amplifies gradient signal for structural regions in the latent space. The encoder allocates more representation capacity to edges and transitions, producing richer latent codes for structurally complex areas. At inference, this translates to **sharper anomaly maps** — the encoder's features are more sensitive to structural detail precisely where defects tend to appear (cracks at edges, inclusions near interfaces).

2. **On `ELBO_rec` / `ELBO_fake`**: Counterintuitively helps equilibrium. Amplified structural errors make reconstructions look "more obviously bad" to the encoder → ELBO values become more negative → `exp(τ_e · ELBO)` shrinks → the encoder's rejection signal **dampens** via its own curriculum. The spatial curriculum and expelbo interact to self-regulate.

**Crucially, `diff_kld` is unaffected** — KLD is computed from `(mean, logvar)` directly, never touching `pixel_mse`. The equilibrium callback sees the same diagnostic signal regardless of spatial temperature.

### Why NOT `expelbo_real` in the Encoder

Using per-sample curriculum weighting on the real term is counterproductive. The gradient magnitude from $-\exp(\tau \cdot \text{ELBO}_{\text{real}})$ scales as:

$$\text{weight} = \tau \cdot \exp(\tau \cdot \text{ELBO}_{\text{real}})$$

| ELBO_real | Quality | Weight (τ=1) | Effect                              |
| --------- | ------- | ------------ | ----------------------------------- |
| ≈ -2.0    | Bad     | 0.14         | **Encoder gives up on bad reals**   |
| ≈ -0.2    | Good    | 0.82         | Encoder polishes already-good reals |

This is exactly backwards. The encoder should push **hardest** when real ELBO is bad (structured latent space not yet formed), not give up. Curriculum weighting makes sense for rejection ("don't waste energy on obvious garbage") but not for fitting ("always build a good latent space for reals"). The linear $-\text{ELBO}_{\text{real}}$ provides constant gradient weight regardless of current quality — which is correct.

**Note**: This applies specifically to **per-sample** curriculum weighting (different samples in the batch getting different weights). The spatial curriculum ($\tau_s$) is a different mechanism — it redistributes gradient **within** each sample without changing the per-sample contribution. Every sample still gets the same total gradient magnitude thanks to per-sample normalization.

### No Embedding Loss in the Encoder

The embedding loss is deliberately **excluded** from the encoder's training objective.

**Considered alternative: Adversarial embedding loss** — maximize embedding distance in the encoder ("produce different features for originals vs reconstructions") while minimizing it in the decoder. This was rejected for structural reasons:

1. **When backbone is frozen**: Embeddings come from frozen backbone intermediate layers. The trainable encoder head is _downstream_ of embedding extraction points. Adding $-\mathcal{L}_{\text{embed}}$ to the encoder loss produces gradients that hit the frozen wall and vanish — it's literally a **no-op**.

2. **When backbone is thawed**: The gradient would push the backbone to learn **adversarial features** — hypersensitive to imperceptible differences between originals and reconstructions (compression artifacts, subtle blur, frequency shifts). This corrupts the backbone's general-purpose feature hierarchy, making it detect noise rather than meaningful anomalies.

3. **Redundancy**: The existing ELBO rejection already provides discrimination signal at the right level — the latent space. The encoder head learns to assign different $(\mu, \text{logvar})$ to originals vs reconstructions, which is what KLD measures. Backbone features remain intact for anomaly detection.

Including embedding loss (even as adversarial maximization) in the encoder is either:

- A no-op (frozen backbone)
- Feature corruption (thawed backbone)
- Redundant with ELBO rejection (always)
