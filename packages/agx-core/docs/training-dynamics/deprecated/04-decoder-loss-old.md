> **DEPRECATED**: This section describes the old training orchestration (monolithic 14-pass graphs). See [../03-training-orchestration.md](../03-training-orchestration.md) for the current design.

# Decoder Loss

The decoder plays the role of a **generator**. It is trained to produce outputs that the encoder considers normal, whether starting from real latent codes or pure noise.

$$\mathcal{L}_{\text{dec}} = \underbrace{-\text{ELBO}_{\text{real}}}_{\text{Reconstruct faithfully}} + \frac{1}{2} \left( \underbrace{\exp(-\tau_d \cdot \text{ELBO}_{\text{rec}})}_{\text{Fool (cycle)}} + \underbrace{\exp(-\tau_d \cdot \text{ELBO}_{\text{fake}})}_{\text{Fool (generation)}} \right) + \underbrace{\lambda_{\text{embed}} \cdot \mathcal{L}_{\text{embed}}}_{\text{Feature consistency}}$$

| Term                                            | Signal to decoder                                                          |
| ----------------------------------------------- | -------------------------------------------------------------------------- |
| $-\text{ELBO}_{\text{real}}$                    | Produce reconstructions $I'$ that match the input at pixel level           |
| $\exp(-\tau_d \cdot \text{ELBO}_{\text{rec}})$  | Re-reconstructions $I''$ should fool the encoder (focus on hard)           |
| $\exp(-\tau_d \cdot \text{ELBO}_{\text{fake}})$ | Random generations, when round-tripped, should look normal (focus on hard) |
| $\mathcal{L}_{\text{embed}}$                    | Reconstructions must be faithful in the encoder's feature space            |

### Exponential Curriculum Weighting (Decoder)

The decoder's adversarial terms use **inverted** curriculum weighting compared to the encoder:

$$\exp(-\tau_d \cdot \text{ELBO})$$

The direction is reversed because the decoder's "hard" cases are the opposite of the encoder's:

| Component   | "Hard" means                           | Curriculum direction                          |
| ----------- | -------------------------------------- | --------------------------------------------- |
| **Encoder** | High ELBO (good fakes, hard to reject) | $\exp(+\tau_e \cdot \text{ELBO})$ — up-weight |
| **Decoder** | Low ELBO (encoder rejects easily)      | $\exp(-\tau_d \cdot \text{ELBO})$ — up-weight |

With ELBO $\in [-2, 0]$ and $\tau_d = 1$:

- **Decoder easily fools encoder** (ELBO $\approx 0$): $\exp(0) = 1$ → normal gradient
- **Decoder failing badly** (ELBO $\approx -2$): $\exp(2) \approx 7.4$ → 7× more gradient

The gradient analysis confirms correct behavior:

$$\frac{\partial}{\partial \theta} \exp(-\tau_d \cdot \text{ELBO}) = -\tau_d \cdot \exp(-\tau_d \cdot \text{ELBO}) \cdot \frac{\partial \text{ELBO}}{\partial \theta}$$

Since the decoder wants to increase ELBO ($\partial\text{ELBO}/\partial\theta > 0$), the gradient points toward minimizing this loss (= increasing ELBO), with magnitude proportional to $\exp(-\tau_d \cdot \text{ELBO})$ — larger for hard cases. ✓

The `dec_expelbo_temp` ($\tau_d$) parameter controls how aggressively the decoder focuses on its failure modes:

- $\tau_d = 1$: Moderate focus (7× ratio between hardest and easiest)
- $\tau_d = 2$: Sharp focus (55× ratio)
- $\tau_d = 0$: No curriculum; falls back to linear $-\text{ELBO}$ (original behavior)

### Why `ELBO_real` Is Essential in the Decoder

Since `kld_real` is stop-gradiented, the only gradient the decoder receives from $-\text{ELBO}_{\text{real}}$ is:

$$\nabla_\theta (-(-\text{logpx\_z\_real})) = -\nabla_\theta \text{MSE}(\text{real}, \text{rec\_real})$$

**This IS the reconstruction loss.** It's the direct pixel-level signal saying "make `rec_real` look like `real`." Without it, the decoder only receives:

- `elbo_rec/fake`: "Fool the encoder" — but no constraint to match the _specific input_
- `embed_loss`: Feature consistency — but only at intermediate representations

Without `elbo_real`, the decoder could collapse to producing the same "average normal image" for every input (mode collapse), or any image the encoder scores highly regardless of input correspondence. `elbo_real` **grounds** the decoder to the input.

The three decoder signals are complementary and all necessary:

- `elbo_real` (pixel MSE): "Make every pixel right" → prevents hallucination
- `embed_loss` (features): "Make it right to the encoder" → prevents blur
- `expelbo_rec + expelbo_fake`: "Fool the encoder" → pushes toward normal manifold,
  with adaptive focus on the samples where the decoder is struggling most

Note that **all** KLD terms (`kld_real`, `kld_rec`, `kld_fake`) in the decoder are **stop-gradiented**. The decoder receives pure reconstruction gradient (`logpx_z`) without any KL signals. The rationale: KLD measures "how normal does the encoder consider this encoding?" — that is the encoder's concern, not the decoder's. If KLD gradients flowed back to the decoder, it could learn to produce outputs that happen to encode with low KLD (close to the standard normal prior) without actually being faithful reconstructions — gaming the encoder's latent parameterization rather than improving pixel fidelity. The KLD terms are present in the ELBO formulation solely for bookkeeping (consistent loss values, meaningful `diff_kld` diagnostics) but contribute zero gradient to the decoder.

### Why NOT `expelbo_real` in the Decoder

The same principle that excludes per-sample curriculum from the encoder's `elbo_real` applies to the decoder: **fitting terms need constant uniform pressure**.

If we applied `exp(-τ_d · ELBO_real)` to the decoder:

| ELBO_real | Quality             | Weight (τ=1)   | Effect                  |
| --------- | ------------------- | -------------- | ----------------------- |
| ≈ -2.0    | Bad reconstruction  | exp(2) ≈ 7.4   | Focus here              |
| ≈ -0.1    | Good reconstruction | exp(0.1) ≈ 1.1 | **Neglect this sample** |

The "neglect" is dangerous for anomaly detection. Those easy-to-reconstruct samples are easy **now** but need maintained effort to **stay** easy. Without gradient pressure, their reconstruction quality can drift. At inference, if a previously-easy normal sample has degraded reconstruction, it produces a **false positive anomaly**.

The decoder must maintain **uniformly good reconstruction across all normal samples** because at inference, any normal sample could be the test case. Curriculum weighting makes sense for adversarial terms ("focus on what you're failing at") but not for fitting terms ("always reconstruct the entire normal manifold uniformly").

**Note**: The spatial curriculum ($\tau_s$) is different — it redistributes gradient **within** each sample without reducing any sample's total contribution. Every sample maintains the same total gradient magnitude thanks to per-sample normalization. Spatial curriculum = "focus on hard REGIONS within every image"; per-sample curriculum = "focus on hard IMAGES in the batch." Only the former is safe for fitting terms.

### Spatial Curriculum in the Decoder

The decoder uses the same `spatial_temperature` as the encoder for its pixel MSE terms. This directly addresses the X-ray spatial dilution problem:

- **Uniform background** (error ≈ 0.001, $\tau_s$=5): weight ≈ 1.005
- **Structural detail** (error ≈ 0.1, $\tau_s$=5): weight ≈ 1.65

After normalization, structural regions get ~1.6× more gradient budget. The decoder is told: "focus capacity on hard-to-reconstruct regions (edges, transitions, fine detail) rather than trivially uniform background."

This creates a positive feedback loop:

1. Spatial curriculum → decoder improves on structural regions
2. Better structural reconstruction → embedding loss decreases
3. Lower embedding loss → less adversarial imbalance
4. Better equilibrium → fewer callback pauses → more consistent training

### Embedding Loss as Perceptual Critic (Decoder Only)

The encoder acts as a **frozen perceptual critic** during decoder training. Its intermediate features define the space where reconstruction fidelity is measured. By restricting embedding loss to the decoder:

- The **encoder** remains free to produce divergent embeddings for anomalous inputs
  — preserving anomaly sensitivity.
- The **decoder** is pushed to always reconstruct "healthy" images that re-encode
  faithfully — learning the normal manifold.

At inference, the anomaly score:

$$\text{score}(x) = d\big(E(x),\ E(D(z))\big)$$

is high precisely because the encoder was **never trained to suppress** the difference between original and reconstructed features.

#### Gradient Asymmetry: Frozen Target vs Trainable Path

The embedding loss compares two sets of encoder features:

- **`embeds_real` = E(I)**: The encoder processes the real image. Since the encoder is frozen during the decoder's turn and `I` is not a decoder output, there are no decoder parameters in this computation graph. These embeddings are explicitly stop-gradiented as a defensive measure — they serve as the **frozen target**.

- **`embeds_rec` = E(I')** where `I' = D(z)`: The encoder processes the decoder's reconstruction. Even though the encoder is frozen (`training=False`), `I'` is a decoder output — gradients flow back through the encoder's forward pass into `D`. These embeddings are the **trainable path**.

This asymmetry is the mechanism: the decoder receives gradient saying "change your output so that when the encoder re-encodes it, the resulting features match the frozen target from the original image." The encoder's features define _what_ to match; the decoder learns _how_ to match it.

#### Why NOT Curriculum on Embedding Loss

The embedding loss already operates in feature space, which provides natural importance weighting — deeper features capture structural information rather than background. Additionally, the spatial curriculum on pixel MSE indirectly helps embedding loss: as the decoder improves reconstruction of structural regions, the feature representations naturally converge.

If `loss_embed` plateaus while `loss_rec` continues dropping, this indicates the embedding features are capturing something the pixel loss doesn't directly address (e.g., phase/frequency information). In that case, depth-weighted embedding loss (§2) is the appropriate intervention, not per-sample curriculum weighting.
