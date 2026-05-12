# Adversarial Equilibrium & Skip Connection Rationale

### Why No Skip Connections

In a standard U-Net autoencoder, skip connections provide a direct path from encoder to decoder at each resolution level. For anomaly detection, this is catastrophic: even when the model is trained exclusively on non-anomalous data, anomalous features in the input propagate through skip connections directly to the output. The reconstruction error — which is the anomaly signal — gets suppressed because the anomaly bypasses the information bottleneck entirely.

By removing skip connections, all information must pass through the latent bottleneck. The decoder can only reconstruct what the bottleneck preserves, and since the bottleneck is trained on normal data, anomalous patterns cannot be faithfully reconstructed.

**Note**: Decoder-internal skip connections (residual paths within the decoder itself) are perfectly fine and beneficial — they don't leak anomalous information from the encoder. The prohibition is specifically on encoder→decoder skip connections.

### KLD Gap as Equilibrium Diagnostic

The difference between KL divergences on decoder outputs and real samples provides a diagnostic of the adversarial balance:

$$\Delta_{\text{KLD}} = \alpha \cdot D_{\text{KL}}^{\text{rec}} + (1 - \alpha) \cdot D_{\text{KL}}^{\text{fake}} - D_{\text{KL}}^{\text{real}}$$

where $\alpha$ = `diff_kld_rec_weight` (default: 0.7).

The weighting emphasizes the reconstruction path because `kld_rec` and `kld_real` share the same data lineage (both derive from the same real image), making their difference a semantically clean measure of round-trip information loss. The `kld_fake` term retains a minority weight (0.3) for early-training safety — detecting catastrophic encoder collapse on easy fakes — without dominating the diagnostic as fakes naturally diverge in late training.

| $\Delta_{\text{KLD}}$ | State                       | Interpretation                                                                                                     |
| --------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| $\gg 0$               | **Encoder dominant**        | Fakes require far more specialized encodings than reals. Decoder is underperforming.                               |
| $> 0$ (small)         | **Approaching equilibrium** | Decoder produces fakes nearly as "normal" as reals to the encoder.                                                 |
| $\approx 0$           | **Equilibrium**             | Encoder cannot distinguish fakes from reals via KLD alone. Ideal state.                                            |
| $< 0$                 | **Encoder collapsing**      | Encoder finds fakes more normal than reals. Pathological — the encoder has lost its reference frame for normality. |
