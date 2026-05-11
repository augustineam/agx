# Adversarial Equilibrium & Skip Connection Rationale

### Why No Skip Connections

In a standard U-Net autoencoder, skip connections provide a direct path from encoder to decoder at each resolution level. For anomaly detection, this is catastrophic: even when the model is trained exclusively on non-anomalous data, anomalous features in the input propagate through skip connections directly to the output. The reconstruction error — which is the anomaly signal — gets suppressed because the anomaly bypasses the information bottleneck entirely.

By removing skip connections, all information must pass through the latent bottleneck. The decoder can only reconstruct what the bottleneck preserves, and since the bottleneck is trained on normal data, anomalous patterns cannot be faithfully reconstructed.

**Note**: Decoder-internal skip connections (residual paths within the decoder itself) are perfectly fine and beneficial — they don't leak anomalous information from the encoder. The prohibition is specifically on encoder→decoder skip connections.

### KLD Gap as Equilibrium Diagnostic

The difference between KL divergences on decoder outputs and real samples provides a diagnostic of the adversarial balance:

$$\Delta_{\text{KLD}} = \frac{1}{2}\left(D_{\text{KL}}^{\text{fake}} + D_{\text{KL}}^{\text{rec}}\right) - D_{\text{KL}}^{\text{real}}$$

Both decoder output paths (fake and rec) are averaged to prevent the encoder from learning an asymmetric shortcut — e.g., becoming dominant on the rec path (easily discriminating re-reconstructions that accumulate two passes of decoder artifacts) while appearing balanced on the fake path. Averaging ensures the equilibrium callback detects dominance on _either_ path.

| $\Delta_{\text{KLD}}$ | State                       | Interpretation                                                                                                     |
| --------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| $\gg 0$               | **Encoder dominant**        | Fakes require far more specialized encodings than reals. Decoder is underperforming.                               |
| $> 0$ (small)         | **Approaching equilibrium** | Decoder produces fakes nearly as "normal" as reals to the encoder.                                                 |
| $\approx 0$           | **Equilibrium**             | Encoder cannot distinguish fakes from reals via KLD alone. Ideal state.                                            |
| $< 0$                 | **Encoder collapsing**      | Encoder finds fakes more normal than reals. Pathological — the encoder has lost its reference frame for normality. |
