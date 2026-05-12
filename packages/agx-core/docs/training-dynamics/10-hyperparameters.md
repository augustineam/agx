# Hyperparameter Reference

## Model Hyperparameters

| Parameter             | Default | Range      | Effect                                                               |
| --------------------- | ------- | ---------- | -------------------------------------------------------------------- |
| `beta_kld`            | 0.25    | 0.01–2.0   | KLD weight in all ELBO terms. <1 = reconstruction focus; ≥1 = latent structure |
| `enc_expkld_temp`     | 1.0     | 0.5–3.0    | Encoder critic curriculum (step 4). Higher = ignore easy discriminations more |
| `dec_expelbo_temp`    | 1.0     | 0.5–2.0    | Decoder adversarial curriculum (steps 2/3). Higher = more focus on failure modes |
| `spatial_temperature` | 1.0     | 0.0–20.0   | Spatial hard-pixel mining in MSE component. Higher = more focus on structural regions |
| `lambda_embed`        | 1.0     | 0.1–5.0    | Embedding loss weight (step 3). Higher = more perceptual fidelity    |
| `alpha_ssim`          | 0.3     | 0.0–0.5    | SSIM blend weight in reconstruction loss. 0 = pure MSE              |
| `diff_kld_rec_weight` | 0.7     | 0.5–1.0    | Weight of kld_rec in diff_kld diagnostic. Higher = less influence from kld_fake |

## Callback Hyperparameters

| Parameter             | Default | Range      | Effect                                                               |
| --------------------- | ------- | ---------- | -------------------------------------------------------------------- |
| `upper_threshold`     | 2.0     | 1.0–5.0    | diff_kld above which encoder (step 4) is paused                      |
| `lower_threshold`     | -0.5    | -2.0–0.0   | diff_kld below which decoder (steps 2/3) is paused                   |
| `ema_momentum`        | 0.99    | 0.95–0.999 | EMA smoothing for equilibrium callback                               |
| `min_pause_steps`     | 50      | 20–200     | Minimum steps a component stays paused                               |
| `backbone_lr_factor`  | 0.01    | 0.001–0.1  | Backbone LR relative to head after thawing                           |

## Curriculum Parameter Interactions

`spatial_temperature` and the exp-curriculum temperatures interact:

- **Spatial curriculum** amplifies structural errors in the MSE component → ELBO values shift more negative
- **SSIM** provides independent structural gradient that doesn't interact with spatial curriculum (SSIM has its own Gaussian windowing)
- **Decoder exp(-τ_d · ELBO)** amplifies these (more negative ELBO → larger exp → more gradient on structural failures)
- **Encoder exp(-τ_e · KLD)** is independent (KLD comes from (μ,σ), not from pixel errors)

### Tuning Guide

1. Start with `spatial_temperature = 5.0` for single-channel X-rays
2. If equilibrium callback triggers excessively → increase `dec_expelbo_temp` (decoder tries harder on failures)
3. If training becomes unstable (loss spikes) → decrease `dec_expelbo_temp` or add gradient clipping
4. Monitor `expelbo_rec` / `expelbo_fake` — if consistently > 20, reduce temperatures
5. `enc_expkld_temp` rarely needs adjustment — it operates on KLD directly, unaffected by spatial curriculum

### Parameter Name Changes from Old Design

| Old name | New name | Reason |
|---|---|---|
| `enc_expelbo_temp` | `enc_expkld_temp` | Encoder uses pure KLD (not ELBO) for discrimination |
