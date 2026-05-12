# Training Dynamics & Equilibrium

## Dual Curriculum: Complementary Pressure

The encoder (step 4) and decoder (steps 2/3) curricula create complementary adaptive pressure:

| Training phase | Encoder critic (step 4) | Decoder adversarial (steps 2/3) |
|---|---|---|
| **Early** (bad fakes) | exp(-τ_e · high_KLD) ≈ 0 → ignores easy cases | exp(-τ_d · low_ELBO) → HIGH amplification on failures |
| **Mid** (decent fakes) | Moderate signal as KLD shrinks | Moderate as ELBO improves |
| **Late** (good fakes) | exp(-τ_e · low_KLD) ≈ 1 → full discrimination effort | exp(-τ_d · high_ELBO) ≈ 1 → normal gradient (succeeding) |

The encoder starts soft and sharpens; the decoder starts aggressive and normalizes. They converge toward equilibrium from opposite directions.

## Sequential Update Dynamics

Unlike the old design where E and D operated on stale weights from the previous step, the new design creates intentional ordering:

```
Step 1: E+D cooperate → D becomes better reconstructor
Step 2/3: D (with step-1 updates) tries to fool E → D becomes adversarially better
Step 4: E (with step-1 updates) tries to discriminate → E sharpens boundaries
```

Each component operates on the **latest** version of its counterpart. This is analogous to GAN training where G and D alternate, each seeing the other's most recent update.

## Spatial Curriculum Interaction

The spatial curriculum (`spatial_temperature`) affects the MSE component of reconstruction loss across all steps:

**Step 1 (collaborative):** Focuses E+D on reconstructing structural regions → better latent codes for edges/transitions → sharper anomaly maps at inference.

**Steps 2/3 (adversarial):** Amplifies structural errors in decoder outputs → more negative ELBO → exp(-τ_d · ELBO) GROWS → decoder gets MORE gradient on structural failures. Creates positive feedback: spatial curriculum identifies where the decoder is failing structurally, and the exp curriculum amplifies that signal further. The SSIM component independently provides structural gradient even without spatial weighting, but the two are complementary — SSIM captures local structure, spatial curriculum amplifies hard regions.

**Step 4 (critic):** Spatial curriculum affects the ELBO used to compute KLD in steps 2/3 (which produces the `fake` and `rec_real` that step 4 encodes). Indirectly, better-structured decoder outputs from spatial focus make step 4's discrimination task harder over time → the encoder must develop finer discrimination.

## Why the Asymmetry Is Correct

1. **The decoder's task is harder** — unconditional generation from bottleneck without skip connections. It needs amplified signal on failures (exp curriculum) and focused signal on structure (spatial curriculum).

2. **Early discrimination is free** — bad fakes have trivially high KLD. Wasting encoder capacity here prevents it from improving its latent space for reals (step 1 suffers). The exp(-τ·KLD) curriculum naturally dampens this.

3. **Late discrimination matters most** — when the decoder produces good output, the encoder must learn subtle differences. This is when exp(-τ·KLD) approaches 1.0 and the encoder works hardest.

## Equilibrium Diagnostic

The `diff_kld` metric provides the primary adversarial balance diagnostic:

$$\Delta_{\text{KLD}} = \alpha \cdot \text{KLD}_{\text{rec}} + (1 - \alpha) \cdot \text{KLD}_{\text{fake}} - \text{KLD}_{\text{real}}$$

where $\alpha$ = `diff_kld_rec_weight` (default: 0.7).

- `KLD_real` comes from step 1 (how the encoder encodes real data)
- `KLD_fake` and `KLD_rec` come from step 4 (how the encoder encodes decoder outputs)

The weighting favors `KLD_rec` because it shares the same data lineage as `KLD_real` (both derive from the same real image) and directly reflects the deployment-relevant signal: can the encoder distinguish a round-tripped image from the original? As training progresses, `KLD_fake` naturally diverges upward (noise-originated fakes are always somewhat "weird"), making it increasingly noisy as a balance diagnostic. The 0.3 weight on `KLD_fake` retains early-training safety (detecting catastrophic encoder collapse on easy fakes) without letting late-training divergence mask issues on the rec path.

| diff_kld | State | Action |
|---|---|---|
| ≫ 0 | Encoder dominant | Pause step 4, let steps 2/3 catch up |
| > 0 (small) | Approaching equilibrium | Train all |
| ≈ 0 | Equilibrium | Train all |
| < 0 | Encoder collapsing | Pause steps 2/3, let step 4 recover |

## Validation diff_kld < 0 Is Expected

On validation data, diff_kld typically stabilizes below zero. This reflects a **generalization gap in discrimination** (encoder is less confident on unseen reals) not in reconstruction. Monitor `val_loss_rec` — as long as it decreases, the model learns effectively regardless of val_diff_kld.

## Step 1 as Stabilizer

The cooperative VAE step (step 1) **always runs** regardless of equilibrium state. This provides:

1. **Continuous reconstruction improvement** independent of adversarial balance
2. **Anchor for normality** — even when adversarial steps are paused, the model maintains its normal manifold
3. **Recovery mechanism** — if adversarial training destabilizes, step 1's cooperative signal pulls both E and D back toward a functional VAE

This is a fundamental advantage over the old design where ALL training was adversarial — there was no purely cooperative signal to stabilize the system.
