# Callbacks

### Adversarial Equilibrium Callback

Monitors an exponential moving average (EMA) of $\Delta_{\text{KLD}}$ and dynamically pauses encoder or decoder training to maintain balance.

```
    diff_kld (EMA)
    ──────────────────────────────────────
    ▲
    │   Encoder dominant
    │   → Pause step 4, steps 2/3 catch up
    │ ─ ─ ─ upper threshold ─ ─ ─
    │
    │   Healthy zone → train all steps
    │
    │ ─ ─ ─ lower threshold ─ ─ ─
    │   Encoder collapsing
    │   → Pause steps 2/3, step 4 recovers
    ▼
```

**Step 1 (collaborative VAE) always runs** — it is never adversarial and provides the stable reconstruction foundation regardless of equilibrium state.

**Design principles:**

- **Hysteresis via minimum pause duration**: Once a component is paused, it stays paused for at least $N$ steps, even if the EMA briefly re-enters the healthy zone. This prevents rapid oscillation where one step of recovery immediately triggers a resume, followed by an immediate re-breach.

- **EMA smoothing**: Instantaneous $\Delta_{\text{KLD}}$ is noisy batch-to-batch. The EMA (default momentum $0.99 \approx$ 100-step window) filters transient spikes while remaining responsive to genuine regime shifts.

- **Asymmetric thresholds**: The lower threshold is tighter than the upper (e.g., $-0.5$ vs $+2.0$) because encoder collapse ($\Delta < 0$) is far more dangerous than encoder dominance ($\Delta \gg 0$). Encoder dominance simply means the decoder needs more training; encoder collapse means the model's concept of normality is corrupted.

- **No-grad forward when paused**: When steps 2/3 are paused, a no-grad forward pass still produces `fake` and `rec_real` tensors needed by step 4. When step 4 is paused, its forward still provides diagnostic `kld_fake`/`kld_rec` for monitoring.

### Validation `diff_kld < 0` Is Expected

A consistent observation is that $\Delta_{\text{KLD}}$ on the validation set stabilizes below zero while the training set reaches equilibrium. This is expected:

- The encoder's trainable head is optimized for training data — it has higher discrimination confidence on samples it has seen.
- On novel validation images, the encoder is less certain about reals, narrowing the KLD gap or pushing it negative.
- This reflects a **generalization gap in discrimination**, not in reconstruction. As long as `val_loss_rec` continues decreasing, the model is learning effectively.

### Backbone Thaw Callback

Monitors validation reconstruction loss and unfreezes the pretrained encoder backbone when training plateaus, applying a discriminative learning rate.

**Prerequisites before thawing (all must be met):**

1. **Decoder reconstruction has converged**: `loss_rec` and `loss_embed` are plateaued or slowly decreasing for an extended period ($\sim$ 20–30 epochs). The decoder must produce good-enough reconstructions that fine-tuning the backbone adapts to meaningful signal, not noise.

2. **Adversarial equilibrium is stable**: The equilibrium callback should be in the "train both" state consistently, not oscillating between pauses.

3. **Validation metrics have plateaued**: If `val_loss_rec` is still improving with a frozen backbone, thawing is premature — there are still free gains available without risking pretrained features.

**Thawing procedure:**

The backbone learning rate is set to a small fraction of the head's learning rate (e.g., $0.01\times$) to prevent **catastrophic forgetting** of pretrained features. The frozen backbone phase is where the decoder learns the "language" of the encoder's features; thawing is the final refinement to adapt those features from the pretrained domain (e.g., ImageNet) to the target domain (e.g., X-ray imagery).

```
Epoch 0                   Plateau detected         Post-thaw
│                         │                        │
▼                         ▼                        ▼
┌─────────────────────────┬────────────────────────┐
│  Frozen backbone        │  Thawed backbone       │
│                         │  (discriminative LR)   │
│  Train: head + decoder  │  Train: all            │
│  Goal: decoder learns   │  Goal: adapt backbone  │
│  encoder's feature      │  features to target    │
│  "language"             │  domain                │
└─────────────────────────┴────────────────────────┘
```
