> **DEPRECATED**: This section describes the old training orchestration (monolithic 14-pass graphs). See [../03-training-orchestration.md](../03-training-orchestration.md) for the current design.

# Loss Weight Balance & Asymmetric Training Dynamics

### Effective Weight Analysis

```
Encoder: -elbo_real + 0.5 * (exp(τ_e · elbo_rec) + exp(τ_e · elbo_fake))
          \_______/   \______________________________________________/
           weight 1.0    weight 0.5 each, curriculum-damped (focus on hard-to-reject)

Decoder: -elbo_real + 0.5 * (exp(-τ_d · elbo_rec) + exp(-τ_d · elbo_fake)) + λ * embed_loss
          \_______/   \________________________________________________/   \___________/
           weight 1.0    weight 0.5 each, curriculum-amplified (focus on hard-to-fool)
                                                                             extra term
```

### Dual Curriculum: Complementary Focus

The encoder and decoder curricula create a **complementary adaptive pressure** system:

| Training phase         | Encoder signal                                  | Decoder signal                                      |
| ---------------------- | ----------------------------------------------- | --------------------------------------------------- |
| **Early** (bad fakes)  | `≈ -elbo_real + 0` (expelbo dampens bad fakes)  | HIGH: exp(-τ_d · bad_ELBO) amplifies failing cases  |
| **Mid** (decent fakes) | `-elbo_real + moderate` (some rejection signal) | MODERATE: signal normalizes as decoder improves     |
| **Late** (good fakes)  | `-elbo_real + full` (hard-to-reject dominate)   | LOW: exp(-τ_d · good_ELBO) ≈ 1 (decoder succeeding) |

The two curricula interact to **compress the equilibrium gap from both sides**:

- **Encoder self-regulates**: spatial curriculum makes fakes look worse → expelbo shrinks → encoder relaxes adversarial pressure
- **Decoder focuses on failures**: inverted expelbo amplifies gradient on samples the encoder easily rejects → decoder catches up faster

```
                        Encoder                          Decoder
                        ───────                          ───────
Spatial curriculum:     Amplifies structural errors      Amplifies structural errors
                              │                                │
                              ▼                                ▼
Effect on ELBO values:  All ELBOs more negative          All ELBOs more negative
                              │                                │
                              ▼                                ▼
Interaction with        exp(+τ_e · ELBO) SHRINKS         exp(-τ_d · ELBO) GROWS
sample curriculum:      (bad fakes dampened more)         (hard samples amplified more)
                              │                                │
                              ▼                                ▼
Net adversarial         LESS aggressive                  MORE aggressive
pressure:               (self-regulates)                 (focuses on failures)
                              │                                │
                              ▼                                ▼
Equilibrium:            ←── GAP NARROWS ──→
```

### Why the Asymmetry Is Correct

The encoder starts soft and sharpens over time; the decoder starts aggressive on failures and normalizes as it improves. This is appropriate because:

1. **The decoder's task is harder** — unconditional generation from bottleneck without skip connections requires consistent strong signal, especially on failure modes
2. **Early discrimination is free** — bad fakes are trivially rejected; wasting encoder capacity on them prevents it from building a good latent space for reals
3. **Late discrimination matters most** — when fakes are good, the encoder must learn subtle differences, which is when expelbo weight approaches 1.0
4. **Decoder must converge faster** — the decoder curriculum provides amplified signal exactly where it's struggling, accelerating convergence toward equilibrium

### Tuning the Balance

The 0.5 coefficient gives equal total weight to fitting (1.0) vs adversarial (0.5 + 0.5 = 1.0). Adjustments:

- If encoder becomes too aggressive (decoder can't keep up): reduce to 0.3 each
- If decoder runs away (encoder can't discriminate): increase to 0.7 each
- `enc_expelbo_temp` ($\tau_e$): higher values dampen easy rejections more, giving the decoder more head start
- `dec_expelbo_temp` ($\tau_d$): higher values amplify decoder's focus on failure modes; if training becomes unstable (loss spikes), reduce
- `spatial_temperature` ($\tau_s$): controls within-image focus for both; higher values concentrate gradient on structurally complex regions
