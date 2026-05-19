# Reversed Autoencoder: Training Dynamics & Loss Design

Documentation for the adversarial variational architecture designed for unsupervised anomaly detection on X-ray imagery.

## Contents

1. [Architecture Overview](./01-architecture-overview.md) — Model structure, forward paths, inference mechanism
2. [Loss Foundations](./02-loss-foundations.md) — ELBO, KLD, embedding loss, spatial curriculum, reduction strategy
3. [Training Orchestration](./03-training-orchestration.md) — 4-step collaborative + adversarial training procedure
4. [Encoder Critic Loss](./04-encoder-critic.md) — Step 4: encoder learns to discriminate via KLD
5. [Decoder Adversarial Loss](./05-decoder-adversarial.md) — Steps 2 & 3: decoder fools encoder with full gradient flow
6. [Training Dynamics & Equilibrium](./06-training-dynamics.md) — Curriculum interactions, asymmetric pressure, equilibrium
7. [Skip Connections & Equilibrium Diagnostic](./07-equilibrium-and-skip-connections.md) — Why no skip connections, diff_kld interpretation
8. [Decoder Architecture](./08-decoder-architecture.md) — MobileNetV3-symmetric design, stage configuration
9. [Callbacks](./09-callbacks.md) — Adversarial equilibrium callback, backbone thaw callback
10. [Hyperparameters](./10-hyperparameters.md) — Parameter reference, tuning guidance
11. [X-Ray Preprocessing](./11-xray-preprocessing.md) — Log transform, Beer-Lambert, spatial curriculum interaction
12. [Data Augmentation](./12-data-augmentation.md) — Physics-aware augmentation pipeline for X-ray anomaly detection
13. [Design Decisions](./13-design-decisions.md) — Considered and rejected alternatives
14. [Conditioning Architecture](./14-conditioning-architecture.md) — FiLM layers, CompositeConditionEncoder, transfer learning
15. [Latent Interpolation](./15-latent-interpolation.md) — Fake path z-sampling strategies (manifold, perturbed, slerp)

## Deprecated Sections

The following files are preserved for historical context but describe the **old training orchestration** (monolithic encoder/decoder graphs with 14 forward passes). They have been superseded by the 4-step design in [03-training-orchestration.md](./03-training-orchestration.md).

- [~~Encoder Loss (old)~~](./deprecated/03-encoder-loss-old.md) — Old encoder loss with phantom logP gradients
- [~~Decoder Loss (old)~~](./deprecated/04-decoder-loss-old.md) — Old decoder loss with severed adversarial gradients
- [~~Training Dynamics (old)~~](./deprecated/05-training-dynamics-old.md) — Old dual curriculum analysis
