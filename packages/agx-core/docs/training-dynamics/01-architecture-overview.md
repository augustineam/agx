# Architecture Overview

The Reversed Autoencoder is an adversarial variational architecture designed for **unsupervised anomaly detection**. It combines **collaborative VAE training** (encoder and decoder jointly maximize ELBO) with **adversarial training** (encoder discriminates decoder outputs via KLD; decoder fools encoder with full gradient flow through a frozen-but-differentiable encoder).

At inference, anomalies are detected by measuring discrepancies between the encoder's representations of the original input and the decoder's reconstruction — regions the decoder cannot faithfully reproduce are anomalous.

```mermaid
flowchart LR
    subgraph Encoder ["Encoder [E]"]
        direction TB
        B["Pretrained Backbone
        (frozen → thawed)"]
        H["Trainable Head
        (μ, log σ²)"]
        B --> H
    end

    subgraph Latent ["Latent Space"]
        R["Reparameterization
        z = μ + σ · ε"]
    end

    subgraph Decoder ["Decoder [D]"]
        direction TB
        D["MobileNetV3-style
        Inverted Residual Stages"]
    end

    I["Image I"] --> Encoder
    Encoder -->|"z_mean, z_log_var
    + embeddings"| Latent
    Latent -->|z| Decoder
    Decoder --> I'["Reconstruction I'"]
    I' -->|"Second pass"| Encoder
    Encoder -->|"embeddings'"| CMP["Compare
    embeddings vs embeddings'"]

    N["Noise ~ N(0,1)"] -->|"z_noise"| Decoder
    Decoder --> F["Fake Image"]
    F -->|"Encode fake"| Encoder
```

### Four Training Steps per Iteration

The training step is decomposed into four sequential sub-graphs with per-step gradient application:

```mermaid
flowchart TD
    subgraph Step1 ["Step 1: Collaborative VAE (E+D jointly)"]
        I["real"] -->|"[E]"| Z1["z_real"]
        Z1 -->|"[D]"| REC["rec_real"]
    end

    subgraph Step2 ["Step 2: Decoder Fake Path (D only, E frozen-differentiable)"]
        noise["noise"] -->|"[D₁]"| F["fake"]
        F -->|"[E]"| ZF["z_fake"]
        ZF -->|"[D₂]"| RF["rec_fake"]
    end

    subgraph Step3 ["Step 3: Decoder Rec Path (D only, E frozen-differentiable)"]
        Z1c["z_real (const)"] -->|"[D₁]"| REC2["rec_real"]
        REC2 -->|"[E]"| ZR["z_rec"]
        ZR -->|"[D₂]"| RR["rec_rec"]
    end

    subgraph Step4 ["Step 4: Encoder Critic (E only)"]
        Fc["fake (const)"] -->|"[E]"| KF["KLD_fake"]
        RECc["rec_real (const)"] -->|"[E]"| KR["KLD_rec"]
    end
```

| Step | Graph | Trains | Purpose |
| --- | --- | --- | --- |
| **1. Collaborative** | `real → [E] → z → [D] → rec` | E + D | Joint ELBO maximization — establishes normal manifold |
| **2. Fake path** | `noise → [D₁] → fake → [E_frozen] → z → [D₂] → rec_fake` | D only | Generation + cycle consistency — D₁ produces normal-looking images |
| **3. Rec path** | `z_real → [D₁] → rec → [E_frozen] → z_rec → [D₂] → rec_rec` | D only | Reconstruction + embed + cycle — D₁ matches original perceptually |
| **4. Critic** | `fake → [E]`, `rec → [E]` | E only | KLD discrimination — E learns to reject decoder outputs |

See [03-training-orchestration.md](./03-training-orchestration.md) for full details on gradient flow and design decisions.
