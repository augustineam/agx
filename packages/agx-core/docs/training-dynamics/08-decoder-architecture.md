# Decoder Architecture

### Problem: Parameter Efficiency

The decoder must perform **unconditional generation from bottleneck** — essentially the same task as a GAN generator — without skip connections. This requires high capacity, but parameter count alone doesn't determine capability.

A 12M-parameter decoder using standard Conv2D + ResBlock is actually **less capable** than a 4M MobileNetV3 encoder because:

| Aspect          | MobileNetV3 Encoder                           | Standard ResNet Decoder                |
| --------------- | --------------------------------------------- | -------------------------------------- |
| Conv type       | Depthwise separable (~9× fewer params per op) | Standard Conv2D                        |
| Attention       | Squeeze-and-Excite at every stage             | None                                   |
| Residual design | Inverted bottleneck (expand→DW→project)       | Standard (conv→conv→add)               |
| Upsampling      | N/A                                           | ConvTranspose (checkerboard artifacts) |
| Activations     | h-swish / relu6                               | LeakyReLU                              |

ConvTranspose upsampling is particularly problematic — it creates checkerboard artifacts that the decoder wastes capacity learning to suppress.

### Solution: MobileNetV3-Symmetric Decoder

The current decoder uses MobileNetV3-style building blocks:

```
Latent z (B, H_bottleneck, W_bottleneck, C)
  → Concat with conditioning
  → [DecoderStage] × N stages
      Each stage: SpatialDropout → [InvRes × num_blocks] → UpsampleRefine 2×
  → InvRes head refinement
  → Conv 3×3 → tanh
```

**Key building blocks:**

- **InvertedResidualBlock**: expand (1×1) → depthwise (k×k) → SE → project (1×1) with residual. Same architecture as MobileNetV3 bottleneck blocks. More expressive per parameter than standard convolution.

- **UpsampleRefine**: Bilinear 2× upsample + Conv2D refinement. Eliminates checkerboard artifacts from transposed convolutions. The bilinear interpolation provides smooth spatial expansion; the learned conv refines and adjusts channels.

- **SqueezeExcite**: Channel attention at every stage. Allows the decoder to dynamically weight which channels matter at each spatial resolution.

- **SpatialDropout2D**: Applied at stage entry. Drops entire channels (not individual pixels), forcing diverse feature usage and preventing the decoder from relying on a small subset of channels.

### Stage Configuration

Default configuration mirrors MobileNetV3 Small's depth profile:

```python
DEFAULT_STAGE_CONFIG = [
    {"filters": 256, "num_blocks": 1, "expand_ratio": 2.0, "kernel_size": 5, "dropout_rate": 0.2},
    {"filters": 96,  "num_blocks": 3, "expand_ratio": 2.0, "kernel_size": 5, "dropout_rate": 0.2},
    {"filters": 48,  "num_blocks": 2, "expand_ratio": 1.5, "kernel_size": 3, "dropout_rate": 0.1},
    {"filters": 40,  "num_blocks": 3, "expand_ratio": 1.5, "kernel_size": 3, "dropout_rate": 0.1},
    {"filters": 24,  "num_blocks": 2, "expand_ratio": 1.5, "kernel_size": 3, "dropout_rate": 0.05},
    {"filters": 16,  "num_blocks": 1, "expand_ratio": 1.0, "kernel_size": 3, "dropout_rate": 0.0},
]
```

Design rationale:

- **More blocks at mid-resolution** (14×14 → 56×56): This is where structural detail is established. Low-res stages define coarse layout; high-res stages just add texture.
- **Larger kernels at low resolution** (5×5): Low-res features have larger effective receptive fields; 5×5 depthwise is cheap at small spatial sizes.
- **Decreasing dropout**: High dropout at coarse stages (force robustness), zero at final stage (preserve fine detail).
- **Conservative expand ratios** (1.0–2.0): The decoder doesn't need the extreme expansion (6×) used in classification encoders. Generation benefits more from depth (more blocks) than width (higher expansion).

### Conditioning: FiLM (Feature-wise Linear Modulation)

The decoder uses **FiLM conditioning** at every stage — the conditioning vector modulates feature maps via per-channel affine transformation:

$$\text{output} = \text{features} \cdot (1 + \gamma) + \beta$$

where $\gamma, \beta \in \mathbb{R}^C$ are predicted from the conditioning vector $c \in \mathbb{R}^{64}$ via a 2-layer MLP projection. FiLM modulates behavior at every resolution without inflating channel dimensions.

```
z → stem → [FiLM(c) → Stage_0] → [FiLM(c) → Stage_1] → ... → to_rgb
```

Each FiLM layer has its own projection weights, allowing different modulation patterns at each resolution. Early stages modulate coarse structure (e.g., "this product is a can vs. a jar"), while later stages modulate fine texture (e.g., "this product has a textured label vs. smooth glass").

The conditioning vector `c` comes from a `CompositeConditionEncoder` that maps categorical IDs (machine, view, product) to a fixed-size dense embedding. See [14-conditioning-architecture.md](./14-conditioning-architecture.md) for full details on the conditioning system.

### Progressive Training (Future)

Since the decoder must generate from scratch without skip connections, it faces the same challenge as GAN generators — learning all spatial scales simultaneously. The ProGAN insight applies: train coarse-to-fine progressively.

```
Stage 1: z → 7×7 → bilinear_upsample(224×224) → encoder → loss
Stage 2: z → 7×7 → 14×14 → bilinear_upsample(224×224) → encoder → loss
  ...
Stage 6: z → 7×7 → ... → 224×224 → encoder → loss
```

With alpha blending during transitions:

$$\text{output} = (1 - \alpha) \cdot \text{upsample}(\text{prev\_stage}) + \alpha \cdot \text{current\_stage}$$

The encoder always sees full-resolution images (bilinearly upsampled). Each stage reaches convergence before the next is activated, preventing cross-scale gradient conflicts.
