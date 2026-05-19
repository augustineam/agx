# Design Decisions Considered & Rejected

| Idea                                  | Verdict     | Reason                                                                                  |
| ------------------------------------- | ----------- | --------------------------------------------------------------------------------------- |
| Monolithic 2-graph orchestration      | ❌ Replaced | 14 passes, 2.1GB peak; encoder got phantom logP gradients; decoder got no adversarial signal through E |
| Severed KLD in decoder (stop_gradient)| ❌ Replaced | D never learned "produce images that encode normally" — the whole adversarial point     |
| `torch.no_grad` on E during D's turn  | ❌ Replaced | Saves zero memory (activations stored regardless); kills gradient flow through E to D   |
| 3 optimizers (VAE + E + D)            | ❌          | Conflicting momentum states on shared params; 2-opt with per-step apply is cleaner      |
| Gradient accumulation across all steps| ❌          | Averages cooperative + adversarial signals; sequential apply preserves GAN-style ordering |
| Stop-gradient z before D₂ (steps 2/3) | ❌          | Loses cycle consistency signal; D₁ no longer learns "produce reconstructable images"    |
| Adversarial embed in encoder          | ❌          | No-op when frozen; corrupts features when thawed; redundant with KLD discrimination     |
| `expelbo_real` in encoder             | ❌          | Curriculum on fitting term reduces signal when reals are bad = wrong direction          |
| `expelbo_real` in decoder             | ❌          | Neglects easy samples → reconstruction drift → false positive anomalies at inference    |
| Per-sample curriculum on embed_loss   | ❌          | Feature space has natural importance weighting; spatial curriculum addresses root cause |
| SSIM as primary reconstruction loss   | ❌          | Window-based → blurs spatial resolution; partially redundant with embedding loss         |
| Skip connections (encoder→decoder)    | ❌          | Leaks anomalous features, suppresses anomaly signal                                     |
| Sum reduction in losses               | ❌ Replaced | Resolution-dependent magnitudes; hard to interpret/tune                                 |
| `log_normal_pdf` for KLD              | ❌ Replaced | Monte Carlo estimate with variance; closed form is exact and cheaper                    |
| Diffusion decoder                     | ❌          | Multiple denoising steps kills CPU real-time inference                                  |
| ViT encoder                           | ❌          | Slower on CPU than MobileNet; no natural multi-scale features for spatial anomaly maps  |
| Standard ConvTranspose upsampling     | ❌ Replaced | Checkerboard artifacts waste decoder capacity                                           |
| Standard contrast (at grayscale mean) | ❌          | Operates at mean intensity; compresses the high-density band where FOs live             |
| Concat conditioning at bottleneck     | ❌ Replaced | Channel count changes with vocab size; breaks transfer learning; no multi-resolution conditioning |
| FiLM inside encoder backbone          | ❌          | Fights pretrained features; requires unfreezing to learn modulation; backbone should be product-agnostic |
| Condition encoder trains from reconstruction loss | ❌ | Creates "cheat sheet" shortcuts; condition should learn from KLD only |
| Slerp interpolation for fake path     | ❌          | Latent space is Gaussian (flat), not hypersphere; slerp ≈ lerp + numerical instability |
| Perturbed noise for fake path         | ❌ Deprecated | Redundant with reparameterization trick; no manifold guarantee |
| Pure noise `z ~ N(0,I)` for fake path | ❌          | Too far from manifold early in training; decoder wastes capacity on OOD inputs |
| Shared embedding table across fields  | ❌          | Machine ID 2 ≠ View ID 2 ≠ Product ID 2; conflates semantically different categories |
| Condition encoder in adversarial steps | ❌          | Adversarial signal corrupts embeddings; condition trains ONLY from collaborative KLD |
| 3 optimizers (enc + dec + cond)       | ⚠️ Optional | Supported but default shares enc_optimizer with condition encoder (simpler, works well) |

### Why FiLM Over Concatenation

The original decoder used `Concatenate([z, c])` at the bottleneck. This was replaced with FiLM for multi-product training:

1. **Fixed channel dimensions**: FiLM's `embed_dim=64` is constant regardless of number of products/views. Concat required `z_channels + cond_channels` — adding products changed conv input dimensions, requiring new weights.

2. **Transfer learning**: Swapping the condition encoder (new products) requires zero changes to decoder conv weights with FiLM. With concat, every downstream convolution must be retrained.

3. **Multi-resolution conditioning**: FiLM at every decoder stage provides condition-specific modulation at all spatial resolutions. Concat only conditions at the bottleneck — downstream stages have no direct condition access.

4. **Edge deployment**: FiLM with a constant condition collapses to a fixed affine transform that can be fused into adjacent BatchNorm layers — zero runtime cost. Concat always requires the extra channels.

### Why Manifold Interpolation (Not Noise) for Fake Path

The fake path (Step 2) needs `z_fake` samples that are:
- On or near the learned manifold (so the decoder can produce meaningful images)
- Different from `z_real` (so the encoder gets novel discrimination targets)

**Manifold interpolation** (`mode="manifold", manifold_op="roll"`) satisfies both: interpolating between two on-manifold points (both are encodings of real images) produces points that stay approximately on-manifold while being distinct from either endpoint.

Alternatives fail because:
- Pure noise is too far from the manifold early in training
- Perturbed noise (z + ε) has no manifold guarantee and is redundant with reparameterization
- Slerp assumes hypersphere geometry that doesn't match KLD-regularized Gaussian latents

See [15-latent-interpolation.md](./15-latent-interpolation.md) for full analysis.

### Why MSE (Not SSIM) for `logpx_z`

SSIM (Structural Similarity Index) was considered as an alternative or complement to MSE for the reconstruction term. While SSIM has appealing properties for perceptual quality (luminance + contrast + structure decomposition), it was rejected for the reconstruction loss:

1. **Window-based computation blurs spatial precision**: SSIM operates in 11×11 local windows, averaging structural similarity over patches. This directly counteracts the spatial curriculum's purpose — fine-grained per-pixel focus on hard regions. A pixel-level error at coordinates (h, w) gets smeared over its 11×11 neighborhood in SSIM space.

2. **Partially redundant with embedding loss**: SSIM captures "does this region look structurally similar?" — the same signal the embedding loss provides in learned feature space. The embedding loss is strictly more powerful (learned, multi-scale, adapted to the encoder's actual representations) while SSIM is a fixed handcrafted heuristic.

3. **Gradient landscape issues**: SSIM has known gradient plateaus when images are already reasonably similar (the denominator stabilization terms $c_1$, $c_2$ flatten gradients near SSIM ≈ 1). MSE provides consistent gradient magnitude regardless of current similarity level — critical for continued improvement.

4. **Interaction with spatial curriculum**: MSE produces a clean per-pixel error map that the spatial curriculum can precisely weight. SSIM's window-based output would require a different weighting strategy.

The architecture already achieves structural reconstruction fidelity through:

- **Spatial curriculum** on MSE → pixel-level structural focus
- **Embedding loss** → learned perceptual/structural fidelity
- **Log preprocessing** → linearized intensity makes MSE physically meaningful

Adding SSIM would be a third structural signal with diminishing returns and engineering cost. If a future ablation shows `loss_rec` low but anomaly maps blurry (MSE is good but structures are shifted), MS-SSIM as an auxiliary loss term (not replacing MSE) could be reconsidered.
