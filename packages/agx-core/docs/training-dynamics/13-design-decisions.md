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
