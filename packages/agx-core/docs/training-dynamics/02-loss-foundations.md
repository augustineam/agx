# Loss Foundations

### Evidence Lower Bound (ELBO)

The Evidence Lower Bound comes from variational inference. For a generative model with observed data $x$, latent variables $z$, and model parameters $\theta$, we want to maximize the log-evidence $\log p_\theta(x)$. Since this is intractable, we instead maximize a lower bound:

$$\log p_\theta(x) \geq \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{Reconstruction}} - \underbrace{D_{\text{KL}}(q_\phi(z|x) \| p(z))}_{\text{Regularization}} = \text{ELBO}$$

The ELBO decomposes into two competing objectives:

- **Reconstruction term** $\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$: Measures how well the decoder can reconstruct the input from the latent code. Maximizing this pushes the model toward faithful reconstructions. In practice, this is computed as the negative mean pixel MSE:

$$\log p_\theta(x|z) \approx -\text{mean}_{h,w}\left[\text{MSE}(x, \hat{x})\right]$$

- **KL Divergence** $D_{\text{KL}}(q_\phi(z|x) \| p(z))$: Closed-form KLD from $\mathcal{N}(\mu, \text{diag}(\exp(\text{logvar})))$ to the standard normal prior $p(z) = \mathcal{N}(0, I)$:

$$D_{\text{KL}} = \frac{1}{2C} \sum_{c=1}^{C} \left( \mu_c^2 + \exp(\text{logvar}_c) - \text{logvar}_c - 1 \right)$$

Reduced via **mean** over both channels and spatial dimensions to produce a scalar per sample. This replaces the previous Monte Carlo estimate (`log q(z|x) - log p(z)`) which has non-zero variance — the closed form is exact, lower variance, and cheaper.

### Reduction Strategy: Mean over Everything

All loss terms use **mean reduction** over spatial dimensions $(H, W)$ and channels $C$. This ensures:

1. **Consistent magnitudes** regardless of latent spatial resolution or image size
2. **Interpretable ranges** — reconstruction loss (MSE) lives in $[0, \sim2]$ for normalized inputs, ELBO in $[-2, 0]$, KLD in $[0, \sim1]$
3. **Meaningful hyperparameters** — `beta_kld` and `lambda_embed` don't need resolution-dependent tuning

The previous approach used an adaptive scaling factor $\alpha = 32 / \sqrt{H_{\text{latent}} \cdot W_{\text{latent}}}$ and sum reductions, which produced large values that obscured the relative balance between terms.

### ELBO Composition

With mean reductions and the `beta_kld` weighting:

$$\text{ELBO} = \underbrace{-\text{mean}_{h,w}[\text{MSE}(x, \hat{x})]}_{\text{logpx\_z} \in [-2, 0]} - \beta_{\text{kld}} \cdot \underbrace{\text{mean}_{h,w}[\text{KLD}]}_{\in [0, \sim1]}$$

Where `beta_kld` controls the reconstruction-regularization tradeoff:

- $\beta_{\text{kld}} < 1$: Prioritize reconstruction fidelity (current default: 0.25)
- $\beta_{\text{kld}} \geq 1$: Prioritize latent structure and disentanglement

### Embedding Loss

Feature consistency loss between two sets of intermediate encoder representations, combining MSE and cosine similarity:

$$\mathcal{L}_{\text{embed}} = \frac{1}{L} \sum_{l=1}^{L} \text{mean}_{h,w} \left[ \frac{1}{2} \| f_l - f_l' \|^2_C + \left(1 - \frac{f_l \cdot f_l'}{\|f_l\|_C \|f_l'\|_C}\right) \right]$$

Where $f_l$ and $f_l'$ are the encoder's intermediate features at layer $l$ for the original input and its reconstruction respectively. The MSE component captures magnitude differences while cosine similarity captures directional alignment in feature space. Both are computed along the channel axis $C$.

#### Depth-Weighted Embedding Loss (Optional)

When embedding loss is hard to decrease, deeper layers (closer to bottleneck) can be given exponentially higher weight since their information is preservable through the VAE bottleneck, while shallow layers' high-resolution spatial detail is fundamentally lost:

$$\mathcal{L}_{\text{embed}}^{\text{weighted}} = \frac{\sum_{l=1}^{L} w_l \cdot \mathcal{L}_l}{\sum_{l=1}^{L} w_l}, \quad w_l = 2^{l - L}$$

For 5 layers: weights $\approx [0.06, 0.12, 0.25, 0.50, 1.00]$. This prevents the decoder from being punished for an impossible task (reproducing high-frequency spatial detail that the bottleneck cannot preserve) while still receiving gradient signal for preservable deep features.

A middle-ground variant uses **cosine-only for shallow layers** (directional alignment is achievable even when magnitude matching is not) and full MSE + cosine for deep layers.

### Pixel MSE with Spatial Curriculum

Per-pixel reconstruction error reduced along the channel dimension:

$$\text{MSE}_{\text{pixel}}(x, \hat{x}) = \frac{1}{C} \sum_{c=1}^{C} (x_c - \hat{x}_c)^2$$

This produces a spatial error map of shape $[B, H, W]$ that preserves spatial information for downstream loss aggregation.

#### The Spatial Dilution Problem

Single-channel X-ray images are information-sparse: 70–80% of pixels are near-uniform background with trivially low reconstruction error. The spatial mean drowns out localized failures in structurally complex regions:

$$\text{mean}_{h,w}[\text{MSE}] \approx 0.8 \times 0.001 + 0.2 \times 0.1 = 0.021$$

Even halving the error on hard regions (0.1 → 0.05) only changes the mean from 0.021 → 0.011 — a tiny gradient signal. The network reaches low MSE by reconstructing uniform regions well while neglecting structural detail.

This creates a cascade: the decoder has no incentive to improve on edges and transitions → embedding loss plateaus (features diverge in structural regions) → the encoder easily discriminates fakes → encoder dominance → equilibrium callback triggers excessively.

#### Spatial Curriculum Weighting

To address this, `pixel_mse` supports exponential spatial curriculum weighting controlled by `spatial_temperature` ($\tau_s$):

$$\text{MSE}_{\text{spatial}}(x, \hat{x}) = w_{h,w} \cdot \text{MSE}_{\text{pixel}}(x, \hat{x})$$

where the weights are:

$$w_{h,w} = \frac{\exp\left(\tau_s \cdot \overline{\text{MSE}}_{h,w}\right)}{\text{mean}_{h,w}\left[\exp\left(\tau_s \cdot \overline{\text{MSE}}_{h,w}\right)\right]}$$

and $\overline{\text{MSE}}_{h,w}$ is the stop-gradiented error map (weights are treated as constants during backpropagation to prevent second-order effects).

**Key properties:**

- **Per-sample normalization** preserves overall loss magnitude — existing hyperparameters (`beta_kld`, `lambda_embed`) don't need re-tuning.
- **Stop gradient on weights** prevents the network from minimizing loss by making weights small (producing uniform error) rather than fixing hard regions.
- At $\tau_s = 0$: reduces to uniform spatial mean (original behavior).
- At $\tau_s = 5$: structural regions (~0.1 error) get ~1.6× weight vs background.
- At $\tau_s = 10$: structural regions get ~2.7× weight.
- At $\tau_s = 20$: structural regions get ~7× weight.

**Applied across all training steps** — see [05-decoder-adversarial.md](./05-decoder-adversarial.md) and [06-training-dynamics.md](./06-training-dynamics.md) for interaction with curriculum weighting.
