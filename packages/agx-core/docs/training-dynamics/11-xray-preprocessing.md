# X-Ray Data Representation & Preprocessing

### The Physical Model

X-ray image formation follows the Beer-Lambert law:

$$I = I_0 \cdot \exp(-\mu \cdot t)$$

Where $I_0$ is the source intensity, $\mu$ is the linear attenuation coefficient (material property), and $t$ is the material thickness. The raw pixel value is **exponentially** related to the physical quantity of interest (total attenuation $\mu \cdot t$).

This exponential relationship creates a fundamental problem: the same foreign object (fixed $\Delta\mu$) produces **different pixel-space contrast** depending on what's behind it:

- FO behind thin content (low background attenuation): large $\Delta I$
- FO behind thick glass (high background attenuation): small $\Delta I$

In pixel space, anomaly signal is inconsistent. The model must implicitly learn the exponential inverse to detect anomalies uniformly — wasting capacity on a task that can be solved analytically.

### Log Transform: Linearizing Attenuation

The log transform converts exponential intensity to linear attenuation space:

$$-\log(I / I_0) = \mu \cdot t$$

After log-linearization:

- Material density differences become **additive and linear**
- A foreign object with attenuation $\Delta\mu$ produces a **constant offset** regardless of background density
- The reconstruction task is "fairer" — the model operates in a space where differences are physically meaningful and uniform

Applied as the first preprocessing step (before any brightness/contrast/normalization):

```python
# LogTransform: linearize X-ray attenuation
log_image = log(255 + ε) - log(image + ε)  # invert for transmission X-ray
# Rescale to [0, 255] for downstream transforms
log_image = (log_image - min) / (max - min) * 255
```

The `invert=True` mode ensures high attenuation (dense material) maps to high pixel value, matching the convention where foreign objects appear bright.

### Brightness Adjustment

After log-linearization, brightness adjustment (`+20` on [0, 255]) shifts the working range:

- Makes background (low density, low pixel value) less dark
- Preserves high-density regions (glass edges, metal caps)
- In the normalized [-1, 1] space: expands the gap between content (~0) and dense structures (~0.8–0.9)

Critically, brightness in log-space has **uniform effect** across the image (unlike raw exponential space where +20 means different things at different densities).

### Sigmoidal Contrast (For Specific Density Bands)

When content and foreign objects occupy a similar intensity band (e.g., both in 0.5–0.9 after normalization), standard gamma correction is suboptimal:

- $\gamma < 1$: expands shadows, **compresses** the 0.5–0.9 band of interest ❌
- $\gamma > 1$: expands highlights but darkens everything

Sigmoidal contrast applies a steep S-curve centered on the band of interest:

$$S(x) = \frac{1}{1 + \exp(-g \cdot (x - c))}$$

Where $c$ is the center intensity (e.g., 0.7 for mid-content) and $g$ controls steepness. This expands contrast **precisely** where FOs and content overlap, maximizing their pixel-space separation without distorting other regions.

### Preprocessing Order

```
Raw X-ray [0, 255]
  → LogTransform (linearize attenuation)      ← physically motivated
  → BrightnessAdjust (+20)                     ← shift working range
  → [Optional: SigmoidalContrast(center=0.7)]  ← expand FO/content band
  → InvertImg                                  ← convention: dense = bright
  → Resize
  → Normalize (mean=0.5, std=0.5) → [-1, 1]   ← model input range
```

### Impact on Anomaly Detection

Preprocessing that makes normal reconstruction slightly harder but anomalies **much** more evident is a net win:

- Without log transform: FO behind glass produces ΔI = 5 in raw space
- With log transform: same FO produces Δ(log I) = constant everywhere

The signal-to-noise ratio for anomaly detection improves because:

1. Reconstruction error at anomaly locations increases more than baseline error
2. The model doesn't waste capacity learning the exponential inverse
3. Spatial curriculum focuses gradient on structural regions where linearized anomaly contrast is now consistent

### Interaction with Spatial Curriculum

Log transform and spatial curriculum are **complementary**:

- **Log transform**: Makes the _data_ more linearly separable (FOs produce consistent contrast regardless of position)
- **Spatial curriculum**: Makes the _loss_ focus on structurally complex regions (where the now-consistent FO signal lives)

Without log transform, spatial curriculum amplifies reconstruction errors that vary in magnitude across the image (same FO, different contrast). With log transform, the amplified errors have consistent physical meaning — the model learns a uniform normal manifold.
