# Data Augmentation for X-Ray Anomaly Detection

## Design Philosophy

The augmentation pipeline is structured around **X-ray physics**, not generic image augmentation. Each transform corresponds to a real-world source of variation in X-ray acquisition systems. The pipeline separates three distinct concerns:

1. **Deterministic physics preprocessing** — defines the input space (always applied, never randomized)
2. **Stochastic calibration variation** — simulates machine-to-machine differences
3. **Stochastic geometric augmentation** — simulates production line variability

### Why Augment for Anomaly Detection?

1. **Inter-product variation**: X-rays from one product to the next can be very different. Augmentations teach the model to be invariant to acquisition nuances.

2. **Acquisition artifacts**: Slight variations in image capture (positioning, exposure, voltage) should NOT trigger anomaly detection. Augmentations ensure these variations are absorbed into the "normal" manifold.

3. **Focus on true anomalies**: By making the model robust to expected visual variation, the anomaly signal becomes specific to genuine defects rather than imaging conditions.

The trade-off is that aggressive augmentations make reconstruction harder (the decoder must reproduce augmented versions, not just canonical views), which is why decoder architecture capacity matters.

---

## Pipeline Architecture

### Training Pipeline

```python
def train_transforms(img_size):
    return A.Compose([
        # ─── Geometry correction (raw image space) ───────────────
        Deskew(p=0.85),                    # Sometimes see raw (uncorrected)

        # ─── Position/size variation ────────────────────────────
        A.OneOf([
            A.Pad((25, 25), 255),
            A.Pad((50, 50), 255),
            A.Pad((75, 75), 255),
            A.NoOp(),                      # Sometimes no padding
        ], p=0.75),

        # ─── Physics preprocessing (ALWAYS ON, params vary) ─────
        LogTransform(epsilon=1, invert=True, p=1.0),
        A.OneOf([
            GammaCorrection(center=0.8,  gain=5),
            GammaCorrection(center=0.85, gain=7),
            GammaCorrection(center=0.9,  gain=9),   # canonical
            GammaCorrection(center=0.95, gain=11),
        ], p=1.0),

        A.Resize(img_size, img_size),

        # ─── Geometric augmentation ────────────────────────────
        A.Affine(scale=(0.88, 0.97), rotate=(-90, 90), shear=(-5, 5), p=0.5),
        A.RandomRotate90(p=0.5),

        # ─── Sensor/acquisition noise ──────────────────────────
        A.OneOf([
            A.GaussianBlur(blur_range=(1, 3)),
            A.MotionBlur(blur_range=(3, 7)),
            A.GaussNoise(std_range=(0.01, 0.03)),
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(1, 5),
                hole_width_range=(1, img_size),
                fill=255,  # Dead pixel lines (white in inverted space)
            ),
            A.ShotNoise(scale_range=(0.01, 0.5)),
        ], p=0.3),

        # ─── Subtle intensity drift (calibration variation) ────
        A.RandomBrightnessContrast(
            brightness_range=(-0.05, 0.05),
            contrast_range=(-0.05, 0.05),
            p=0.2,
        ),

        A.Normalize(mean=[0.5], std=[0.5]),
    ])
```

### Validation Pipeline (Fully Deterministic)

```python
def valid_transforms(img_size):
    return A.Compose([
        Deskew(p=1.0),                              # Always correct
        LogTransform(epsilon=1, invert=True, p=1.0),  # Always apply
        GammaCorrection(center=0.9, gain=9, p=1.0),   # Canonical params
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.5], std=[0.5]),
    ])
```

Validation uses **fixed, canonical parameters** for every transform. This ensures:
- Repeatable metrics across runs
- Fair comparison between epochs
- The model is evaluated on the "ideal" preprocessing path

---

## Transform Categories Explained

### 1. Physics Preprocessing (Always On, p=1.0)

These transforms define the **input space**. They are not augmentations — they are signal processing that converts raw sensor data to a representation suitable for learning.

| Transform | Purpose | Why Always On |
|-----------|---------|---------------|
| `LogTransform` | Linearizes Beer-Lambert attenuation (see [11-xray-preprocessing.md](./11-xray-preprocessing.md)) | Without it, anomaly contrast is position-dependent. The model MUST operate in log-space. |
| `GammaCorrection` | Sigmoidal contrast expansion in the density band where FOs live | Without it, content and foreign objects are compressed into the same pixel range. |

**Varying GammaCorrection parameters** during training simulates calibration differences between machines. Different X-ray systems have slightly different voltage/current settings, producing images that are "equally correct" but have different contrast curves. The model must be invariant to these:

- `center=0.8, gain=5` — under-tuned machine (low contrast)
- `center=0.85, gain=7` — slightly conservative
- `center=0.9, gain=9` — canonical (used at validation/inference)
- `center=0.95, gain=11` — over-tuned machine (high contrast)

All produce valid preprocessed images; the model should give the same anomaly decision regardless.

### 2. Geometry Correction (Deskew)

`Deskew(p=0.85)` — applied 85% of the time during training.

**Why not always?** In production, the deskew algorithm may fail (circular objects, heavily occluded products). By occasionally presenting un-deskewed images, the model becomes robust to imperfect geometry correction at inference.

The deskew transform itself includes a **shape guard** that skips correction for non-rectangular objects (circles, ellipses). This means `p=1.0` at validation is safe — it will either correct rectangular objects or pass through non-rectangular ones unchanged.

### 3. Geometric Augmentation

| Transform | Real-world source | Parameters |
|-----------|------------------|------------|
| `Pad((25–75, 25–75), 255)` | Product not centered on belt; different package sizes | White fill (background in inverted space) |
| `Affine(scale, rotate, shear)` | Belt speed variation, slight tilt, product rotation | Conservative scale (0.88–0.97); full rotation |
| `RandomRotate90` | Products placed in arbitrary orientation on belt | 50% probability |

**Why full 360° rotation?** Many products (cans, jars, pouches) have no preferred orientation on a conveyor belt. The model must detect anomalies regardless of orientation. Products with a preferred orientation (e.g., cartons) are handled by the condition encoder — the model learns orientation-specific features per product.

### 4. Sensor/Acquisition Noise

Applied as a `OneOf` at `p=0.3` — low probability, one noise type at a time.

| Transform | Real-world source |
|-----------|-----------------|
| `GaussianBlur` | Slight defocus, scintillator spread |
| `MotionBlur` | Belt movement during exposure |
| `GaussNoise` | Electronic noise in detector |
| `CoarseDropout` (line artifacts) | Dead pixel rows in linear array detector |
| `ShotNoise` | Photon counting statistics (Poisson noise) |

**Why low probability?** These artifacts are rare in well-maintained systems. Over-applying them would make reconstruction unnecessarily difficult without proportional benefit for anomaly robustness.

**CoarseDropout configuration:** Simulates dead pixel lines in linear array detectors — horizontal streaks of constant value. Height 1–5px, full width, white fill (dead pixels in inverted space are saturated).

### 5. Intensity Drift

`RandomBrightnessContrast(±0.05, ±0.05, p=0.2)` — very subtle, applied after all physics preprocessing.

Simulates slow drift in X-ray source intensity (tube aging, voltage fluctuation). Applied AFTER log/gamma because the drift happens at the sensor level, before any digital processing. The small range (±5%) reflects that real systems have feedback loops maintaining intensity within tight bounds.

---

## Augmentation vs Fixed Preprocessing

A critical architectural distinction:

| Category | Train | Validation | Inference | Example |
|----------|-------|------------|-----------|----------|
| **Fixed preprocessing** | Applied deterministically | Applied deterministically | Applied deterministically | LogTransform, canonical GammaCorrection |
| **Varied preprocessing** | Parameters randomized | Canonical parameters | Canonical parameters | GammaCorrection with varying center/gain |
| **Random augmentation** | Applied stochastically | NOT applied | NOT applied | Rotation, affine, noise |

The key insight: **varying the preprocessing parameters** is different from random augmentation. It simulates real variation in the data acquisition pipeline, teaching the model that different-looking images can be "the same" in terms of anomaly content.

---

## Interaction with Anomaly Detection

### What Augmentation Should NOT Do

1. **Never augment the log transform itself** — it's a physical law, not a parameter
2. **Never add augmentations that mimic anomalies** — dark spots, foreign-object-shaped artifacts, local density changes
3. **Never normalize differently** at train vs inference — the model's learned "normal" distribution must match inference input

### What Augmentation SHOULD Do

1. **Absorb acquisition variation into "normal"** — so only true anomalies trigger detection
2. **Make reconstruction harder** — forcing the decoder to learn the product's structure, not memorize specific images
3. **Prevent shortcut learning** — if all training images are perfectly centered and deskewed, the model might learn to detect "not centered" as anomalous

### The Reconstruction Difficulty Trade-off

More augmentation → harder reconstruction → higher baseline `loss_rec` → need better decoder capacity.

But also: more augmentation → tighter normal manifold → sharper anomaly boundary.

The equilibrium point is where validation `loss_rec` continues decreasing (model learns the augmented distribution) while anomaly detection performance improves. If `loss_rec` plateaus at a high value, augmentation is too aggressive for the decoder's capacity.

---

## Deskew Shape Guard

The `Deskew` transform includes an automatic shape guard to prevent harmful corrections on non-rectangular objects:

| Criterion | Threshold | Purpose |
|-----------|-----------|--------|
| **Rectangularity** = ContourArea / minAreaRect_Area | ≥ 0.85 | Rejects circles (π/4 ≈ 0.785), ellipses, irregular shapes |
| **Aspect ratio** = long_side / short_side | ≥ 1.2 | Rejects near-square objects where rotation angle is ambiguous |

If either criterion fails, the image is returned unchanged. This prevents the deskew from rotating circular cans (no meaningful "straight" orientation) or square packages (90° ambiguity).
