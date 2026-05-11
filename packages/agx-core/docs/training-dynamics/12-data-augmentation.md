# Data Augmentation Rationale

Robust augmentations (rotation, illumination, contrast) are applied despite being unlikely in production. The reasoning:

1. **Inter-product variation**: X-rays from one product to the next can be very different. Augmentations teach the model to be invariant to acquisition nuances.

2. **Acquisition artifacts**: Slight variations in image capture (positioning, exposure, voltage) should NOT trigger anomaly detection. Augmentations ensure these variations are absorbed into the "normal" manifold.

3. **Focus on true anomalies**: By making the model robust to expected visual variation, the anomaly signal becomes specific to genuine defects rather than imaging conditions.

The trade-off is that aggressive augmentations make reconstruction harder (the decoder must reproduce augmented versions, not just canonical views), which is why decoder architecture capacity matters.

### Augmentation vs Fixed Preprocessing

A key distinction:

- **Fixed preprocessing** (log transform, brightness): Always applied identically at train and inference. Maximizes signal quality. The model always sees the same representation.
- **Random augmentation** (rotation, affine, blur): Applied stochastically during training only. Builds invariance. The model learns to handle variation.

For anomaly detection specifically: fixed preprocessing that maximizes anomaly contrast is always beneficial. Random augmentation of contrast/brightness is beneficial for robustness but should NOT be applied to the log transform or normalization steps.
