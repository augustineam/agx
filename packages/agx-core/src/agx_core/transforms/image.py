import numpy as np

from albumentations import ImageOnlyTransform
from albumentations.augmentations.pixel import functional as F


class BrightnessAdjustment(ImageOnlyTransform):
    """Apply brightness adjustment."""

    def __init__(self, brightness=20, p=1.0):
        super().__init__(p=p)
        self._brightness = brightness

    def apply(self, image: np.ndarray, **params):
        image = F.adjust_brightness_torchvision(image, self._brightness)
        return image

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("brightness",)


class LogTransform(ImageOnlyTransform):
    """Linearize X-ray attenuation via log transform.

    Converts exponential Beer-Lambert intensity to linear attenuation space.
    After this transform, material density differences become additive/linear,
    making foreign objects equally detectable regardless of background density.

    For X-ray images where:
        pixel_value ∝ I₀ · exp(-μ·t)

    The log transform gives:
        output ∝ μ·t  (linear attenuation)

    This means a foreign object with attenuation Δμ produces a CONSTANT
    offset in log-space regardless of what's behind it — unlike raw pixel
    space where the same FO produces different contrast depending on
    background density.

    Args:
        epsilon: Small constant to avoid log(0). Default handles uint8 inputs.
        invert: If True, applies -log (standard for transmission X-ray where
            dense = dark in raw image). If False, applies log directly.
    """

    def __init__(self, epsilon: float = 1.0, invert: bool = False, p: float = 1.0):
        super().__init__(p=p)
        self.epsilon = epsilon
        self.invert = invert

    def apply(self, image: np.ndarray, **params):
        image = image.astype(np.float32)
        # Add epsilon to avoid log(0), then log transform
        log_image = np.log(image + self.epsilon)

        if self.invert:
            # For transmission X-ray: high pixel = low attenuation
            # Invert so that high attenuation (dense material) = high value
            log_image = np.log(255.0 + self.epsilon) - log_image

        # Rescale to [0, 255] for compatibility with downstream transforms
        log_min = log_image.min()
        log_max = log_image.max()
        if log_max - log_min > 1e-6:
            log_image = (log_image - log_min) / (log_max - log_min) * 255.0

        return log_image.astype(np.uint8)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("epsilon", "invert")


class GammaCorrection(ImageOnlyTransform):
    """Apply gamma correction optimized for X-ray density separation.

    For X-ray images where foreign objects and content occupy a similar
    intensity band, gamma correction can expand the contrast within that band.

    Unlike standard gamma (x^γ), this supports a "band-focused" mode that
    applies contrast expansion around a specified intensity center point,
    useful when both content and anomalies sit in the 0.5–0.9 range.

    Args:
        gamma: Gamma value. >1 darkens (expands highlights), <1 brightens.
            For standard X-ray enhancement: 0.6–0.8 typical.
        center: If provided, applies sigmoidal contrast around this intensity
            level (in [0, 1] range). None = standard power-law gamma.
        gain: Sigmoidal gain (steepness). Higher = sharper contrast at center.
    """

    def __init__(
        self,
        gamma: float = 0.7,
        center: float | None = None,
        gain: float = 5.0,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.gamma = gamma
        self.center = center
        self.gain = gain

    def apply(self, image: np.ndarray, **params):
        image = image.astype(np.float32) / 255.0

        if self.center is not None:
            # Sigmoidal contrast: expands contrast around 'center'
            # S(x) = 1 / (1 + exp(-gain * (x - center)))
            # Normalized to map [0,1] → [0,1]
            image = self._sigmoidal_contrast(image)
        else:
            # Standard power-law gamma
            image = np.power(np.clip(image, 0, 1), self.gamma)

        return (image * 255.0).astype(np.uint8)

    def _sigmoidal_contrast(self, image):
        """Apply sigmoidal contrast centered at self.center."""
        # Sigmoid function
        sig = lambda x: 1.0 / (1.0 + np.exp(-self.gain * (x - self.center)))

        # Normalize so that [0, 1] maps to [0, 1]
        sig_0 = sig(0.0)
        sig_1 = sig(1.0)

        result = (sig(image) - sig_0) / (sig_1 - sig_0)
        return np.clip(result, 0, 1)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("gamma", "center", "gain")


__all__ = ["BrightnessAdjustment", "LogTransform", "GammaCorrection"]
