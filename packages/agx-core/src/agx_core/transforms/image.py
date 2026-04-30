import numpy as np

from albumentations import ImageOnlyTransform
from albumentations.augmentations.pixel import functional as F


class BrightnessAndContrast(ImageOnlyTransform):
    """Apply brightness and contrast."""

    def __init__(self, contrast=0.93, brightness=20, p=1.0):
        super().__init__(p=p)
        self._contrast = contrast
        self._brightness = brightness

    def apply(self, image: np.ndarray, **params):
        image = F.adjust_contrast_torchvision(image, self._contrast)
        image = F.adjust_brightness_torchvision(image, self._brightness)
        return image

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "contrast",
            "brightness",
        )


__all__ = ["BrightnessAndContrast"]
