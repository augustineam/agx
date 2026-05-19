"""
Deskew & Deshear transform for X-ray images of rectangular objects.

Corrects both rotation (skew) and parallelogram distortion (shear) caused by
conveyor-speed / camera-timing mismatch.  Works entirely with OpenCV + NumPy
and exposes an Albumentations ``ImageOnlyTransform`` for pipeline integration.

Strategy
--------
Instead of the fragile  Hough → cluster → median-angle → shear-matrix  pipeline,
we use a geometrically principled 4-point perspective correction:

1. Threshold the image to obtain a binary mask of the object.
2. Find the largest external contour.
3. **Guard:** Check rectangularity and aspect ratio. Non-rectangular objects
   (circles, ellipses, irregular shapes) are returned unchanged — deskew
   only makes sense for objects that *should be* rectangular.
4. Approximate the contour to a quadrilateral (``approxPolyDP``).
5. Compute ``minAreaRect`` on the contour to get the ideal (unsheared,
   unrotated) rectangle.
6. Build a perspective warp that maps the detected quad corners → the
   ``minAreaRect`` corners, correcting **both** skew and shear in a single
   ``warpPerspective`` call.

If the contour cannot be reduced to exactly 4 vertices (heavy noise, partial
occlusion, etc.) we fall back to a pure-rotation correction derived from the
``minAreaRect`` angle alone.
"""

from __future__ import annotations

import cv2
import logging
import numpy as np

from typing import Optional

from albumentations import ImageOnlyTransform

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Pure-function geometry helpers (usable without Albumentations)
# ---------------------------------------------------------------------------


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalise any single-channel image to ``uint8``."""
    if img.dtype == np.uint8:
        return img
    # handles uint16, float32/64, etc.
    lo, hi = img.min(), img.max()
    if hi == lo:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - lo) / (hi - lo) * 255).astype(np.uint8)


def _order_quadrilateral(pts: np.ndarray) -> np.ndarray:
    """
    Order four 2-D points as [top-left, top-right, bottom-right, bottom-left].

    Uses the sum (x+y) to find TL/BR and the difference (y-x) to find TR/BL,
    which is the standard "document-scanner" ordering trick.
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)  # x + y
    d = np.diff(pts, axis=1)  # y - x

    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    ordered[1] = pts[np.argmin(d)]  # top-right
    ordered[3] = pts[np.argmax(d)]  # bottom-left
    return ordered


def _order_box_points(box: np.ndarray) -> np.ndarray:
    """Order ``cv2.boxPoints`` output the same way as ``_order_quadrilateral``."""
    return _order_quadrilateral(box)


def _find_object_quad(
    mask: np.ndarray,
    min_area_frac: float = 0.01,
    approx_eps_frac: float = 0.02,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[tuple]]:
    """
    From a binary mask, find the largest contour and try to approximate it
    as a quadrilateral.

    Returns
    -------
    quad : (4, 2) float32 array or ``None``
        Ordered quadrilateral corners, or ``None`` if no quad could be fit.
    contour : np.ndarray or ``None``
        The raw largest contour (for fallback ``minAreaRect``).
    min_rect : ((cx, cy), (w, h), angle) or ``None``
        The ``minAreaRect`` of the largest contour.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    img_area = mask.shape[0] * mask.shape[1]

    if area < min_area_frac * img_area:
        logger.debug(
            "Largest contour too small (%.1f%% of image)", 100 * area / img_area
        )
        return None, None, None

    min_rect = cv2.minAreaRect(largest)

    # Try to get a 4-vertex approximation
    peri = cv2.arcLength(largest, closed=True)
    approx = cv2.approxPolyDP(largest, approx_eps_frac * peri, closed=True)

    if len(approx) == 4:
        quad = _order_quadrilateral(approx.reshape(4, 2))
        return quad, largest, min_rect

    # Adaptive: widen epsilon until we get 4 (or give up)
    for factor in (0.03, 0.04, 0.05, 0.07, 0.10):
        approx = cv2.approxPolyDP(largest, factor * peri, closed=True)
        if len(approx) == 4:
            quad = _order_quadrilateral(approx.reshape(4, 2))
            return quad, largest, min_rect

    # Could not reduce to 4 vertices → return None quad, caller uses minAreaRect
    return None, largest, min_rect


def _is_rectangular(
    contour: np.ndarray,
    min_rect: tuple,
    rectangularity_threshold: float = 0.85,
    min_aspect_ratio: float = 1.2,
) -> tuple[bool, str]:
    """Determine if the detected object is rectangular enough to deskew.

    Uses two criteria:

    1. **Rectangularity** = ContourArea / minAreaRect_Area.
       Perfect rectangle ≈ 1.0, ellipse/circle ≈ π/4 ≈ 0.785.
       Only objects above the threshold are considered rectangular.

    2. **Aspect ratio** = long_side / short_side of the minAreaRect.
       Near-square objects (AR < threshold) have ambiguous orientation
       and should not be rotated — the minAreaRect angle is unreliable.

    Returns
    -------
    is_rect : bool
        Whether the object should be corrected.
    skip_reason : str
        Empty if ``is_rect`` is True; otherwise a short diagnostic string.
    """
    contour_area = cv2.contourArea(contour)
    (_, _), (rect_w, rect_h), _ = min_rect
    rect_area = rect_w * rect_h

    if rect_area == 0:
        return False, "zero_area"

    rectangularity = contour_area / rect_area
    if rectangularity < rectangularity_threshold:
        return False, f"not_rectangular (rect={rectangularity:.3f})"

    aspect_ratio = max(rect_w, rect_h) / (min(rect_w, rect_h) + 1e-6)
    if aspect_ratio < min_aspect_ratio:
        return False, f"ambiguous_orientation (AR={aspect_ratio:.2f})"

    return True, ""


def _background_value(img: np.ndarray, mask: np.ndarray) -> int:
    """Median intensity of background pixels (where mask == 0)."""
    bg = img[mask == 0]
    if bg.size == 0:
        return int(np.max(img))
    return int(np.median(bg))


# ---------------------------------------------------------------------------
#  Core correction functions
# ---------------------------------------------------------------------------


def compute_deskew_perspective(
    img: np.ndarray,
    *,
    threshold_method: str = "otsu",
    threshold_value: int = 128,
    blur_ksize: int = 5,
    morph_ksize: int = 5,
    min_area_frac: float = 0.01,
    approx_eps_frac: float = 0.02,
    rectangularity_threshold: float = 0.85,
    min_aspect_ratio: float = 1.2,
    border_value: Optional[int] = None,
) -> tuple[np.ndarray, dict]:
    """
    Detect and correct skew **and** shear on a single grayscale X-ray image.

    Only applies correction to objects that are sufficiently rectangular
    (rectangularity > threshold) and elongated (aspect ratio > threshold).
    Circular, elliptical, and near-square objects are returned unchanged.

    Parameters
    ----------
    img : 2-D ndarray
        Grayscale image (any bit-depth).
    threshold_method : ``"otsu"`` | ``"fixed"``
        Binarisation strategy.
    threshold_value : int
        Fixed threshold (ignored when *otsu*).
    blur_ksize : int
        Gaussian blur kernel size (set ≤ 1 to skip).
    morph_ksize : int
        Morphological-close kernel size (set ≤ 1 to skip).
    min_area_frac : float
        Reject contours smaller than this fraction of the image area.
    approx_eps_frac : float
        Initial ``approxPolyDP`` epsilon as a fraction of arc-length.
    rectangularity_threshold : float
        Minimum ContourArea/minAreaRect_Area ratio to consider an object
        rectangular. Below this the object is assumed curved (circle,
        ellipse, etc.) and returned unchanged. Default 0.85.
    min_aspect_ratio : float
        Minimum long_side/short_side ratio of the minAreaRect. Objects
        below this are near-square with ambiguous orientation and are
        returned unchanged. Default 1.2.
    border_value : int or None
        Fill value for new pixels; ``None`` → auto from background.

    Returns
    -------
    corrected : ndarray
        Corrected image (same dtype as input).
    info : dict
        Diagnostic dict with keys ``angle_deg``, ``method``
        (``"perspective"`` | ``"rotation"`` | ``"none"`` | ``"skip_*"``),
        and ``quad``.
    """
    # --- binarise ---------------------------------------------------------
    u8 = _to_uint8(img)

    if blur_ksize > 1:
        ks = blur_ksize | 1  # ensure odd
        u8 = cv2.GaussianBlur(u8, (ks, ks), 0)

    if threshold_method == "otsu":
        _, mask = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(u8, threshold_value, 255, cv2.THRESH_BINARY_INV)

    if morph_ksize > 1:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)

    # --- detect geometry --------------------------------------------------
    quad, contour, min_rect = _find_object_quad(
        mask,
        min_area_frac=min_area_frac,
        approx_eps_frac=approx_eps_frac,
    )

    if min_rect is None:
        logger.info("No object found — returning image unchanged")
        return img.copy(), {"angle_deg": 0.0, "method": "none", "quad": None}

    # --- shape guard: only correct rectangular, elongated objects ----------
    if contour is not None:
        is_rect, skip_reason = _is_rectangular(
            contour, min_rect, rectangularity_threshold, min_aspect_ratio
        )
        if not is_rect:
            logger.debug("Skipping deskew: %s", skip_reason)
            return img.copy(), {
                "angle_deg": 0.0,
                "method": f"skip_{skip_reason}",
                "quad": None,
            }

    fill = border_value if border_value is not None else _background_value(img, mask)
    (_, _), (rect_w, rect_h), angle = min_rect

    # --- perspective path (handles skew + shear) --------------------------
    if quad is not None:
        # Destination: axis-aligned rectangle matching the minAreaRect dims.
        # Ensure width > height so the long side stays horizontal.
        if rect_w < rect_h:
            rect_w, rect_h = rect_h, rect_w

        dst_pts = np.array(
            [
                [0, 0],
                [rect_w, 0],
                [rect_w, rect_h],
                [0, rect_h],
            ],
            dtype=np.float32,
        )

        M = cv2.getPerspectiveTransform(quad, dst_pts)
        out_w, out_h = int(np.ceil(rect_w)), int(np.ceil(rect_h))
        corrected = cv2.warpPerspective(
            img,
            M,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderValue=fill,
        )
        logger.debug(
            "Perspective correction — quad detected, rect %.0f×%.0f, angle %.1f°",
            rect_w,
            rect_h,
            angle,
        )
        if corrected.ndim == 2:
            corrected = corrected[..., np.newaxis]

        return corrected, {
            "angle_deg": angle,
            "method": "perspective",
            "quad": quad,
        }

    # --- fallback: pure rotation from minAreaRect -------------------------
    # OpenCV minAreaRect angle convention: the angle is between the
    # width-edge of the RotatedRect and the horizontal axis.
    # Normalise to [-45, 45] rotation.
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90

    if abs(angle) < 0.3:  # negligible
        return img.copy(), {"angle_deg": angle, "method": "none", "quad": None}

    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # expand canvas so nothing is cropped
    cos_a = np.abs(M[0, 0])
    sin_a = np.abs(M[0, 1])
    new_w = int(np.ceil(h * sin_a + w * cos_a))
    new_h = int(np.ceil(h * cos_a + w * sin_a))
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0

    corrected = cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderValue=fill,
    )
    logger.debug("Rotation-only correction — angle %.1f°", angle)
    return corrected, {
        "angle_deg": angle,
        "method": "rotation",
        "quad": None,
    }


# ---------------------------------------------------------------------------
#  Albumentations transform
# ---------------------------------------------------------------------------


class Deskew(ImageOnlyTransform):
    """
    Albumentations transform that straightens rectangular objects in X-ray
    images, correcting both rotation (skew) and parallelogram distortion
    (shear) in a single perspective warp.

    Includes a shape guard that skips correction for non-rectangular objects
    (circles, ellipses, near-square shapes) where deskew is meaningless or
    harmful.

    Parameters
    ----------
    threshold_method : ``"otsu"`` | ``"fixed"``
    threshold_value : int
    blur_ksize : int
    morph_ksize : int
    min_area_frac : float
    approx_eps_frac : float
    rectangularity_threshold : float
        Minimum ContourArea/minAreaRect_Area to consider the object
        rectangular. Default 0.85. Ellipses/circles score ~0.785.
    min_aspect_ratio : float
        Minimum long/short side ratio. Near-square objects have ambiguous
        orientation and are skipped. Default 1.2.
    border_value : int or None
    p : float
        Probability of applying the transform (default ``1.0`` — this is a
        deterministic correction, not a random augmentation).

    Example
    -------
    >>> import albumentations as A
    >>> from agx_core.transforms.deskew import Deskew
    >>> pipe = A.Compose([Deskew(p=1.0)])
    >>> result = pipe(image=my_xray)["image"]
    """

    def __init__(
        self,
        threshold_method: str = "otsu",
        threshold_value: int = 128,
        blur_ksize: int = 5,
        morph_ksize: int = 5,
        min_area_frac: float = 0.01,
        approx_eps_frac: float = 0.02,
        rectangularity_threshold: float = 0.85,
        min_aspect_ratio: float = 1.2,
        border_value: Optional[int] = None,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        self.blur_ksize = blur_ksize
        self.morph_ksize = morph_ksize
        self.min_area_frac = min_area_frac
        self.approx_eps_frac = approx_eps_frac
        self.rectangularity_threshold = rectangularity_threshold
        self.min_aspect_ratio = min_aspect_ratio
        self.border_value = border_value

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        corrected, _info = compute_deskew_perspective(
            img,
            threshold_method=self.threshold_method,
            threshold_value=self.threshold_value,
            blur_ksize=self.blur_ksize,
            morph_ksize=self.morph_ksize,
            min_area_frac=self.min_area_frac,
            approx_eps_frac=self.approx_eps_frac,
            rectangularity_threshold=self.rectangularity_threshold,
            min_aspect_ratio=self.min_aspect_ratio,
            border_value=self.border_value,
        )
        return corrected

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "threshold_method",
            "threshold_value",
            "blur_ksize",
            "morph_ksize",
            "min_area_frac",
            "approx_eps_frac",
            "rectangularity_threshold",
            "min_aspect_ratio",
            "border_value",
        )


__all__ = ["Deskew", "compute_deskew_perspective"]
