"""Post-processing anomaly detection from reconstruction-based models.

This module operates on numpy arrays (CPU) and is intended to run
AFTER the neural network forward pass. It is NOT part of the ONNX graph.

Pipeline:
    1. Compute raw anomaly score map from model outputs
    2. Calibrate to per-pixel probability via sigmoid
    3. Threshold to binary mask
    4. Connected components extraction
    5. Cluster scoring with confidence/area weighting

Typical usage in deployment (C++ equivalent exists):

    scores = compute_anomaly_scores(image, reconstruction, kld_map)
    prob_map = calibrate_scores(scores, temperature, bias)
    detections = detect_anomalies(prob_map, ...)
"""

from __future__ import annotations

import numpy as np
import cv2

from dataclasses import dataclass, field
from typing import Sequence, Optional


@dataclass
class AnomalyDetection:
    """A single detected anomalous region.

    Attributes:
        bbox: Bounding box as (x, y, w, h) in pixel coordinates.
        centroid: Center of mass as (x, y).
        area_px: Number of pixels in the cluster.
        mean_probability: Mean calibrated anomaly probability within the cluster.
        max_probability: Peak anomaly probability in the cluster.
        score: Final cluster score combining confidence and area.
        mask: Binary mask of the cluster at full image resolution.
            Only populated if `return_masks=True` in `detect_anomalies`.
    """

    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area_px: int
    mean_probability: float
    max_probability: float
    score: float
    mask: Optional[np.ndarray] = field(default=None, repr=False)


def compute_anomaly_scores(
    image: np.ndarray,
    reconstruction: np.ndarray,
    kld_map: Optional[np.ndarray] = None,
    feature_map: Optional[np.ndarray] = None,
    weights: tuple[float, float, float] = (1.0, 0.5, 0.3),
) -> np.ndarray:
    """Compute raw anomaly score map from model outputs.

    All inputs should be numpy arrays with shape (H, W) or (H, W, C).
    If spatial resolutions differ (e.g., kld_map is 7×7), they are
    upsampled to match the image resolution via bilinear interpolation.

    Args:
        image: Original input image, shape (H, W) or (H, W, C).
        reconstruction: Decoder output, same shape as image.
        kld_map: Optional KLD map from encoder, shape (h, w). May be
            lower resolution than image.
        feature_map: Optional feature magnitude map, shape (h, w).
        weights: (w_pixel, w_kld, w_feature) combination weights.

    Returns:
        Score map of shape (H, W), unnormalized (higher = more anomalous).
    """
    # Ensure 2D for pixel error
    if image.ndim == 3:
        pixel_error = np.mean((image - reconstruction) ** 2, axis=-1)
    else:
        pixel_error = (image - reconstruction) ** 2

    h, w = pixel_error.shape[:2]
    score = weights[0] * _normalize_map(pixel_error)

    if kld_map is not None and weights[1] > 0:
        kld_resized = _resize_to(kld_map, h, w)
        score += weights[1] * _normalize_map(kld_resized)

    if feature_map is not None and weights[2] > 0:
        feat_resized = _resize_to(feature_map, h, w)
        score += weights[2] * _normalize_map(feat_resized)

    return score


def calibrate_scores(
    score_map: np.ndarray,
    temperature: float = 5.0,
    bias: float = -3.0,
) -> np.ndarray:
    """Convert raw scores to calibrated per-pixel anomaly probabilities.

    Applies: P(anomaly | score) = sigmoid(temperature * normalized_score + bias)

    The temperature controls sharpness (higher = more binary output).
    The bias shifts the decision boundary (more negative = fewer positives).

    These should be calibrated on a validation set with known anomalies.
    A reasonable starting point for X-ray inspection:
        - temperature=5.0: moderate sharpness
        - bias=-3.0: conservative (low false positive rate)

    Args:
        score_map: Raw anomaly scores, shape (H, W).
        temperature: Sigmoid steepness. Higher = sharper transition.
        bias: Shifts the midpoint. Negative = more conservative.

    Returns:
        Probability map in [0, 1], shape (H, W).
    """
    normalized = _normalize_map(score_map)
    logits = temperature * normalized + bias
    return _sigmoid(logits)


def detect_anomalies(
    probability_map: np.ndarray,
    pixel_threshold: float = 0.5,
    min_area_px: int = 4,
    max_area_px: int = 50000,
    confidence_exponent: float = 1.5,
    area_exponent: float = 0.3,
    score_threshold: float = 0.0,
    return_masks: bool = False,
) -> list[AnomalyDetection]:
    """Extract and score anomalous clusters from a probability map.

    Pipeline:
        1. Threshold probability map to binary mask
        2. Morphological cleanup (close small gaps, remove noise)
        3. Connected components extraction
        4. Per-cluster scoring: score = mean_prob^α * area^β
        5. Filter by score threshold, area bounds

    The scoring function rewards:
        - Small clusters with high confidence (foreign objects)
        - Penalizes large diffuse regions (reconstruction artifacts)

    Adjusting exponents:
        - confidence_exponent > 1: strongly rewards high-confidence clusters
        - area_exponent < 1: sublinear area scaling (small hot > large warm)
        - area_exponent = 0: pure confidence, area ignored in score
        - area_exponent < 0: actively penalizes large regions

    Args:
        probability_map: Calibrated anomaly probabilities in [0, 1], shape (H, W).
        pixel_threshold: Probability above which a pixel is "anomalous".
        min_area_px: Minimum cluster size to consider (removes noise).
        max_area_px: Maximum cluster size to consider (removes background).
        confidence_exponent: α in score = mean_prob^α * area^β.
        area_exponent: β in score = mean_prob^α * area^β.
        score_threshold: Minimum cluster score to include in results.
        return_masks: If True, populate each detection's .mask field.

    Returns:
        List of AnomalyDetection, sorted by score descending.
    """
    # 1. Binary threshold
    binary = (probability_map >= pixel_threshold).astype(np.uint8)

    # 2. Morphological cleanup
    #    Close: fill small gaps within anomalous regions
    #    Open: remove isolated noise pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 3. Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    # 4. Score each cluster
    detections: list[AnomalyDetection] = []

    for label_id in range(1, num_labels):  # skip background (0)
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area < min_area_px or area > max_area_px:
            continue

        # Extract cluster mask
        cluster_mask = labels == label_id

        # Compute confidence metrics from probability map
        cluster_probs = probability_map[cluster_mask]
        mean_prob = float(np.mean(cluster_probs))
        max_prob = float(np.max(cluster_probs))

        # Score: high confidence + small area = high score
        score = (mean_prob ** confidence_exponent) * (area ** area_exponent)

        if score < score_threshold:
            continue

        # Bounding box
        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]

        detection = AnomalyDetection(
            bbox=(x, y, w, h),
            centroid=(float(centroids[label_id, 0]), float(centroids[label_id, 1])),
            area_px=int(area),
            mean_probability=mean_prob,
            max_probability=max_prob,
            score=score,
            mask=cluster_mask.astype(np.uint8) if return_masks else None,
        )
        detections.append(detection)

    # Sort by score descending (most anomalous first)
    detections.sort(key=lambda d: d.score, reverse=True)
    return detections


def anomaly_pipeline(
    image: np.ndarray,
    reconstruction: np.ndarray,
    kld_map: Optional[np.ndarray] = None,
    feature_map: Optional[np.ndarray] = None,
    score_weights: tuple[float, float, float] = (1.0, 0.5, 0.3),
    temperature: float = 5.0,
    bias: float = -3.0,
    pixel_threshold: float = 0.5,
    min_area_px: int = 4,
    max_area_px: int = 50000,
    confidence_exponent: float = 1.5,
    area_exponent: float = 0.3,
    score_threshold: float = 0.0,
    return_masks: bool = False,
    return_maps: bool = False,
) -> dict:
    """Full anomaly detection pipeline: score → calibrate → detect.

    Convenience function combining all steps. For production deployment,
    you may want to call steps individually for more control.

    Args:
        image: Input image (H, W) or (H, W, C), float32 normalized [0, 1].
        reconstruction: Model reconstruction, same shape as image.
        kld_map: Optional encoder KLD map (may be lower resolution).
        feature_map: Optional feature magnitude map.
        score_weights: Weights for (pixel_error, kld, feature) combination.
        temperature: Sigmoid calibration temperature.
        bias: Sigmoid calibration bias.
        pixel_threshold: Per-pixel anomaly threshold on probability.
        min_area_px: Minimum cluster area.
        max_area_px: Maximum cluster area.
        confidence_exponent: α for cluster scoring.
        area_exponent: β for cluster scoring.
        score_threshold: Minimum cluster score to report.
        return_masks: Attach per-detection binary masks.
        return_maps: Include intermediate maps in output dict.

    Returns:
        Dictionary with:
            - "detections": List[AnomalyDetection]
            - "is_anomalous": bool (any detections found)
            - "max_score": float (highest cluster score, 0.0 if none)
            - "score_map": (optional) raw scores if return_maps=True
            - "probability_map": (optional) calibrated probs if return_maps=True
    """
    score_map = compute_anomaly_scores(
        image, reconstruction, kld_map, feature_map, score_weights
    )

    probability_map = calibrate_scores(score_map, temperature, bias)

    detections = detect_anomalies(
        probability_map,
        pixel_threshold=pixel_threshold,
        min_area_px=min_area_px,
        max_area_px=max_area_px,
        confidence_exponent=confidence_exponent,
        area_exponent=area_exponent,
        score_threshold=score_threshold,
        return_masks=return_masks,
    )

    result = {
        "detections": detections,
        "is_anomalous": len(detections) > 0,
        "max_score": detections[0].score if detections else 0.0,
    }

    if return_maps:
        result["score_map"] = score_map
        result["probability_map"] = probability_map

    return result


# ─── Internal Utilities ───────────────────────────────────────────────


def _normalize_map(m: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] per-sample. Handles constant maps gracefully."""
    min_val = m.min()
    max_val = m.max()
    range_val = max_val - min_val
    if range_val < 1e-8:
        return np.zeros_like(m)
    return (m - min_val) / range_val


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def _resize_to(m: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize a 2D map to target resolution via bilinear interpolation."""
    if m.shape[0] == target_h and m.shape[1] == target_w:
        return m
    return cv2.resize(
        m.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR
    )


__all__ = [
    "AnomalyDetection",
    "compute_anomaly_scores",
    "calibrate_scores",
    "detect_anomalies",
    "anomaly_pipeline",
]