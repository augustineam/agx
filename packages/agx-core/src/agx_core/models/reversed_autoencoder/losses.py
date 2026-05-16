from __future__ import annotations

from typing import Sequence

from keras import KerasTensor, ops, losses

from agx_core.helpers import _channel_axis


def ssim_loss(
    y_true: KerasTensor,
    y_pred: KerasTensor,
    kernel: KerasTensor,
    max_val: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> KerasTensor:
    """
    Structural Similarity Index (SSIM) loss via `ops`.

    Computes `1 - SSIM(y_true, y_pred)` so that minimizing the loss
    maximizes structural similarity.

    Args:
        y_true: Ground-truth images, shape `(B, H, W, C)`.
        y_pred: Predicted images, shape `(B, H, W, C)`.
        max_val: The dynamic range of the images (1.0 for normalized, 255.0 for uint8).
        filter_size: Side length of the Gaussian kernel.
        filter_sigma: Standard deviation of the Gaussian kernel.
        k1: Stability constant for the luminance term.
        k2: Stability constant for the contrast/structure term.

    Returns:
        Scalar loss value: `1 - mean(SSIM)`.
    """
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    def _apply_filter(x: KerasTensor) -> KerasTensor:
        return ops.depthwise_conv(
            x, kernel, strides=(1, 1), padding="same", dilation_rate=(1, 1)
        )

    mu_x = _apply_filter(y_true)
    mu_y = _apply_filter(y_pred)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = _apply_filter(y_true * y_true) - mu_x_sq
    sigma_y_sq = _apply_filter(y_pred * y_pred) - mu_y_sq
    sigma_xy = _apply_filter(y_true * y_pred) - mu_xy

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

    ssim_map = numerator / denominator
    return 1.0 - ops.mean(ssim_map, axis=_channel_axis())


def kl_divergence(mean, logvar, cap=10.0):
    """Closed-form KLD from N(mean, exp(logvar)) to N(0, I), per spatial position.

    With a spatial latent (B, H, W, C), this returns a (B, H, W) map where
    each position measures how "unusual" the encoding is at that location.

    KLD = 0.5 * Σ_c (μ² + exp(logvar) - logvar - 1) / C

    This replaces the Monte Carlo estimate (log q(z|x) - log p(z)) which
    requires sampling z and has non-zero variance. The closed form is
    exact, lower variance, and cheaper to compute.

    Returns:
        KLD map of shape (B, H, W) — mean over channels, spatial preserved.
    """
    kld = 0.5 * ops.mean(
        ops.square(mean) + ops.exp(logvar) - logvar - 1.0,
        axis=_channel_axis(),
    )
    return ops.minimum(kld, cap)


def embedding_loss(
    teacher_embedds: Sequence[KerasTensor],
    student_embedds: Sequence[KerasTensor],
):
    """Feature consistency loss (MSE + Cosine Similarity) between two sets of embeddings."""
    total_loss = 0
    scale = 1.0 / len(teacher_embedds)
    for teacher_feature, student_feature in zip(teacher_embedds, student_embedds):
        mse = ops.mean(
            ops.square(ops.stop_gradient(teacher_feature) - student_feature),
            axis=_channel_axis(),
        )
        # [B, H, W]
        cosine_similarity = losses.cosine_similarity(
            ops.stop_gradient(teacher_feature), student_feature, axis=_channel_axis()
        )
        total_loss += ops.mean(0.5 * mse + (1 + cosine_similarity), axis=[1, 2])
    return scale * total_loss


def mse_weighted(y_true, y_pred, spatial_temperature: float = 0.0):
    """Per-pixel MSE with optional spatial curriculum weighting.

    When spatial_temperature > 0, hard-to-reconstruct regions receive
    exponentially more gradient attention. Critical for information-sparse
    images (single-channel X-rays) where uniform background dominates
    the spatial mean.

    Args:
        y_true: Ground truth [B, H, W, C]
        y_pred: Prediction [B, H, W, C]
        spatial_temperature: Curriculum sharpness. 0 = uniform (original behavior).
            Recommended range for X-rays: 2.0-10.0

    Returns:
        Error map [B, H, W]. When temperature > 0, the map is spatially
        reweighted so that hard regions contribute more to the downstream mean.
    """
    error = ops.mean(ops.square(y_true - y_pred), axis=_channel_axis())  # [B, H, W]

    if spatial_temperature <= 0.0:
        return error

    # Exponential weighting: hard pixels (high error) get more attention
    # stop_gradient prevents second-order effects (don't optimize weights themselves)
    weights = ops.exp(spatial_temperature * ops.stop_gradient(error))
    # Normalize per-sample so overall loss magnitude is preserved
    weights = weights / ops.mean(weights, axis=[1, 2], keepdims=True)

    return weights * error


__all__ = ["mse_weighted", "embedding_loss", "kl_divergence", "ssim_loss"]
