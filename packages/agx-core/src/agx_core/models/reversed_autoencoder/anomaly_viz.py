"""Interactive anomaly detection visualization with ipywidgets.

Allows real-time tuning of anomaly pipeline parameters on model outputs.
Designed for Jupyter notebook usage during development/calibration.

Usage:
    from agx_core.models.reversed_autoencoder.anomaly_viz import AnomalyExplorer

    explorer = AnomalyExplorer(model, dataloader, device="cuda")
    explorer.show()
"""

from __future__ import annotations

import numpy as np
import cv2
import matplotlib.pyplot as plt

from typing import Optional, Any
from dataclasses import dataclass

import ipywidgets as widgets
from IPython.display import display


@dataclass
class CachedInference:
    """Pre-computed model outputs for interactive exploration."""

    images: np.ndarray  # (N, H, W) or (N, H, W, C), normalized [0,1]
    reconstructions: np.ndarray  # same shape as images
    kld_maps: np.ndarray  # (N, h, w) — latent resolution
    feature_maps: Optional[np.ndarray] = None  # (N, h, w) — optional


class AnomalyExplorer:
    """Interactive widget for tuning anomaly detection parameters.

    Pre-computes model inference once, then allows real-time parameter
    adjustment via sliders without re-running the model.

    Args:
        model: ReversedAutoencoder model (eval mode).
        dataloader: DataLoader yielding ((images, labels), _) batches.
        n_samples: Number of samples to cache for exploration.
        device: Torch device string.
        use_feature_maps: If True, run a second encoder pass on
            reconstructions to get feature difference maps.
    """

    def __init__(
        self,
        model: Any,
        dataloader: Any,
        n_samples: int = 16,
        device: str = "cuda",
        use_feature_maps: bool = False,
    ):
        self.n_samples = n_samples
        self.use_feature_maps = use_feature_maps
        self._cache = self._run_inference(model, dataloader, device)
        self._widgets = self._build_widgets()
        self._output = widgets.Output()

    def _run_inference(
        self, model: Any, dataloader: Any, device: str
    ) -> CachedInference:
        """Run model inference once and cache all outputs as numpy."""
        import torch
        from keras import ops
        from agx_core.models.reversed_autoencoder.model import kl_divergence

        model.eval()
        (samples, labels), _ = next(iter(dataloader))

        n = min(self.n_samples, samples.shape[0])

        with torch.no_grad():
            samples_resized = model.resize_progressive_output(samples)

            (mean, logvar), embeds = model.encoder(
                [samples_resized, labels], training=False
            )
            z = mean  # deterministic inference
            reconstructions = model.decoder([z, labels], training=False)

            kld_maps = kl_divergence(mean, logvar)

            # Optional: feature maps from re-encoding reconstructions
            feature_maps = None
            if self.use_feature_maps:
                _, embeds_rec = model.encoder([reconstructions, labels], training=False)
                print([e.shape for e in embeds_rec])
                # Multi-scale feature difference — use last (coarsest) embedding
                feat_real = embeds[-1]
                feat_rec = embeds_rec[-1]

                # Use channel axis based on format
                import keras

                if keras.config.image_data_format() == "channels_first":
                    feature_maps = ops.mean(ops.square(feat_real - feat_rec), axis=1)
                else:
                    feature_maps = ops.mean(ops.square(feat_real - feat_rec), axis=-1)

        # Convert to numpy, limit to n_samples
        images_np = samples_resized[:n].cpu().numpy()
        recon_np = reconstructions[:n].cpu().numpy()
        kld_np = kld_maps[:n].cpu().numpy()

        # Denormalize from [-1, 1] to [0, 1]
        images_np = 0.5 * images_np + 0.5
        recon_np = 0.5 * recon_np + 0.5

        # Handle channel dimension — squeeze if single channel
        if images_np.ndim == 4 and images_np.shape[-1] == 1:
            images_np = images_np[..., 0]
            recon_np = recon_np[..., 0]
        elif images_np.ndim == 4 and images_np.shape[1] == 1:
            images_np = images_np[:, 0]
            recon_np = recon_np[:, 0]

        feat_np = None
        if feature_maps is not None:
            feat_np = feature_maps[:n].cpu().numpy()

        return CachedInference(
            images=images_np,
            reconstructions=recon_np,
            kld_maps=kld_np,
            feature_maps=feat_np,
        )

    def _build_widgets(self) -> dict[str, widgets.Widget]:
        """Build all parameter sliders."""
        w = {}

        # Sample selector
        w["sample_idx"] = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self._cache.images) - 1,
            step=1,
            description="Sample:",
            continuous_update=False,
        )

        # Score weights
        w["w_pixel"] = widgets.FloatSlider(
            value=1.0,
            min=0.0,
            max=3.0,
            step=0.05,
            description="W pixel:",
            continuous_update=False,
        )
        w["w_kld"] = widgets.FloatSlider(
            value=0.5,
            min=0.0,
            max=3.0,
            step=0.05,
            description="W KLD:",
            continuous_update=False,
        )
        w["w_feature"] = widgets.FloatSlider(
            value=0.3 if self._cache.feature_maps is not None else 0.0,
            min=0.0,
            max=3.0,
            step=0.05,
            description="W feature:",
            continuous_update=False,
            disabled=self._cache.feature_maps is None,
        )

        # Calibration
        w["temperature"] = widgets.FloatSlider(
            value=5.0,
            min=0.5,
            max=20.0,
            step=0.25,
            description="Temperature:",
            continuous_update=False,
        )
        w["bias"] = widgets.FloatSlider(
            value=-3.0,
            min=-8.0,
            max=2.0,
            step=0.1,
            description="Bias:",
            continuous_update=False,
        )

        # Detection thresholds
        w["pixel_threshold"] = widgets.FloatSlider(
            value=0.5,
            min=0.05,
            max=0.99,
            step=0.01,
            description="Px thresh:",
            continuous_update=False,
        )
        w["min_area"] = widgets.IntSlider(
            value=4,
            min=1,
            max=200,
            step=1,
            description="Min area:",
            continuous_update=False,
        )
        w["max_area"] = widgets.IntSlider(
            value=50000,
            min=100,
            max=100000,
            step=100,
            description="Max area:",
            continuous_update=False,
        )

        # Scoring
        w["confidence_exp"] = widgets.FloatSlider(
            value=1.5,
            min=0.1,
            max=4.0,
            step=0.1,
            description="Conf α:",
            continuous_update=False,
        )
        w["area_exp"] = widgets.FloatSlider(
            value=0.3,
            min=-1.0,
            max=2.0,
            step=0.05,
            description="Area β:",
            continuous_update=False,
        )
        w["score_threshold"] = widgets.FloatSlider(
            value=0.0,
            min=0.0,
            max=5.0,
            step=0.05,
            description="Score thresh:",
            continuous_update=False,
        )

        # Visualization options
        w["show_bboxes"] = widgets.Checkbox(
            value=True,
            description="Show bboxes",
        )
        w["show_contours"] = widgets.Checkbox(
            value=True,
            description="Show contours",
        )
        w["colormap"] = widgets.Dropdown(
            options=["inferno", "hot", "jet", "magma", "viridis"],
            value="inferno",
            description="Colormap:",
        )

        return w

    def _render(self, **kwargs):
        """Re-render visualization with current widget values."""
        from agx_core.models.reversed_autoencoder.anomaly import (
            compute_anomaly_scores,
            calibrate_scores,
            detect_anomalies,
        )

        idx = kwargs["sample_idx"]
        image = self._cache.images[idx]
        recon = self._cache.reconstructions[idx]
        kld = self._cache.kld_maps[idx]
        feat = (
            self._cache.feature_maps[idx]
            if self._cache.feature_maps is not None
            else None
        )

        # 1. Compute scores
        score_map = compute_anomaly_scores(
            image,
            recon,
            kld,
            feat,
            weights=(kwargs["w_pixel"], kwargs["w_kld"], kwargs["w_feature"]),
        )

        # 2. Calibrate
        prob_map = calibrate_scores(
            score_map,
            temperature=kwargs["temperature"],
            bias=kwargs["bias"],
        )

        # 3. Detect
        detections = detect_anomalies(
            prob_map,
            pixel_threshold=kwargs["pixel_threshold"],
            min_area_px=kwargs["min_area"],
            max_area_px=kwargs["max_area"],
            confidence_exponent=kwargs["confidence_exp"],
            area_exponent=kwargs["area_exp"],
            score_threshold=kwargs["score_threshold"],
            return_masks=True,
        )

        # 4. Render
        self._output.clear_output(wait=True)
        with self._output:
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))

            # Row 1: Original, Reconstruction, Pixel Error
            axs[0, 0].imshow(image, cmap="gray")
            axs[0, 0].set_title("Original")

            axs[0, 1].imshow(recon, cmap="gray")
            axs[0, 1].set_title("Reconstruction")

            pixel_error = (image - recon) ** 2
            if pixel_error.ndim == 3:
                pixel_error = pixel_error.mean(axis=-1)
            axs[0, 2].imshow(pixel_error, cmap=kwargs["colormap"])
            axs[0, 2].set_title("Pixel Error (MSE)")

            # Row 2: Score Map, Probability Map, Detections overlay
            axs[1, 0].imshow(score_map, cmap=kwargs["colormap"])
            axs[1, 0].set_title("Combined Score Map")

            axs[1, 1].imshow(prob_map, cmap=kwargs["colormap"], vmin=0, vmax=1)
            axs[1, 1].set_title(
                f"Probability Map (T={kwargs['temperature']:.1f}, b={kwargs['bias']:.1f})"
            )

            # Detection overlay on original
            overlay = (
                np.stack([image] * 3, axis=-1) if image.ndim == 2 else image.copy()
            )
            overlay = overlay.copy()

            for det in detections:
                if kwargs["show_bboxes"]:
                    x, y, w, h = det.bbox
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (1, 0, 0), 2)

                if kwargs["show_contours"] and det.mask is not None:
                    contours, _ = cv2.findContours(
                        det.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(overlay, contours, -1, (0, 1, 0), 1)

            axs[1, 2].imshow(overlay)
            n_det = len(detections)
            max_score = detections[0].score if detections else 0.0
            axs[1, 2].set_title(f"Detections: {n_det} | Max score: {max_score:.3f}")

            # Clean up axes
            for ax_row in axs:
                for ax in ax_row:
                    ax.set_xticks([])
                    ax.set_yticks([])

            plt.tight_layout()
            plt.show()

            # Print detection details
            if detections:
                print(f"\n{'─'*60}")
                print(
                    f"{'ID':<4} {'Score':<8} {'Mean P':<8} {'Max P':<8} {'Area':<8} {'BBox'}"
                )
                print(f"{'─'*60}")
                for i, det in enumerate(detections):
                    print(
                        f"{i:<4} {det.score:<8.3f} {det.mean_probability:<8.3f} "
                        f"{det.max_probability:<8.3f} {det.area_px:<8} {det.bbox}"
                    )

    def show(self):
        """Display the interactive explorer widget."""
        # Group widgets into sections
        sample_box = widgets.VBox(
            [self._widgets["sample_idx"]],
            layout=widgets.Layout(margin="0 0 10px 0"),
        )

        weights_box = widgets.VBox(
            [
                widgets.HTML("<b>Score Weights</b>"),
                self._widgets["w_pixel"],
                self._widgets["w_kld"],
                self._widgets["w_feature"],
            ]
        )

        calibration_box = widgets.VBox(
            [
                widgets.HTML("<b>Calibration</b>"),
                self._widgets["temperature"],
                self._widgets["bias"],
            ]
        )

        detection_box = widgets.VBox(
            [
                widgets.HTML("<b>Detection</b>"),
                self._widgets["pixel_threshold"],
                self._widgets["min_area"],
                self._widgets["max_area"],
            ]
        )

        scoring_box = widgets.VBox(
            [
                widgets.HTML("<b>Cluster Scoring</b>"),
                self._widgets["confidence_exp"],
                self._widgets["area_exp"],
                self._widgets["score_threshold"],
            ]
        )

        viz_box = widgets.VBox(
            [
                widgets.HTML("<b>Visualization</b>"),
                self._widgets["show_bboxes"],
                self._widgets["show_contours"],
                self._widgets["colormap"],
            ]
        )

        controls_left = widgets.VBox([sample_box, weights_box, calibration_box])
        controls_right = widgets.VBox([detection_box, scoring_box, viz_box])
        controls = widgets.HBox(
            [controls_left, controls_right],
            layout=widgets.Layout(margin="0 0 10px 0"),
        )

        # Interactive output
        interactive = widgets.interactive_output(
            self._render,
            {name: widget for name, widget in self._widgets.items()},
        )

        display(controls, self._output)

        # Trigger initial render
        self._render(**{name: w.value for name, w in self._widgets.items()})


__all__ = ["AnomalyExplorer"]
