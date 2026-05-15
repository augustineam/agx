from __future__ import annotations

import keras

from keras import layers, ops, applications as apps

from typing import Sequence, List, Optional

from agx_core.models.reversed_autoencoder.base import BaseEncoder
from agx_core.models.mobilenet_v3.layers import InvertedResidualBlock
from agx_core.layers import Sequential, SoftClamp
from agx_core.helpers import _channel_axis, _spatial_slice


class _MobileNetV3SmallEncoderBase(BaseEncoder):
    """Shared hyperparameters and latent projection head for both encoder variants."""

    def __init__(
        self,
        latent_size: int = 512,
        logvar_init: float = -2.0,
        logvar_min: float = -10.0,
        logvar_max: float = 2.0,
        name: str = "mbnetv3_encoder",
        **kwargs,
    ):
        super().__init__(latent_size=latent_size, name=name, **kwargs)
        self.logvar_init = logvar_init
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

    def _build_projection_head(self, x_shape, c_shape, ch_axis: int):
        """Build to_latent_concat + to_mean + to_logvar."""
        self.to_latent_concat = layers.Concatenate(axis=ch_axis)
        self.to_mean = layers.Conv2D(self.latent_size, 1, use_bias=True, name="to_mean")
        self.to_logvar = Sequential(
            layers.Conv2D(
                self.latent_size,
                1,
                use_bias=True,
                bias_initializer=keras.initializers.Constant(self.logvar_init),
                name="to_logvar_conv",
            ),
            SoftClamp(
                min_val=self.logvar_min, max_val=self.logvar_max, name="logvar_clamp"
            ),
            name="to_logvar",
        )

        self.to_latent_concat.build([x_shape, c_shape])
        concat_shape = self.to_latent_concat.compute_output_shape([x_shape, c_shape])
        self.to_mean.build(concat_shape)
        self.to_logvar.build(concat_shape)

    def _project_latent(self, x, c, training=None):
        h = self.to_latent_concat([x, c])
        mean = self.to_mean(h)
        logvar = self.to_logvar(h, training=training)
        return mean, logvar

    def get_config(self):
        config = super().get_config()
        config.update(
            logvar_init=self.logvar_init,
            logvar_min=self.logvar_min,
            logvar_max=self.logvar_max,
        )
        return config


@keras.saving.register_keras_serializable(
    package="agx_core.models.reversed_autoencoder"
)
class MobileNetV3SmallEncoder(_MobileNetV3SmallEncoderBase):
    """Encoder backed by ``keras.applications.MobileNetV3Small``.

    Supports pretrained ImageNet weights. Input channels other than 3 are
    projected to RGB via a 1x1 conv before the backbone.

    The full backbone runs to layer 156 (``activation_17``, 7x7x576); its
    output feeds directly into concat → to_mean/to_logvar.
    Intermediate taps for ``embedding_loss``:
        4   activation                 112x112x16
        16  expanded_conv_project_bn    56x56x16
        34  expanded_conv_2_add         28x28x24
        49  expanded_conv_3_project_bn  14x14x40
        153 expanded_conv_10_add         7x7x96
    """

    def __init__(
        self,
        latent_size: int = 512,
        rgb_activation: str = "tanh",
        logvar_init: float = -2.0,
        logvar_min: float = -10.0,
        logvar_max: float = 2.0,
        weights: Optional[str] = None,
        freeze_backbone: bool = True,
        name: str = "mbnetv3_backbone_encoder",
        **kwargs,
    ):
        super().__init__(
            latent_size=latent_size,
            logvar_init=logvar_init,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
            name=name,
            **kwargs,
        )
        self.rgb_activation = rgb_activation
        self._pretrained_weights = weights
        self.freeze_backbone = freeze_backbone

    def training_enabled(self, training: bool):
        self.trainable = training
        self.backbone.trainable = training and not self.freeze_backbone

    def build(self, input_shape: Sequence[Sequence[int]]):
        x_shape, c_shape = input_shape

        ch_axis = _channel_axis()
        h, w = x_shape[_spatial_slice()]
        in_ch = x_shape[ch_axis]

        self._latent_spatial_res = (h // 32, w // 32)

        base_model = apps.MobileNetV3Small(
            input_shape=(h, w, 3), include_top=False, weights=self._pretrained_weights
        )
        skip_taps = [4, 16, 34, 49, 153]
        self.backbone = keras.Model(
            inputs=base_model.input,
            outputs=[base_model.layers[i].output for i in skip_taps]
            + [base_model.layers[156].output],
            name="mbnetv3_small",
        )

        self.to_rgb = (
            layers.Identity()
            if in_ch == 3
            else Sequential(
                layers.Conv2D(3, 1, padding="same", use_bias=True),
                layers.Activation(self.rgb_activation),
                name="to_rgb",
            )
        )

        self.to_rgb.build(x_shape)
        x_shape = self.to_rgb.compute_output_shape(x_shape)
        self.backbone.build(x_shape)
        # backbone outputs: [skip_0..skip_4, full_output] — projection head takes full_output
        x_shape = self.backbone.compute_output_shape(x_shape)[-1]

        self._build_projection_head(x_shape, c_shape, ch_axis)
        super().build(input_shape)

    def call(self, inputs, training=None):
        x, c = inputs
        x = self.to_rgb(x, training=training)
        *features, x = self.backbone(x, training=training)
        mean, logvar = self._project_latent(x, c, training=training)
        return (mean, logvar), features

    def compute_output_shape(self, input_shape):
        x_shape, _ = input_shape
        B = x_shape[0]
        ch_axis = _channel_axis()

        latent_shape = [B, 0, 0, 0]
        latent_shape[ch_axis] = self.latent_size
        latent_shape[_spatial_slice()] = list(self._latent_spatial_res)
        latent_shape = tuple(latent_shape)

        # Tap channels: [16@112, 16@56, 24@28, 40@14, 96@7]
        h, w = x_shape[_spatial_slice()]
        filters = [16, 16, 24, 40, 96]
        features_shape = []
        for f in filters:
            h, w = h // 2, w // 2
            shape = [B, 0, 0, 0]
            shape[ch_axis] = f
            shape[_spatial_slice()] = [h, w]
            features_shape.append(tuple(shape))

        return (latent_shape, latent_shape), features_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            rgb_activation=self.rgb_activation,
            weights=self._pretrained_weights,
            freeze_backbone=self.freeze_backbone,
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(
    package="agx_core.models.reversed_autoencoder"
)
class MobileNetV3SmallProgressiveEncoder(_MobileNetV3SmallEncoderBase):
    """Progressive-growing MobileNetV3-Small encoder.

    Starts at the deepest stage only and grows outward via ``grow()``.
    ``alpha`` controls the fade-in blend (0 → 1) for each newly activated stage.

    Stage layout (highest index = deepest / first active):
        0: 224→112  (Conv stem)
        1: 112→56   (IRB ×1)
        2: 56→28    (IRB ×1)
        3: 28→14    (IRB ×2)
        4: 14→7     (IRB ×7)
    """

    def __init__(
        self,
        latent_size: int = 512,
        rgb_activation: str = "tanh",
        logvar_init: float = -2.0,
        logvar_min: float = -10.0,
        logvar_max: float = 2.0,
        initial_stage: int | None = None,
        initial_alpha: float | None = None,
        name: str = "mbnetv3_progressive_encoder",
        **kwargs,
    ):
        super().__init__(
            latent_size=latent_size,
            logvar_init=logvar_init,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
            name=name,
            **kwargs,
        )
        self.rgb_activation = rgb_activation
        self._initial_stage = initial_stage
        self._initial_alpha = initial_alpha

    # ------------------------------------------------------------------
    # Progressive growth API
    # ------------------------------------------------------------------

    @property
    def current_stage(self) -> int:
        """Index of the highest active encoder stage."""
        return min(self._current_stage, len(self.stages) - 1)

    @property
    def alpha(self) -> float:
        """Fade-in blend factor for the newest active stage (0 → 1)."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = max(0.0, min(1.0, value))

    @property
    def is_fully_grown(self) -> bool:
        return self._current_stage >= len(self.stages) - 1 and self._alpha >= 1.0

    def grow(self):
        """Activate the next stage and reset alpha to 0 for fade-in."""
        if self._current_stage >= len(self.stages) - 1:
            return
        self._current_stage += 1
        self._alpha = 0.0

    def current_input_size(self) -> tuple:
        """Current progressive input resolution as (H, W)."""
        if self.is_fully_grown:
            return self._stage_resolutions[0]
        return self._stage_resolutions[len(self.stages) - 1 - self.current_stage]

    def training_enabled(self, training: bool):
        super().training_enabled(training)
        start_stage_index = len(self.stages) - 1 - self.current_stage

        for idx, from_rgb in enumerate(self.from_rgb):
            from_rgb.trainable = (idx == start_stage_index) and training

        if training:
            for stage in self.stages[:start_stage_index]:
                stage.trainable = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_stages(self, ch_axis: int):
        self.stages: List[layers.Layer] = [
            # 224→112
            Sequential(
                layers.Conv2D(16, 3, 2, padding="same", use_bias=False),
                layers.BatchNormalization(ch_axis, epsilon=1e-3, momentum=0.999),
                layers.Activation("hard_swish"),
                name="stage_0",
            ),
            # 112→56
            Sequential(
                InvertedResidualBlock(16, 1.0, strides=2, expand=False),
                layers.SpatialDropout2D(0.05),
                name="stage_1",
            ),
            # 56→28
            Sequential(
                InvertedResidualBlock(24, 72.0 / 24, strides=2, se_ratio=0.0),
                layers.SpatialDropout2D(0.1),
                name="stage_2",
            ),
            # 28→14
            Sequential(
                InvertedResidualBlock(24, 88.0 / 24, se_ratio=0.0),
                InvertedResidualBlock(40, 4, 5, 2, activation="hard_swish"),
                layers.SpatialDropout2D(0.15),
                name="stage_3",
            ),
            # 14→7
            Sequential(
                InvertedResidualBlock(40, 6, 5, activation="hard_swish"),
                InvertedResidualBlock(40, 6, 5, activation="hard_swish"),
                InvertedResidualBlock(48, 3, 5, activation="hard_swish"),
                InvertedResidualBlock(48, 3, 5, activation="hard_swish"),
                InvertedResidualBlock(96, 6, 5, 2, activation="hard_swish"),
                layers.SpatialDropout2D(0.2),
                InvertedResidualBlock(96, 6, 5, activation="hard_swish"),
                InvertedResidualBlock(96, 6, 5, activation="hard_swish"),
                name="stage_4",
            ),
        ]

    def build(self, input_shape: Sequence[Sequence[int]]):
        x_shape, c_shape = input_shape

        ch_axis = _channel_axis()
        in_ch = x_shape[ch_axis]
        h, w = x_shape[_spatial_slice()]
        self._latent_spatial_res = (h // 32, w // 32)

        self._build_stages(ch_axis)

        self.stages_head = Sequential(
            layers.Conv2D(576, 1, padding="same", use_bias=False),
            layers.GroupNormalization(groups=16, axis=ch_axis, epsilon=1e-3),
            layers.Activation("hard_swish"),
            layers.SpatialDropout2D(0.3),
            name="stages_head",
        )

        # from_rgb projections: identity when input channels already match
        input_channels = [in_ch, 16, 16, 24, 40]
        self.from_rgb: List[layers.Layer] = []
        for i, ch in enumerate(input_channels):
            self.from_rgb.append(
                layers.Identity(name=f"from_rgb_{i}")
                if ch == in_ch
                else Sequential(
                    layers.Conv2D(ch, 1, padding="same", use_bias=True),
                    layers.Activation(self.rgb_activation),
                    name=f"from_rgb_{i}",
                )
            )

        # Build stages + from_rgb, tracking spatial resolution at each entry
        self._stage_resolutions = []
        cur_shape = x_shape
        for stage, from_rgb in zip(self.stages, self.from_rgb):
            rgb_shape = list(cur_shape)
            rgb_shape[ch_axis] = in_ch
            self._stage_resolutions.append(tuple(rgb_shape[_spatial_slice()]))
            stage.build(cur_shape)
            from_rgb.build(tuple(rgb_shape))
            cur_shape = stage.compute_output_shape(cur_shape)

        # Seed progressive state
        if self._initial_stage is not None:
            self._current_stage: int = self._initial_stage
            self._alpha: float = (
                self._initial_alpha if self._initial_alpha is not None else 1.0
            )
        else:
            self._current_stage = 0
            self._alpha = 1.0

        self.stages_head.build(cur_shape)
        head_shape = self.stages_head.compute_output_shape(cur_shape)

        self._build_projection_head(head_shape, c_shape, ch_axis)
        super().build(input_shape)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def call(self, inputs, training=None):
        x, c = inputs

        if self.is_fully_grown:
            features = []
            for stage in self.stages:
                x = stage(x, training=training)
                features.append(x)
            x = self.stages_head(features[-1], training=training)
            mean, logvar = self._project_latent(x, c, training=training)
            return (mean, logvar), features

        return self._call_progressive(inputs, training=training)

    def _call_progressive(self, inputs, training=None):
        x, c = inputs
        alpha = self._alpha
        start = len(self.stages) - 1 - self._current_stage

        # No predecessor to blend from at stage 0 — treat as alpha=1
        if alpha >= 1.0 or self._current_stage == 0:
            x = self.from_rgb[start](x, training=training)
            features = []
            for i in range(start, len(self.stages)):
                x = self.stages[i](x, training=training)
                features.append(x)
        else:
            # Old path: downscale input to the resolution of the next deeper stage
            old_size = self._stage_resolutions[start + 1]
            x_old = ops.stop_gradient(
                self.from_rgb[start + 1](ops.image.resize(x, old_size), training=False)
            )

            x_new = self.stages[start](
                self.from_rgb[start](x, training=training), training=training
            )
            x = (1.0 - alpha) * x_old + alpha * x_new
            features = [x]

            for i in range(start + 1, len(self.stages)):
                x = self.stages[i](x, training=training)
                features.append(x)

        x = self.stages_head(x, training=training)
        mean, logvar = self._project_latent(x, c, training=training)
        return (mean, logvar), features

    # ------------------------------------------------------------------
    # Shape inference
    # ------------------------------------------------------------------

    def compute_output_shape(self, input_shape):
        x_shape, _ = input_shape
        B = x_shape[0]
        ch_axis = _channel_axis()

        latent_shape = [B, 0, 0, 0]
        latent_shape[ch_axis] = self.latent_size
        latent_shape[_spatial_slice()] = list(self._latent_spatial_res)
        latent_shape = tuple(latent_shape)

        h, w = x_shape[_spatial_slice()]
        start_idx = len(self.stages) - 1 - self._current_stage
        filters = [16, 16, 24, 40, 96]
        features_shape = []
        for i, f in enumerate(filters):
            h, w = h // 2, w // 2
            if i >= start_idx:
                shape = [B, 0, 0, 0]
                shape[ch_axis] = f
                shape[_spatial_slice()] = [h, w]
                features_shape.append(tuple(shape))

        return (latent_shape, latent_shape), features_shape

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_config(self):
        config = super().get_config()
        config.update(
            rgb_activation=self.rgb_activation,
            initial_stage=self._current_stage,
            initial_alpha=self._alpha,
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


__all__ = [
    "MobileNetV3SmallEncoder",
    "MobileNetV3SmallProgressiveEncoder",
]
