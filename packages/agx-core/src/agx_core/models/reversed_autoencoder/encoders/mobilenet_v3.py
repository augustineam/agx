from __future__ import annotations

import keras

from keras import layers, ops

from typing import Sequence, List

from agx_core.models.reversed_autoencoder.base import BaseEncoder
from agx_core.models.mobilenet_v3.layers import InvertedResidualBlock
from agx_core.layers import Sequential, Split
from agx_core.helpers import _channel_axis, _spatial_slice


@keras.saving.register_keras_serializable(
    package="agx_core.models.reversed_autoencoder"
)
class MobileNetV3SmallEncoder(BaseEncoder):

    def __init__(
        self,
        latent_size: int = 512,
        progressive: bool = False,
        rgb_activation: str = "tanh",
        name: str = "mbnetv3_encoder",
        **kwargs,
    ):
        super(MobileNetV3SmallEncoder, self).__init__(
            latent_size=latent_size, name=name, **kwargs
        )
        self.progressive = progressive
        self.rgb_activation = rgb_activation

    @property
    def current_stage(self) -> int:
        """Index of the highest active encoder stage."""
        return min(self._current_stage, len(self.stages) - 1)

    @property
    def alpha(self) -> float:
        """Fade-in blending factor for the newest active stage (0→1)."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = max(0.0, min(1.0, value))

    @property
    def is_fully_grown(self) -> bool:
        """True when all stages are active and fade-in is complete."""
        return self._current_stage >= len(self.stages) - 1 and self._alpha >= 1.0

    def grow(self):
        """Activate the next encoder stage and reset alpha to 0."""
        if self._current_stage >= len(self.stages) - 1:
            return  # already fully grown

        self._current_stage += 1
        self._alpha = 0.0

    def build(self, input_shape: Sequence[Sequence[int]]):
        x_shape, c_shape = input_shape

        ch_axis = _channel_axis()
        spatial = _spatial_slice()
        in_ch = x_shape[ch_axis]

        h, w = x_shape[spatial]
        self._latent_spatial_res = (h // 32, w // 32)

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

        self.stages_head = Sequential(
            layers.Conv2D(576, 1, padding="same", use_bias=False),
            layers.BatchNormalization(ch_axis, epsilon=1e-3, momentum=0.999),
            layers.Activation("hard_swish"),
            layers.SpatialDropout2D(0.3),
            name="stages_head",
        )

        self.to_latent = Sequential(
            layers.Concatenate(),
            layers.Conv2D(2 * self.latent_size, 1),
            Split(2),
            name="to_latent",
        )

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

        if self.progressive:
            self._current_stage: int = 0
            self._alpha: float = 1.0
        else:
            self._current_stage: int = len(self.stages) - 1
            self._alpha: float = 1.0

        self._stage_resolutions = []
        for layer, from_rgb in zip(self.stages, self.from_rgb):

            rgb_shape = list(x_shape)
            rgb_shape[ch_axis] = in_ch
            self._stage_resolutions.append(tuple(rgb_shape[_spatial_slice()]))

            layer.build(x_shape)
            from_rgb.build(tuple(rgb_shape))
            x_shape = layer.compute_output_shape(x_shape)

        self.stages_head.build(x_shape)
        x_shape = self.stages_head.compute_output_shape(x_shape)

        self.to_latent.build([x_shape, c_shape])

        super(MobileNetV3SmallEncoder, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        x_shape, _ = input_shape

        B = x_shape[0]
        channel_axis = _channel_axis()
        spatial_slice = _spatial_slice()

        latent_shape = [B, 0, 0, 0]
        latent_shape[channel_axis] = self.latent_size
        latent_shape[spatial_slice] = list(self.latent_spatial_res)
        latent_shape = tuple(latent_shape)

        features_shape = []
        h, w = x_shape[spatial_slice]
        start_idx = len(self.stages) - 1 - self._current_stage

        filters = [16, 16, 24, 40, 96]
        for i, f in enumerate(filters):
            h, w = h // 2, w // 2
            if i >= start_idx:
                shape = [B, 0, 0, 0]
                shape[channel_axis] = f
                shape[spatial_slice] = [h, w]
                features_shape.append(tuple(shape))

        return (latent_shape, latent_shape), features_shape

    def call(self, inputs, training=None):
        x, c = inputs
        if self.progressive and not self.is_fully_grown:
            return self._call_progressive(inputs, training=training)

        # Full-network forward (non-progressive or fully grown)
        features = []
        for layer in self.stages:
            x = layer(x, training=training)
            features.append(x)
        x = self.stages_head(x, training=training)
        mean, logvar = self.to_latent([x, c], training=training)
        return (mean, logvar), features

    def _call_progressive(self, inputs, training=None):
        x, c = inputs

        # Map stage index to stages indices (reversed)
        # Stage 0 uses stages[4], stage 1 uses stages[3:], etc.
        alpha = self._alpha
        start_stage_index = len(self.stages) - 1 - self._current_stage

        if alpha >= 1.0:
            # No blending — single path
            x = self.from_rgb[start_stage_index](x, training=training)
            features = []
            for i in range(start_stage_index, len(self.stages)):
                x = self.stages[i](x, training=training)
                features.append(x)
        else:
            # Old path: deeper from_rgb at lower resolution
            old_size = self._stage_resolutions[start_stage_index + 1]
            x_old = ops.image.resize(x, old_size)
            x_old = self.from_rgb[start_stage_index + 1](x_old, training=training)

            # New path: current from_rgb, then run the new stage
            x_new = self.from_rgb[start_stage_index](x, training=training)
            x_new = self.stages[start_stage_index](x_new, training=training)

            # Blend at the entry point of the next deeper stage
            x = (1.0 - alpha) * x_old + alpha * x_new

            features = [x]

            # Run remaining deeper stages
            for i in range(start_stage_index + 1, len(self.stages)):
                x = self.stages[i](x, training=training)
                features.append(x)

        x = self.stages_head(x, training=training)
        mean, logvar = self.to_latent([x, c], training=training)
        return (mean, logvar), features

    def current_input_size(self) -> tuple:
        """Current progressive input resolution as (H, W)."""
        if not self.progressive or self.is_fully_grown:
            return self._stage_resolutions[0]
        return self._stage_resolutions[len(self.stages) - 1 - self.current_stage]

    def get_config(self):
        config = super(MobileNetV3SmallEncoder, self).get_config()
        config.update(
            dict(
                progressive=self.progressive,
                rgb_activation=self.rgb_activation,
            )
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


__all__ = [
    "MobileNetV3SmallEncoder",
]
