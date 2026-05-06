from __future__ import annotations

import keras

from keras import layers
from typing import Sequence, List


from agx_core.helpers import _channel_axis, _spatial_slice
from agx_core.models.mobilenet_v3 import InvertedResidualBlock
from agx_core.models.reversed_autoencoder.base import BaseDecoder
from agx_core.models.reversed_autoencoder.layers import *
from agx_core.models.reversed_autoencoder._spatial import *
from agx_core.layers import Sequential, Upsample2x


@keras.saving.register_keras_serializable(
    package="agx_core.models.reversed_autoencoder"
)
class MobileNetV3SmallDecoder(BaseDecoder):
    """MobileNetV3-Small-symmetric decoder.
    Args:
        target_shape: Output spatial shape (H, W, C) or (C, H, W).
        rgb_activation: Final activation ('tanh' or 'sigmoid').
    """

    def __init__(
        self,
        target_shape: Sequence[int] = (224, 224, 1),
        rgb_activation: str = "tanh",
        progressive: bool = False,
        initial_stage: int | None = None,
        initial_alpha: float | None = None,
        name: str = "mbnetv3_decoder",
        **kwargs,
    ):
        super(MobileNetV3SmallDecoder, self).__init__(
            target_shape=target_shape, name=name, **kwargs
        )

        self.rgb_activation = rgb_activation
        self.progressive = progressive
        self._initial_stage = initial_stage
        self._initial_alpha = initial_alpha

    @property
    def current_stage(self) -> int:
        """Index of the highest active decoder stage."""
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
        """Activate the next decoder stage and reset alpha to 0."""
        if self._current_stage >= len(self.stages) - 1:
            return  # already fully grown

        self._current_stage += 1
        self._alpha = 0.0

    def build(self, input_shape):
        x_shape, c_shape = input_shape

        ch_axis = _channel_axis()
        out_ch = self.target_shape[ch_axis]

        self.concat = layers.Concatenate(ch_axis)

        self.stem = Sequential(
            layers.Conv2D(576, 1, padding="same", use_bias=False),
            layers.BatchNormalization(ch_axis, epsilon=1e-3, momentum=0.999),
            layers.Activation("hard_swish"),
            layers.SpatialDropout2D(0.3),
            name="stem",
        )

        self.stages: List[layers.Layer] = [
            Sequential(
                # 7->14
                InvertedResidualBlock(96, 6, 5, activation="hard_swish", expand=False),
                InvertedResidualBlock(96, 6, 5, activation="hard_swish"),
                Upsample2x("nearest"),
                InvertedResidualBlock(96, 6, 5, activation="hard_swish"),
                layers.SpatialDropout2D(0.2),
                name="stage_0",
            ),
            Sequential(
                # 14->28
                InvertedResidualBlock(40, 6, 5, activation="hard_swish"),
                InvertedResidualBlock(40, 6, 5, activation="hard_swish"),
                InvertedResidualBlock(48, 3, 5, activation="hard_swish"),
                InvertedResidualBlock(48, 3, 5, activation="hard_swish"),
                Upsample2x("nearest"),
                InvertedResidualBlock(40, 4, 5, activation="hard_swish"),
                layers.SpatialDropout2D(0.15),
                name="stage_1",
            ),
            Sequential(
                # 28->56
                InvertedResidualBlock(24, 88.0 / 24, se_ratio=0.0),
                Upsample2x("nearest"),
                InvertedResidualBlock(24, 72.0 / 24, se_ratio=0.0),
                layers.SpatialDropout2D(0.1),
                name="stage_2",
            ),
            Sequential(
                # 56->112
                ResidualBlock(24, False, activation="hard_swish"),
                Upsample2x("bilinear"),
                ResidualBlock(16, False, activation="hard_swish"),
                ResidualBlock(16, False, activation="hard_swish"),
                layers.SpatialDropout2D(0.05),
                name="stage_3",
            ),
            Sequential(
                # 112->224
                Upsample2x("bilinear"),
                ResidualBlock(16, False, activation="hard_swish"),
                ResidualBlock(16, False, activation="hard_swish"),
                layers.Conv2D(16, 3, padding="same", use_bias=False),
                layers.BatchNormalization(ch_axis, epsilon=1e-3, momentum=0.999),
                layers.Activation("hard_swish"),
                name="stage_4",
            ),
        ]

        # to_rgb heads: always built — final head used in non-progressive mode too
        self.to_rgb: List[layers.Layer] = []
        for i in range(len(self.stages)):
            self.to_rgb.append(
                Sequential(
                    layers.Conv2D(out_ch, 1, padding="same", use_bias=True),
                    layers.Activation(self.rgb_activation),
                    name=f"to_rgb_{i}",
                )
            )

        if self._initial_stage is not None:
            # Restoring from serialized state (mid-growth or fully grown)
            self._current_stage = self._initial_stage
            self._alpha = (
                self._initial_alpha if self._initial_alpha is not None else 1.0
            )
        elif self.progressive:
            self._current_stage = 0
            self._alpha = 1.0
        else:
            self._current_stage = len(self.stages) - 1
            self._alpha = 1.0

        # Bilinear 2× for fade-in blending
        self._fade_upsample = layers.UpSampling2D(
            size=2, interpolation="bilinear", name="fade_upsampling"
        )

        self.concat.build([x_shape, c_shape])
        x_shape = self.concat.compute_output_shape([x_shape, c_shape])

        self.stem.build(x_shape)
        x_shape = self.stem.compute_output_shape(x_shape)

        for idx, stage in enumerate(self.stages):
            stage.build(x_shape)
            x_shape = stage.compute_output_shape(x_shape)
            self.to_rgb[idx].build(x_shape)

        super(MobileNetV3SmallDecoder, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        x_shape, c_shape = input_shape
        x_shape = self.concat.compute_output_shape([x_shape, c_shape])
        x_shape = self.stem.compute_output_shape(x_shape)

        n_active = self._current_stage + 1
        for i in range(n_active):
            x_shape = self.stages[i].compute_output_shape(x_shape)
        return self.to_rgb[self._current_stage].compute_output_shape(x_shape)

    def call(self, inputs, training=None):
        x, cond = inputs

        x = self.concat([x, cond])
        x = self.stem(x, training=training)

        if self.progressive and not self.is_fully_grown:
            return self._call_progressive(x, training=training)

        # Full-network forward
        for stage in self.stages:
            x = stage(x, training=training)
        return self.to_rgb[-1](x, training=training)

    def _call_progressive(self, x, training=None):
        cur = self._current_stage
        alpha = self._alpha

        # Run all stabilized stages
        for i in range(cur):
            x = self.stages[i](x, training=training)

        if alpha >= 1.0:
            x = self.stages[cur](x, training=training)
            return self.to_rgb[cur](x, training=training)

        # Fade-in: blend old (upsampled prev to_rgb) with new (current stage to_rgb)
        old_rgb = self.to_rgb[cur - 1](x, training=training)
        old_rgb = self._fade_upsample(old_rgb)

        x = self.stages[cur](x, training=training)
        new_rgb = self.to_rgb[cur](x, training=training)

        return (1.0 - alpha) * old_rgb + alpha * new_rgb

    def current_output_size(self) -> tuple:
        """Current progressive output resolution as (H, W)."""
        spatial = _spatial_slice()

        if not self.progressive or self.is_fully_grown:
            h, w = self.target_shape[spatial]
            return (h, w)

        h, w = self.target_shape[spatial]
        reduction = 2 ** (len(self.stages) - 1 - self._current_stage)
        return (h // reduction, w // reduction)

    def get_config(self):
        config = super(MobileNetV3SmallDecoder, self).get_config()
        config.update(
            {
                "rgb_activation": self.rgb_activation,
                "progressive": self.progressive,
                "initial_stage": self._current_stage,
                "initial_alpha": self._alpha,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


__all__ = ["MobileNetV3SmallDecoder"]
