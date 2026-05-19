from __future__ import annotations

import keras

from keras import layers, ops
from typing import Sequence, List

from agx_core.helpers import _channel_axis, _spatial_slice
from agx_core.models.mobilenet_v3 import InvertedResidualBlock
from agx_core.models.reversed_autoencoder.base import BaseDecoder
from agx_core.models.reversed_autoencoder.layers import FiLM
from agx_core.models.mobilenet_v3.layers import ResidualBlock
from agx_core.layers import Sequential, Upsample2x


class _MobileNetV3SmallDecoderBase(BaseDecoder):
    """Shared stage definitions and stem for both decoder variants."""

    def __init__(
        self,
        target_shape: Sequence[int] = (224, 224, 1),
        rgb_activation: str = "tanh",
        name: str = "mbnetv3_decoder",
        **kwargs,
    ):
        super().__init__(target_shape=target_shape, name=name, **kwargs)
        self.rgb_activation = rgb_activation

    def _build_stem(self, x_shape, ch_axis: int):
        """Build concat + stem; returns shape after stem."""
        self.stem = Sequential(
            layers.Conv2D(576, 1, padding="same", use_bias=False),
            layers.BatchNormalization(ch_axis, epsilon=1e-3, momentum=0.999),
            layers.Activation("hard_swish"),
            layers.SpatialDropout2D(0.3),
            name="stem",
        )
        self.stem.build(x_shape)
        return self.stem.compute_output_shape(x_shape)

    def _build_film_layers(self, num_stages: int):
        """Build one FiLM layer per stage."""
        return [FiLM(name=f"film_{i}") for i in range(num_stages)]

    def _build_stages(self, ch_axis: int) -> List[layers.Layer]:
        return [
            Sequential(
                # 7→14
                InvertedResidualBlock(96, 6, 5, activation="hard_swish", expand=False),
                InvertedResidualBlock(96, 6, 5, activation="hard_swish"),
                Upsample2x("nearest"),
                layers.SpatialDropout2D(0.2),
                ResidualBlock(96, False, activation="hard_swish"),
                ResidualBlock(96, False, activation="hard_swish"),
                name="stage_0",
            ),
            Sequential(
                # 14→28
                InvertedResidualBlock(40, 6, 5, activation="hard_swish"),
                InvertedResidualBlock(40, 6, 5, activation="hard_swish"),
                InvertedResidualBlock(48, 3, 5, activation="hard_swish"),
                InvertedResidualBlock(48, 3, 5, activation="hard_swish"),
                Upsample2x("nearest"),
                layers.SpatialDropout2D(0.15),
                ResidualBlock(40, False, activation="hard_swish"),
                ResidualBlock(40, False, activation="hard_swish"),
                name="stage_1",
            ),
            Sequential(
                # 28→56
                InvertedResidualBlock(24, 88.0 / 24, se_ratio=0.0),
                Upsample2x("nearest"),
                layers.SpatialDropout2D(0.1),
                ResidualBlock(24, False, activation="hard_swish"),
                ResidualBlock(24, False, activation="hard_swish"),
                name="stage_2",
            ),
            Sequential(
                # 56→112
                ResidualBlock(24, False, activation="hard_swish"),
                Upsample2x("nearest"),
                layers.SpatialDropout2D(0.05),
                ResidualBlock(16, False, activation="hard_swish"),
                ResidualBlock(16, False, activation="hard_swish"),
                name="stage_3",
            ),
            Sequential(
                # 112→224
                Upsample2x("nearest"),
                ResidualBlock(16, True, activation="hard_swish"),
                ResidualBlock(16, True, activation="hard_swish"),
                layers.Conv2D(16, 3, padding="same", use_bias=False),
                layers.BatchNormalization(ch_axis, epsilon=1e-3, momentum=0.999),
                layers.Activation("hard_swish"),
                name="stage_4",
            ),
        ]

    def get_config(self):
        config = super().get_config()
        config.update(rgb_activation=self.rgb_activation)
        return config


@keras.saving.register_keras_serializable(
    package="agx_core.models.reversed_autoencoder"
)
class MobileNetV3SmallDecoder(_MobileNetV3SmallDecoderBase):
    """Non-progressive MobileNetV3-Small decoder.

    Runs all 5 stages unconditionally. No growth API, no per-stage to_rgb
    heads — just concat → stem → stages → to_rgb.
    """

    def __init__(
        self,
        target_shape: Sequence[int] = (224, 224, 1),
        rgb_activation: str = "tanh",
        name: str = "mbnetv3_decoder",
        **kwargs,
    ):
        super().__init__(
            target_shape=target_shape,
            rgb_activation=rgb_activation,
            name=name,
            **kwargs,
        )

    def build(self, input_shape):
        x_shape, c_shape = input_shape

        ch_axis = _channel_axis()
        out_ch = self.target_shape[ch_axis]

        x_shape = self._build_stem(x_shape, ch_axis)

        self.stages: List[layers.Layer] = self._build_stages(ch_axis)
        self.film: List[FiLM] = self._build_film_layers(len(self.stages))

        self.to_rgb = Sequential(
            layers.Conv2D(out_ch, 1, padding="same", use_bias=True),
            layers.Activation(self.rgb_activation),
            name="to_rgb",
        )

        for film, stage in zip(self.film, self.stages):
            film.build([x_shape, c_shape])
            stage.build(x_shape)
            x_shape = stage.compute_output_shape(x_shape)

        self.to_rgb.build(x_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        x_shape, _ = input_shape
        x_shape = self.stem.compute_output_shape(x_shape)
        for stage in self.stages:
            x_shape = stage.compute_output_shape(x_shape)
        return self.to_rgb.compute_output_shape(x_shape)

    def call(self, inputs, training=None):
        x, cond = inputs
        x = self.stem(x, training=training)
        for film, stage in zip(self.film, self.stages):
            x = film([x, cond], training=training)
            x = stage(x, training=training)
        return self.to_rgb(x, training=training)

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(
    package="agx_core.models.reversed_autoencoder"
)
class MobileNetV3SmallProgressiveDecoder(_MobileNetV3SmallDecoderBase):
    """Progressive-growing MobileNetV3-Small decoder.

    Starts at stage 0 (7→14) only and grows outward via ``grow()``.
    Each stage has its own ``to_rgb`` head; fade-in blends the previous
    head (bilinear upsampled) with the new head during alpha ∈ (0, 1).

    Stage layout (index = order of activation):
        0: 7→14
        1: 14→28
        2: 28→56
        3: 56→112
        4: 112→224
    """

    def __init__(
        self,
        target_shape: Sequence[int] = (224, 224, 1),
        rgb_activation: str = "tanh",
        initial_stage: int | None = None,
        initial_alpha: float | None = None,
        name: str = "mbnetv3_progressive_decoder",
        **kwargs,
    ):
        super().__init__(
            target_shape=target_shape,
            rgb_activation=rgb_activation,
            name=name,
            **kwargs,
        )
        self._initial_stage = initial_stage
        self._initial_alpha = initial_alpha

    # ------------------------------------------------------------------
    # Progressive growth API
    # ------------------------------------------------------------------

    @property
    def current_stage(self) -> int:
        return min(self._current_stage, len(self.stages) - 1)

    @property
    def alpha(self) -> float:
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

    def current_output_size(self) -> tuple:
        """Current output resolution as (H, W)."""
        h, w = self.target_shape[_spatial_slice()]
        if self.is_fully_grown:
            return (h, w)
        reduction = 2 ** (len(self.stages) - 1 - self._current_stage)
        return (h // reduction, w // reduction)

    def training_enabled(self, training: bool):
        super().training_enabled(training)
        curr = self.current_stage

        for idx, to_rgb in enumerate(self.to_rgb):
            to_rgb.trainable = (idx == curr) and training

        if training:
            # Freeze stages not yet grown
            for idx in range(curr + 1, len(self.stages)):
                self.film[idx].trainable = False
                self.stages[idx].trainable = False

            # During fade-in freeze earlier stages to prevent destabilization
            if self._alpha < 1.0:
                for idx in range(curr):
                    self.film[idx].trainable = False
                    self.stages[idx].trainable = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, input_shape):
        x_shape, c_shape = input_shape

        ch_axis = _channel_axis()
        out_ch = self.target_shape[ch_axis]

        x_shape = self._build_stem(x_shape, ch_axis)

        self.stages: List[layers.Layer] = self._build_stages(ch_axis)
        self.film: List[FiLM] = self._build_film_layers(len(self.stages))

        self.to_rgb: List[layers.Layer] = [
            Sequential(
                layers.Conv2D(out_ch, 1, padding="same", use_bias=True),
                layers.Activation(self.rgb_activation),
                name=f"to_rgb_{i}",
            )
            for i in range(len(self.stages))
        ]

        if self._initial_stage is not None:
            self._current_stage = self._initial_stage
            self._alpha = (
                self._initial_alpha if self._initial_alpha is not None else 1.0
            )
        else:
            self._current_stage = 0
            self._alpha = 1.0

        self._fade_upsample = layers.UpSampling2D(
            size=2, interpolation="bilinear", name="fade_upsampling"
        )

        for idx, (film, stage) in enumerate(zip(self.film, self.stages)):
            film.build([x_shape, c_shape])
            stage.build(x_shape)
            x_shape = stage.compute_output_shape(x_shape)
            self.to_rgb[idx].build(x_shape)

        super().build(input_shape)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def call(self, inputs, training=None):
        x, cond = inputs
        x = self.stem(x, training=training)

        if self.is_fully_grown:
            for film, stage in zip(self.film, self.stages):
                x = film([x, cond], training=training)
                x = stage(x, training=training)
            return self.to_rgb[-1](x, training=training)

        return self._call_progressive([x, cond], training=training)

    def _call_progressive(self, inputs, training=None):
        x, cond = inputs
        cur = self._current_stage
        alpha = self._alpha

        for i in range(cur):
            x = self.film[i]([x, cond], training=training)
            x = self.stages[i](x, training=training)

        if alpha >= 1.0 or cur == 0:
            x = self.film[cur]([x, cond], training=training)
            x = self.stages[cur](x, training=training)
            return self.to_rgb[cur](x, training=training)

        # Fade-in: blend upsampled previous head with new head
        old_rgb = self._fade_upsample(
            self.to_rgb[cur - 1](ops.stop_gradient(x), training=False)
        )

        x = self.film[cur]([x, cond], training=training)
        x = self.stages[cur](x, training=training)
        new_rgb = self.to_rgb[cur](x, training=training)

        return (1.0 - alpha) * ops.stop_gradient(old_rgb) + alpha * new_rgb

    # ------------------------------------------------------------------
    # Shape inference
    # ------------------------------------------------------------------

    def compute_output_shape(self, input_shape):
        x_shape, _ = input_shape
        x_shape = self.stem.compute_output_shape(x_shape)
        for i in range(self._current_stage + 1):
            x_shape = self.stages[i].compute_output_shape(x_shape)
        return self.to_rgb[self._current_stage].compute_output_shape(x_shape)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_config(self):
        config = super().get_config()
        config.update(
            initial_stage=self._current_stage,
            initial_alpha=self._alpha,
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def MobileNetV3SmallDecoder_factory(progressive: bool = False, **kwargs):
    """Factory for backwards compatibility.

    Returns ``MobileNetV3SmallProgressiveDecoder`` when ``progressive=True``,
    otherwise ``MobileNetV3SmallDecoder``.
    """
    cls = MobileNetV3SmallProgressiveDecoder if progressive else MobileNetV3SmallDecoder
    return cls(**kwargs)


__all__ = [
    "MobileNetV3SmallDecoder",
    "MobileNetV3SmallProgressiveDecoder",
]
