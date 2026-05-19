from __future__ import annotations

import keras

from keras import layers

from agx_core.helpers import _channel_axis
from agx_core.layers import Sequential
from .squeeze_excite import SqueezeExcite
from ._helpers import _depth


@keras.saving.register_keras_serializable(package="agx_core.models.mobilenet_v3")
class InvertedResidualBlock(layers.Layer):
    """MobileNetV3-style inverted residual block.

    expand (1x1) → depthwise (k x k) → SE → project (1x1) → + residual

    Args:
        filters: Output channels (projection size).
        expand_ratio: Expansion ratio for intermediate channels.
        kernel_size: Depthwise convolution kernel size.
        stride: Depthwise convolution stride.
        se_ratio: Squeeze-and-excite reduction ratio (0 to disable).
        activation: Activation type ('relu' or 'hard_swish').
    """

    def __init__(
        self,
        filters: int,
        expand_ratio: float = 4.0,
        kernel_size: int = 3,
        strides: int = 1,
        se_ratio: float = 0.25,
        activation: str = "relu",
        expand: bool = True,
        name="inverted_residual",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        self.strides = strides
        self.se_ratio = se_ratio
        self._expand = expand
        self._activation = activation

    def build(self, input_shape):
        ch_axis = _channel_axis()
        in_ch = input_shape[ch_axis]
        mid_ch = _depth(int(in_ch * self.expand_ratio))

        _bn = dict(axis=ch_axis, epsilon=1e-3, momentum=0.999)

        # Expand: in_ch → mid_ch
        self.expand_conv = (
            Sequential(
                layers.Conv2D(mid_ch, 1, padding="same", use_bias=False),
                layers.BatchNormalization(**_bn),
                layers.Activation(self._activation),
                name="expand_conv",
            )
            if self._expand
            else layers.Identity()
        )

        # Depthwise: mid_ch → mid_ch
        self.dw_conv = Sequential(
            # Explicit padding for stride-2 (keeps spatial dims predictable)
            (
                layers.ZeroPadding2D(padding=(self.kernel_size - 1) // 2)
                if self.strides == 2
                else layers.Identity()
            ),
            layers.DepthwiseConv2D(
                self.kernel_size,
                strides=self.strides,
                padding="same" if self.strides == 1 else "valid",
                use_bias=False,
            ),
            layers.BatchNormalization(**_bn),
            layers.Activation(self._activation),
            name="dw_conv",
        )

        # Squeeze-and-Excite
        self.se = (
            SqueezeExcite(self.se_ratio) if self.se_ratio > 0 else layers.Identity()
        )

        # Project: mid_ch → filters (no activation — linear projection)
        self.project_conv = Sequential(
            layers.Conv2D(self.filters, 1, padding="same", use_bias=False),
            layers.BatchNormalization(**_bn),
            name="project_conv",
        )

        # Residual shortcut
        self.use_shortcut = self.strides == 1 and in_ch == self.filters
        if self.use_shortcut:
            self.add = layers.Add()

        x_shape = input_shape

        self.expand_conv.build(input_shape)
        x_shape = self.expand_conv.compute_output_shape(input_shape)

        self.dw_conv.build(x_shape)
        x_shape = self.dw_conv.compute_output_shape(x_shape)

        self.se.build(x_shape)
        x_shape = self.se.compute_output_shape(x_shape)

        self.project_conv.build(x_shape)
        x_shape = self.project_conv.compute_output_shape(x_shape)

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        ch_axis = _channel_axis()
        if self.strides > 1:
            # Spatial dims are halved by the stride
            if keras.config.image_data_format() == "channels_last":
                shape[1] = (
                    (shape[1] + self.strides - 1) // self.strides if shape[1] else None
                )
                shape[2] = (
                    (shape[2] + self.strides - 1) // self.strides if shape[2] else None
                )
            else:
                shape[2] = (
                    (shape[2] + self.strides - 1) // self.strides if shape[2] else None
                )
                shape[3] = (
                    (shape[3] + self.strides - 1) // self.strides if shape[3] else None
                )
        shape[ch_axis] = self.filters
        return tuple(shape)

    def call(self, x, training=None):

        # Expand or Identity (no-op)
        out = self.expand_conv(x, training=training)

        # Depthwise Convolution
        out = self.dw_conv(out, training=training)

        # Squeeze-and-Excite or Identity (no-op)
        out = self.se(out, training=training)

        # Project (linear — no activation)
        out = self.project_conv(out, training=training)

        # Shortcut connection
        if self.use_shortcut:
            out = self.add([x, out])

        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                filters=self.filters,
                expand_ratio=self.expand_ratio,
                kernel_size=self.kernel_size,
                strides=self.strides,
                se_ratio=self.se_ratio,
                expand=self._expand,
                activation=self._activation,
            )
        )
        return config


__all__ = [
    "InvertedResidualBlock",
]
