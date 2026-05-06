from __future__ import annotations

import keras

from keras import layers
from agx_core.helpers import _channel_axis


def _depth(v: int, divisor: int = 8, min_value: int | None = None) -> int:
    """Round channel count to the nearest multiple of *divisor*."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Ensure rounding doesn't reduce by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@keras.saving.register_keras_serializable(package="agx_core.models.mobilenet_v3")
class SqueezeExcite(layers.Layer):
    """Squeeze-and-Excite channel attention.

    Global pool → reduce (ReLU) → expand → hard-sigmoid gate.

    Args:
        se_ratio: Reduction ratio relative to input channels.
    """

    def __init__(self, se_ratio: float = 0.25, name="squeeze_and_excite", **kwargs):
        super().__init__(name=name, **kwargs)
        self.se_ratio = se_ratio

    def build(self, input_shape):
        ch = input_shape[_channel_axis()]
        se_ch = _depth(int(ch * self.se_ratio))

        self.pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.reduce = layers.Conv2D(se_ch, 1, padding="same")
        self.relu = layers.Activation("relu")
        self.expand = layers.Conv2D(ch, 1, padding="same")
        self.gate = layers.Activation("hard_sigmoid")
        self.mul = layers.Multiply()

        self.pool.build(input_shape)
        x_shape = self.pool.compute_output_shape(input_shape)
        self.reduce.build(x_shape)
        x_shape = self.reduce.compute_output_shape(x_shape)
        self.expand.build(x_shape)

        super().build(input_shape)

    def call(self, x, training=None):
        scale = self.pool(x)
        scale = self.reduce(scale, training=training)
        scale = self.relu(scale)
        scale = self.expand(scale, training=training)
        scale = self.gate(scale)
        return self.mul([x, scale])

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"se_ratio": self.se_ratio})
        return config


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
            layers.Conv2D(mid_ch, 1, padding="same", use_bias=False)
            if self._expand
            else None
        )
        self.expand_norm = layers.BatchNormalization(**_bn) if self._expand else None
        self.expand_act = layers.Activation(self._activation) if self._expand else None

        # Explicit padding for stride-2 (keeps spatial dims predictable)
        if self.strides == 2:
            pad = (self.kernel_size - 1) // 2
            self.dw_pad = layers.ZeroPadding2D(padding=pad)
        else:
            self.dw_pad = None

        # Depthwise: mid_ch → mid_ch
        self.dw_conv = layers.DepthwiseConv2D(
            self.kernel_size,
            strides=self.strides,
            padding="same" if self.strides == 1 else "valid",
            use_bias=False,
        )
        self.dw_norm = layers.BatchNormalization(**_bn)
        self.dw_act = layers.Activation(self._activation)

        # Squeeze-and-Excite
        self.se = SqueezeExcite(self.se_ratio) if self.se_ratio > 0 else None

        # Project: mid_ch → filters (no activation — linear projection)
        self.project_conv = layers.Conv2D(
            self.filters, 1, padding="same", use_bias=False
        )
        self.project_norm = layers.BatchNormalization(**_bn)

        # Residual shortcut
        self.use_shortcut = self.strides == 1 and in_ch == self.filters
        if self.use_shortcut:
            self.add = layers.Add()

        x_shape = input_shape
        if self._expand:
            # --- Build the layers ---
            self.expand_conv.build(input_shape)
            x_shape = self.expand_conv.compute_output_shape(input_shape)
            self.expand_norm.build(x_shape)

        if self.dw_pad is not None:
            self.dw_pad.build(x_shape)
            x_shape = self.dw_pad.compute_output_shape(x_shape)

        self.dw_conv.build(x_shape)
        x_shape = self.dw_conv.compute_output_shape(x_shape)
        self.dw_norm.build(x_shape)

        if self.se is not None:
            self.se.build(x_shape)
            x_shape = self.se.compute_output_shape(x_shape)

        self.project_conv.build(x_shape)
        x_shape = self.project_conv.compute_output_shape(x_shape)
        self.project_norm.build(x_shape)

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
        shortcut = x

        # Expand
        if self._expand:
            out = self.expand_conv(x)
            out = self.expand_norm(out, training=training)
            out = self.expand_act(out)
        else:
            out = x

        # Depthwise
        if self.dw_pad is not None:
            out = self.dw_pad(out)
        out = self.dw_conv(out)
        out = self.dw_norm(out, training=training)
        out = self.dw_act(out)

        # Squeeze-and-Excite
        if self.se is not None:
            out = self.se(out, training=training)

        # Project (linear — no activation)
        out = self.project_conv(out)
        out = self.project_norm(out, training=training)

        # Shortcut connection
        if self.use_shortcut:
            out = self.add([shortcut, out])

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
    "SqueezeExcite",
    "InvertedResidualBlock",
]
