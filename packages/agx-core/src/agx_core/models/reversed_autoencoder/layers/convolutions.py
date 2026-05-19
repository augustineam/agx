from __future__ import annotations

import keras

from keras import layers
from typing import Union, Tuple, Sequence, Optional

from agx_core.helpers import _channel_axis, _layer_norm_axis

_size = Union[int, Tuple[int, int]]


@keras.saving.register_keras_serializable(
    package="agx_core.models.reversed_autoencoder"
)
class ConvBlock(layers.Layer):
    """Fusing Conv2D, BatchNormalization, and optional LeakyReLU.

    Args:
        filters: Output channels.
        kernel_size: Convolution kernel size.
        strides: Convolution strides.
        padding: Padding mode ("valid" or "same").
        groups: Number of convolution groups.
        use_bias: Whether to use bias in Conv2D.
        use_activation: Whether to apply LeakyReLU.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: _size,
        strides: _size = (1, 1),
        padding: str = "valid",
        groups: int = 1,
        use_bias: bool = True,
        activation: Optional[str] = None,
        name="conv_block",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.groups = groups
        self.use_bias = use_bias
        self._activation = activation

    def build(self, input_shape: Sequence[int]):

        ch_axis = _channel_axis()
        groups = max(1, self.filters // 4)

        self.conv = layers.Conv2D(
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            groups=self.groups,
            use_bias=self.use_bias,
        )
        self.norm = layers.GroupNormalization(groups=groups, axis=ch_axis, epsilon=1e-3)
        self.activation = (
            layers.Activation(self._activation) if self._activation else None
        )

        self.conv.build(input_shape)
        x_shape = self.conv.compute_output_shape(input_shape)
        self.norm.build(x_shape)

        super().build(input_shape)

    def compute_output_shape(self, input_shape: Sequence[int]) -> Sequence[int]:
        return self.conv.compute_output_shape(input_shape)

    def call(
        self, x: keras.KerasTensor, training: Optional[bool] = None
    ) -> keras.KerasTensor:
        x = self.conv(x)
        x = self.norm(x, training=training)
        if self.activation:
            return self.activation(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                groups=self.groups,
                use_bias=self.use_bias,
                activation=self._activation,
            )
        )
        return config


@keras.saving.register_keras_serializable(
    package="agx_core.models.reversed_autoencoder"
)
class DeConvBlock(layers.Layer):
    """Fusing Conv2DTranspose, BatchNormalization, and LeakyReLU.

    Args:
        filters: Output channels.
        kernel_size: Convolution kernel size.
        strides: Convolution strides.
        padding: Padding mode ("valid" or "same").
        use_bias: Whether to use bias in Conv2DTranspose.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: _size,
        strides: _size = (1, 1),
        padding: str = "valid",
        output_padding: tuple[int, int] | None = None,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.output_padding = output_padding
        self.use_bias = use_bias

        self.conv = layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            use_bias=use_bias,
        )
        self.norm = layers.BatchNormalization(axis=_layer_norm_axis())
        self.lrelu = layers.LeakyReLU(0.2)

    def build(self, input_shape: Sequence[int]):
        self.conv.build(input_shape)
        x_shape = self.conv.compute_output_shape(input_shape)
        self.norm.build(x_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape: Sequence[int]) -> Sequence[int]:
        return self.conv.compute_output_shape(input_shape)

    def call(
        self, x: keras.KerasTensor, training: Optional[bool] = None
    ) -> keras.KerasTensor:
        x = self.conv(x)
        x = self.norm(x, training=training)
        return self.lrelu(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                output_padding=self.output_padding,
                use_bias=self.use_bias,
            )
        )
        return config


__all__ = [
    "ConvBlock",
    "DeConvBlock",
]
