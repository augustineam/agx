from __future__ import annotations

import keras

from keras import layers
from typing import Dict, Any, Sequence, Optional

from agx_core.helpers import _channel_axis
from agx_core.layers import Sequential


@keras.saving.register_keras_serializable(
    package="agx_core.models.reversed_autoencoder"
)
class ResidualBlock(layers.Layer):
    """Bottleneck residual connection.

    Args:
        filters: Number of output filters.
        activation: Activation to be used. Default 'leaky_rely'
    """

    def __init__(
        self,
        filters: int,
        bottleneck: bool = True,
        activation: str = "leaky_relu",
        name="residual",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.bottleneck = bottleneck
        self._activation = activation

    def build(self, input_shape: Sequence[int]):

        ch_axis = _channel_axis()
        in_ch = input_shape[ch_axis]

        mid_ch = self.filters // 2 if self.bottleneck else self.filters
        kernel_size = 1 if self.bottleneck else 3

        self.expand_conv = (
            Sequential(
                layers.Conv2D(self.filters, 1, padding="same", use_bias=False),
                layers.GroupNormalization(
                    max(1, self.filters // 4), axis=ch_axis, epsilon=1e-3
                ),
                name="expand_shortcut",
            )
            if in_ch != self.filters
            else layers.Identity()
        )

        self.conv1 = Sequential(
            layers.Conv2D(mid_ch, kernel_size, padding="same", use_bias=False),
            layers.GroupNormalization(max(1, mid_ch // 4), axis=ch_axis, epsilon=1e-3),
            layers.Activation(self._activation),
            name="conv1",
        )

        self.conv2 = Sequential(
            layers.Conv2D(mid_ch, 3, padding="same", use_bias=False),
            layers.GroupNormalization(max(1, mid_ch // 4), axis=ch_axis, epsilon=1e-3),
            layers.Activation(self._activation),
            name="conv2",
        )

        self.conv3 = (
            Sequential(
                layers.Conv2D(self.filters, 1, padding="same", use_bias=False),
                layers.GroupNormalization(
                    max(1, self.filters // 4), axis=ch_axis, epsilon=1e-3
                ),
                name="conv3",
            )
            if self.bottleneck
            else layers.Identity()
        )

        self.add = layers.Add()
        self.activation = layers.Activation(self._activation)

        self.expand_conv.build(input_shape)
        y_shape = self.expand_conv.compute_output_shape(input_shape)

        self.conv1.build(input_shape)
        x_shape = self.conv1.compute_output_shape(input_shape)

        self.conv2.build(x_shape)
        x_shape = self.conv2.compute_output_shape(x_shape)

        self.conv3.build(x_shape)
        x_shape = self.conv3.compute_output_shape(x_shape)

        self.add.build([x_shape, y_shape])

        super().build(input_shape)

    def compute_output_shape(self, input_shape: Sequence[int]) -> Sequence[int]:
        x_shape = self.conv1.compute_output_shape(input_shape)
        x_shape = self.conv2.compute_output_shape(x_shape)
        x_shape = self.conv3.compute_output_shape(x_shape)
        return x_shape

    def call(
        self, x: keras.KerasTensor, training: Optional[bool] = None
    ) -> keras.KerasTensor:

        # Either convolution or identity (no-op)
        shortcut = self.expand_conv(x, training=training)

        out = self.conv1(x, training=training)
        out = self.conv2(out, training=training)
        out = self.conv3(out, training=training)

        out = self.add([out, shortcut])

        return self.activation(out)

    def get_config(self) -> Dict[Any, Any]:
        config = super().get_config()
        config.update(
            dict(
                filters=self.filters,
                bottleneck=self.bottleneck,
                activation=self._activation,
            )
        )
        return config


__all__ = ["ResidualBlock"]
