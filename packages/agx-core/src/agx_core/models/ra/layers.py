from __future__ import annotations

import keras
from keras import ops

from typing import Dict, Any, Union, Tuple, Sequence, Optional

_size = Union[int, Tuple[int, int]]


@keras.saving.register_keras_serializable(package="agx_core.models.ra")
class Split(keras.layers.Layer):
    """Splits an input tensor along a specified axis.

    Args:
        num_or_size_splits: Number of equal splits or sizes of each split.
        axis: Axis to split along (default: -1).
    """

    def __init__(self, num_or_size_splits, axis=-1, name="split", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_or_size_splits = num_or_size_splits
        self.axis = axis

    def call(self, inputs):
        return ops.split(inputs, self.num_or_size_splits, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"num_or_size_splits": self.num_or_size_splits, "axis": self.axis}
        )
        return config


@keras.saving.register_keras_serializable(package="agx_core.models.ra")
class ConvBlock(keras.layers.Layer):
    """Fusing Conv2D, LayerNormalization, and optional LeakyReLU.

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
        use_activation: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.groups = groups
        self.use_bias = use_bias
        self.use_activation = use_activation

        self.conv = keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            groups=groups,
            use_bias=use_bias,
        )
        self.norm = keras.layers.LayerNormalization()
        self.act = (
            keras.layers.LeakyReLU(0.2) if use_activation else keras.layers.Identity()
        )

    def call(
        self, x: keras.KerasTensor, training: Optional[bool] = None
    ) -> keras.KerasTensor:
        x = self.conv(x)
        x = self.norm(x, training=training)
        return self.act(x)

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
                use_activation=self.use_activation,
            )
        )
        return config


@keras.saving.register_keras_serializable(package="agx_core.models.ra")
class DeConvBlock(keras.layers.Layer):
    """Fusing Conv2DTranspose, LayerNormalization, and LeakyReLU.

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
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias

        self.conv = keras.layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
        )
        self.norm = keras.layers.LayerNormalization()
        self.lrelu = keras.layers.LeakyReLU(0.2)

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
                use_bias=self.use_bias,
            )
        )
        return config


@keras.saving.register_keras_serializable(package="agx_core.models.ra")
class ResidualBlock(keras.layers.Layer):
    """Residual connection with optional channel expansion and grouped convolution.

    Args:
        filters: Number of output filters.
        scale: Scale factor for intermediate filters.
        groups: Number of convolution groups for the second conv layer.
    """

    def __init__(self, filters: int, scale: float = 1.0, groups: int = 1, **kwargs):
        super().__init__(**kwargs)

        fmid = int(filters * scale)

        self.filters = filters
        self.scale = scale
        self.groups = groups

        self.conv_expand = None
        self.conv1 = ConvBlock(fmid, 3, strides=1, padding="same", use_bias=False)
        self.conv2 = ConvBlock(
            filters,
            3,
            strides=1,
            padding="same",
            groups=groups,
            use_bias=False,
            use_activation=False,
        )
        self.lrelu = keras.layers.LeakyReLU(0.2)
        self.add = keras.layers.Add()

    def build(self, input_shape: Sequence[int]):
        expand = input_shape[-1] != self.filters
        self.conv_expand = (
            keras.layers.Conv2D(self.filters, 1, use_bias=False)
            if expand
            else keras.layers.Identity()
        )
        self.built = True

    def call(
        self, x: keras.KerasTensor, training: Optional[bool] = None
    ) -> keras.KerasTensor:
        expand = self.conv_expand(x, training=training)
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.add([x, expand])
        return self.lrelu(x)

    def get_config(self) -> Dict[Any, Any]:
        config = super().get_config()
        config.update(dict(filters=self.filters, scale=self.scale, groups=self.groups))
        return config


@keras.saving.register_keras_serializable(package="agx_core.models.ra")
class Reparameterization(keras.layers.Layer):
    """Samples from latent space using the reparameterization trick: z = μ + σ * ε.

    Input: [z_mean, z_log_var]
    Output: Sampled latent tensor z.
    """

    def __init__(self, name="reparameterization", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = keras.random.normal(shape=ops.shape(z_mean))
        return eps * ops.exp(z_log_var * 0.5) + z_mean

    def compute_output_shape(self, input_shape: Sequence[Sequence[int]]):
        return input_shape[0]


__all__ = [
    "Split",
    "ConvBlock",
    "DeConvBlock",
    "ResidualBlock",
    "Reparameterization",
]
