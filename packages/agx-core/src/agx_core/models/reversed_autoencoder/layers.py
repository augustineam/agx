from __future__ import annotations

import keras

from agx_core.helpers import _channel_axis, _layer_norm_axis
from keras import ops, layers
from typing import Dict, Any, Union, Tuple, Sequence, Optional

_size = Union[int, Tuple[int, int]]


def _depth(v: int, divisor: int = 8, min_value: int | None = None) -> int:
    """Round channel count to the nearest multiple of *divisor*."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Ensure rounding doesn't reduce by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@keras.saving.register_keras_serializable(package="agx_core.models.reversed_autoencoder")
class ConvBlock(layers.Layer):
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

        self.conv = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            groups=groups,
            use_bias=use_bias,
        )
        self.norm = layers.LayerNormalization(axis=_layer_norm_axis())
        self.act = layers.LeakyReLU(0.2) if use_activation else layers.Identity()

    def build(self, input_shape: Sequence[int]):
        self.conv.build(input_shape)
        x_shape = self.conv.compute_output_shape(input_shape)
        self.norm.build(x_shape)
        super(ConvBlock, self).build(input_shape)

    def compute_output_shape(self, input_shape: Sequence[int]) -> Sequence[int]:
        return self.conv.compute_output_shape(input_shape)

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


@keras.saving.register_keras_serializable(package="agx_core.models.reversed_autoencoder")
class DeConvBlock(layers.Layer):
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
        self.norm = layers.LayerNormalization(axis=_layer_norm_axis())
        self.lrelu = layers.LeakyReLU(0.2)

    def build(self, input_shape: Sequence[int]):
        self.conv.build(input_shape)
        x_shape = self.conv.compute_output_shape(input_shape)
        self.norm.build(x_shape)
        super(DeConvBlock, self).build(input_shape)

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


@keras.saving.register_keras_serializable(package="agx_core.models.reversed_autoencoder")
class ResidualBlock(layers.Layer):
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
        self.lrelu = layers.LeakyReLU(0.2)
        self.add = layers.Add()

    def build(self, input_shape: Sequence[int]):

        expand = input_shape[_channel_axis()] != self.filters
        self.conv_expand = (
            layers.Conv2D(self.filters, 1, use_bias=False)
            if expand
            else layers.Identity()
        )

        self.conv_expand.build(input_shape)
        y_shape = self.conv_expand.compute_output_shape(input_shape)

        self.conv1.build(input_shape)
        x_shape = self.conv1.compute_output_shape(input_shape)

        self.conv2.build(x_shape)
        x_shape = self.conv2.compute_output_shape(x_shape)

        self.add.build([x_shape, y_shape])

        super(ResidualBlock, self).build(input_shape)

    def compute_output_shape(self, input_shape: Sequence[int]) -> Sequence[int]:
        x_shape = self.conv1.compute_output_shape(input_shape)
        x_shape = self.conv2.compute_output_shape(x_shape)
        return x_shape

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


@keras.saving.register_keras_serializable(package="agx_core.models.reversed_autoencoder")
class Reparameterization(layers.Layer):
    """Samples from latent space using the reparameterization trick: z = μ + σ * ε.

    Input: [z_mean, z_log_var]
    Output: Sampled latent tensor z.
    """

    def __init__(self, name="reparameterization", **kwargs):
        super().__init__(name=name, **kwargs)

    def build(self, input_shape: Sequence[Sequence[int]]):
        super(Reparameterization, self).build(input_shape)

    def compute_output_shape(
        self, input_shape: Sequence[Sequence[int]]
    ) -> Sequence[int]:
        return input_shape[0]

    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = keras.random.normal(shape=ops.shape(z_mean))
        return eps * ops.exp(z_log_var * 0.5) + z_mean

    def compute_output_shape(self, input_shape: Sequence[Sequence[int]]):
        return input_shape[0]


@keras.saving.register_keras_serializable(package="agx_core.models.reversed_autoencoder")
class UpsampleRefine(layers.Layer):
    """Bilinear 2 x upsample + Conv refinement.

    Replaces ConvTranspose upsampling to eliminate checkerboard artifacts.
    Bilinear provides smooth spatial expansion; the conv learns to refine
    and adjust channels.

    Args:
        filters: Output channels after refinement.
        kernel_size: Refinement conv kernel size.
    """

    def __init__(self, filters: int, kernel_size: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

        self.upsample = layers.UpSampling2D(size=2, interpolation="bilinear")
        self.conv = layers.Conv2D(filters, kernel_size, padding="same", use_bias=False)
        self.norm = layers.LayerNormalization(axis=_layer_norm_axis())
        self.act = layers.Activation("relu6")

    def build(self, input_shape):
        up_shape = self.upsample.compute_output_shape(input_shape)
        self.conv.build(up_shape)
        conv_shape = self.conv.compute_output_shape(up_shape)
        self.norm.build(conv_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        if keras.config.image_data_format() == "channels_last":
            shape[1] *= 2
            shape[2] *= 2
            shape[3] = self.filters
        else:
            shape[2] *= 2
            shape[3] *= 2
            shape[1] = self.filters
        return tuple(shape)

    def call(self, x, training=None):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x, training=training)
        return self.act(x)

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "kernel_size": self.kernel_size})
        return config


@keras.saving.register_keras_serializable(package="agx_core.models.reversed_autoencoder")
class FiLM(layers.Layer):
    """Feature-wise Linear Modulation for conditioning.

    Replaces concatenation-based conditioning. Applies per-channel
    affine transformation conditioned on an external vector:

        output = features * (1 + γ) + β

    where γ, β are predicted from the conditioning vector.
    Initializes to identity transform (γ=0, β=0).

    More expressive than concat because it modulates at every stage
    without inflating channel dimensions.

    Args:
        None — adapts to input shapes at build time.
    """

    def build(self, input_shape):
        from agx_core.layers import Sequential

        feature_shape, cond_shape = input_shape
        n_channels = feature_shape[_channel_axis()]
        self.gamma = Sequential(
            [
                layers.Conv2D(
                    n_channels, 1, kernel_initializer="zeros", bias_initializer="zeros"
                ),
                layers.GlobalAveragePooling2D(keepdims=True),
            ]
        )
        self.beta = Sequential(
            [
                layers.Conv2D(
                    n_channels, 1, kernel_initializer="zeros", bias_initializer="zeros"
                ),
                layers.GlobalAveragePooling2D(keepdims=True),
            ]
        )
        self.gamma.build(cond_shape)
        self.beta.build(cond_shape)
        super(FiLM, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs):
        features, cond = inputs
        g = self.gamma(cond)
        b = self.beta(cond)
        return features * (1.0 + g) + b

    def get_config(self):
        return super().get_config()


__all__ = [
    "ConvBlock",
    "DeConvBlock",
    "ResidualBlock",
    "Reparameterization",
    "UpsampleRefine",
    "FiLM",
]
