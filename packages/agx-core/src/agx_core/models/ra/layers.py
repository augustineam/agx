from __future__ import annotations

import keras
from keras import ops

from typing import Dict, Any, Union, Tuple, Sequence, Optional

_size = Union[int, Tuple[int, int]]


def _axis_channel():
    if keras.config.image_data_format() == "channels_last":
        return -1
    return 1


def _layer_norm_axis():
    if keras.config.image_data_format() == "channels_last":
        return -1
    return [2, 3]


@keras.saving.register_keras_serializable(package="agx_core.models.ra")
class Split(keras.layers.Layer):
    """Splits an input tensor along a specified axis.

    Args:
        num_or_size_splits: Number of equal splits or sizes of each split.
        axis: Axis to split along (default: -1 if 'channels_last', 1 otherwise).
    """

    def __init__(self, num_or_size_splits, axis=-1, name="split", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_or_size_splits = num_or_size_splits
        self.axis = axis

    def build(self, input_shape: Sequence[int]):
        self.input_spec = keras.layers.InputSpec(shape=(None, *input_shape[1:]))
        super(Split, self).build(input_shape)

    def compute_output_shape(self, input_shape: Sequence[int]):
        input_shape = list(input_shape)
        axis = self.axis if self.axis >= 0 else len(input_shape) + self.axis

        if isinstance(self.num_or_size_splits, int):
            # Equal splits
            split_size = input_shape[axis] // self.num_or_size_splits
            output_shapes = []
            for _ in range(self.num_or_size_splits):
                output_shape = input_shape.copy()
                output_shape[axis] = split_size
                output_shapes.append(tuple(output_shape))
        else:
            # Custom split sizes
            output_shapes = []
            for size in self.num_or_size_splits:
                output_shape = input_shape.copy()
                output_shape[axis] = size
                output_shapes.append(tuple(output_shape))

        return output_shapes

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
        self.norm = keras.layers.LayerNormalization(axis=_layer_norm_axis())
        self.act = (
            keras.layers.LeakyReLU(0.2) if use_activation else keras.layers.Identity()
        )

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

        self.conv = keras.layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            use_bias=use_bias,
        )
        self.norm = keras.layers.LayerNormalization(axis=_layer_norm_axis())
        self.lrelu = keras.layers.LeakyReLU(0.2)

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

        expand = input_shape[_axis_channel()] != self.filters
        self.conv_expand = (
            keras.layers.Conv2D(self.filters, 1, use_bias=False)
            if expand
            else keras.layers.Identity()
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


@keras.saving.register_keras_serializable(package="agx_core.models.ra")
class Reparameterization(keras.layers.Layer):
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


# ... existing imports and helpers ...


@keras.saving.register_keras_serializable(package="agx_core.models.ra")
class SqueezeExcite(keras.layers.Layer):
    """Squeeze-and-Excite channel attention.

    Global pool → reduce → expand → sigmoid gate.

    Args:
        se_ratio: Reduction ratio relative to input channels.
    """

    def __init__(self, se_ratio: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.se_ratio = se_ratio

    def build(self, input_shape):
        ch = input_shape[_axis_channel()]
        se_ch = max(1, int(ch * self.se_ratio))

        self.pool = keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.reduce = keras.layers.Conv2D(se_ch, 1, activation="relu")
        self.expand = keras.layers.Conv2D(ch, 1, activation="sigmoid")

        self.pool.build(input_shape)
        x_shape = self.pool.compute_output_shape(input_shape)
        self.reduce.build(x_shape)
        x_shape = self.reduce.compute_output_shape(x_shape)
        self.expand.build(x_shape)

        super().build(input_shape)

    def call(self, x, training=None):
        scale = self.pool(x)
        scale = self.reduce(scale)
        scale = self.expand(scale)
        return x * scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"se_ratio": self.se_ratio})
        return config


@keras.saving.register_keras_serializable(package="agx_core.models.ra")
class InvertedResidualBlock(keras.layers.Layer):
    """MobileNetV3-style inverted residual block.

    expand (1×1) → depthwise (k×k) → SE → project (1×1) → + residual

    Uses depthwise separable convolutions for parameter efficiency
    and squeeze-and-excite for channel attention. Significantly more
    expressive per parameter than standard Conv+ResBlock.

    Args:
        filters: Output channels (projection size).
        expand_ratio: Expansion ratio for intermediate channels.
        kernel_size: Depthwise convolution kernel size.
        se_ratio: Squeeze-and-excite reduction ratio (0 to disable).
        activation: Activation type ('relu6' or 'hard_swish').
    """

    def __init__(
        self,
        filters: int,
        expand_ratio: float = 4.0,
        kernel_size: int = 3,
        se_ratio: float = 0.25,
        activation: str = "relu6",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        self.se_ratio = se_ratio
        self._activation = activation

    def build(self, input_shape):
        in_ch = input_shape[_axis_channel()]
        mid_ch = int(in_ch * self.expand_ratio)
        norm_axis = _layer_norm_axis()

        # Expand: in_ch → mid_ch
        self.expand_conv = keras.layers.Conv2D(mid_ch, 1, use_bias=False)
        self.expand_norm = keras.layers.LayerNormalization(axis=norm_axis)
        self.expand_act = keras.layers.Activation(self._activation)
        self.expand_conv.build(input_shape)
        x_shape = self.expand_conv.compute_output_shape(input_shape)
        self.expand_norm.build(x_shape)

        # Depthwise: mid_ch → mid_ch (same channels, same spatial with padding="same")
        self.dw_conv = keras.layers.DepthwiseConv2D(
            self.kernel_size, padding="same", use_bias=False
        )
        self.dw_norm = keras.layers.LayerNormalization(axis=norm_axis)
        self.dw_act = keras.layers.Activation(self._activation)
        self.dw_conv.build(x_shape)
        x_shape = self.dw_conv.compute_output_shape(x_shape)
        self.dw_norm.build(x_shape)

        # Squeeze-and-Excite
        if self.se_ratio > 0:
            self.se = SqueezeExcite(self.se_ratio)
            self.se.build(x_shape)
            x_shape = self.se.compute_output_shape(x_shape)
        else:
            self.se = None

        # Project: mid_ch → filters (no activation — linear projection)
        self.project_conv = keras.layers.Conv2D(self.filters, 1, use_bias=False)
        self.project_norm = keras.layers.LayerNormalization(axis=norm_axis)
        self.project_conv.build(x_shape)
        x_shape = self.project_conv.compute_output_shape(x_shape)
        self.project_norm.build(x_shape)
        
        # Residual shortcut
        self.use_residual = in_ch == self.filters
        if self.use_residual:
            self.shortcut = None
        else:
            self.shortcut = keras.layers.Conv2D(self.filters, 1, use_bias=False)
            self.shortcut.build(input_shape)

        super(InvertedResidualBlock, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[_axis_channel()] = self.filters
        return tuple(shape)

    def call(self, x, training=None):
        residual = x

        # Expand → Depthwise → SE → Project
        out = self.expand_act(self.expand_norm(self.expand_conv(x), training=training))
        out = self.dw_act(self.dw_norm(self.dw_conv(out), training=training))
        if self.se is not None:
            out = self.se(out, training=training)
        out = self.project_norm(self.project_conv(out), training=training)

        # Residual connection
        if self.use_residual:
            out = out + residual
        elif self.shortcut is not None:
            out = out + self.shortcut(residual)

        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "expand_ratio": self.expand_ratio,
                "kernel_size": self.kernel_size,
                "se_ratio": self.se_ratio,
                "activation": self._activation,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="agx_core.models.ra")
class UpsampleRefine(keras.layers.Layer):
    """Bilinear 2× upsample + Conv refinement.

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

        self.upsample = keras.layers.UpSampling2D(size=2, interpolation="bilinear")
        self.conv = keras.layers.Conv2D(
            filters, kernel_size, padding="same", use_bias=False
        )
        self.norm = keras.layers.LayerNormalization(axis=_layer_norm_axis())
        self.act = keras.layers.Activation("relu6")

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


@keras.saving.register_keras_serializable(package="agx_core.models.ra")
class FiLM(keras.layers.Layer):
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
        n_channels = feature_shape[_axis_channel()]
        self.gamma = Sequential(
            [
                keras.layers.Conv2D(
                    n_channels, 1, kernel_initializer="zeros", bias_initializer="zeros"
                ),
                keras.layers.GlobalAveragePooling2D(keepdims=True),
            ]
        )
        self.beta = Sequential(
            [
                keras.layers.Conv2D(
                    n_channels, 1, kernel_initializer="zeros", bias_initializer="zeros"
                ),
                keras.layers.GlobalAveragePooling2D(keepdims=True),
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
    "Split",
    "ConvBlock",
    "DeConvBlock",
    "ResidualBlock",
    "Reparameterization",
    "SqueezeExcite",
    "InvertedResidualBlock",
    "UpsampleRefine",
    "FiLM",
]
