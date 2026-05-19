from __future__ import annotations

import keras

from keras import ops, layers
from typing import Sequence

from agx_core.helpers import _layer_norm_axis


@keras.saving.register_keras_serializable(
    package="agx_core.models.reversed_autoencoder"
)
class Reparameterization(layers.Layer):
    """Samples from latent space using the reparameterization trick: z = μ + σ * ε.

    Input: [z_mean, z_log_var]
    Output: Sampled latent tensor z.
    """

    def __init__(self, name="reparameterization", **kwargs):
        super().__init__(name=name, **kwargs)

    def build(self, input_shape: Sequence[Sequence[int]]):
        super().build(input_shape)

    def compute_output_shape(
        self, input_shape: Sequence[Sequence[int]]
    ) -> Sequence[int]:
        return input_shape[0]

    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = keras.random.normal(shape=ops.shape(z_mean))
        return ops.stop_gradient(eps) * ops.exp(z_log_var * 0.5) + z_mean

    def compute_output_shape(self, input_shape: Sequence[Sequence[int]]):
        return input_shape[0]


@keras.saving.register_keras_serializable(
    package="agx_core.models.reversed_autoencoder"
)
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
        self.norm = layers.BatchNormalization(axis=_layer_norm_axis())
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


__all__ = [
    "Reparameterization",
    "UpsampleRefine",
]
