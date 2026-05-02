from __future__ import annotations

import keras

from typing import Sequence, Optional

from agx_core.models.reversed_autoencoder.base import BaseEncoder
from agx_core.models.reversed_autoencoder.layers import *
from agx_core.layers import Split
from agx_core.helpers import _channel_axis, _spatial_slice


@keras.saving.register_keras_serializable(
    package="agx_core.models.reversed_autoencoder"
)
class ResNetEncoder(BaseEncoder):
    """Encoder mapping images to latent mean and log-variance parameters.

    Fully Convolutional Network (FCN) architecture:
    input -> conv -> pool -> [resblock -> pool]*N -> resblock -> global_conv -> split

    Args:
        filters: Filter sizes for each convolutional stage.
        latent_size: Size of the latent space.
        name: Layer name.
    """

    def __init__(
        self,
        filters: Sequence[int] = [64, 128, 256, 512, 512, 512],
        name: str = "resnet_encoder",
        **kwargs,
    ):
        super(ResNetEncoder, self).__init__(name=name, **kwargs)
        self.filters = filters

    def build(self, input_shape: Sequence[Sequence[int]]):
        x_shape, c_shape = input_shape

        h, w = x_shape[_spatial_slice()]
        num_stages = len(self.filters)
        resolution = (h // 2**num_stages, w // 2**num_stages)
        self._latent_spatial_res = (1, 1)

        self.blocks: list[keras.layers.Layer] = [
            ConvBlock(self.filters[0], 5, padding="same", use_bias=False),
            keras.layers.AvgPool2D(2),
        ]

        for f in self.filters[1:]:
            self.blocks.extend([ResidualBlock(f), keras.layers.AvgPool2D(2)])

        self.blocks.append(ResidualBlock(self.filters[-1]))

        self.conv = keras.layers.Conv2D(
            2 * self.latent_size,
            kernel_size=resolution,
            padding="valid",
        )
        self.concat = keras.layers.Concatenate(_channel_axis())
        self.split = Split(2, axis=_channel_axis())

        for block in self.blocks:
            block.build(x_shape)
            x_shape = block.compute_output_shape(x_shape)

        self.concat.build([x_shape, c_shape])
        x_shape = self.concat.compute_output_shape([x_shape, c_shape])

        self.conv.build(x_shape)
        x_shape = self.conv.compute_output_shape(x_shape)
        self.split.build(x_shape)

        super(ResNetEncoder, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        x_shape, _ = input_shape

        B = x_shape[0]
        channel_axis = _channel_axis()
        spatial_slice = _spatial_slice()

        latent_shape = [B, 1, 1, 1]
        latent_shape[channel_axis] = self.latent_size
        latent_shape = tuple(latent_shape)

        features_shape = []
        h, w = x_shape[spatial_slice]

        for f in self.filters:
            shape = [B, 0, 0, 0]
            h, w = h // 2, w // 2
            shape[channel_axis] = f
            shape[spatial_slice] = [h, w]
            features_shape.append(tuple(shape))

        return (latent_shape, latent_shape), features_shape

    def call(
        self,
        inputs: Sequence[keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> Sequence[keras.KerasTensor]:
        x, c = inputs

        embeddings = []
        for block in self.blocks:
            x = block(x, training=training)
            if isinstance(block, keras.layers.AvgPool2D):
                embeddings.append(x)

        x = self.concat([x, c])
        x = self.conv(x)
        mean, logvar = self.split(x)

        return (mean, logvar), embeddings

    def get_config(self):
        config = super(ResNetEncoder, self).get_config()
        config.update(dict(filters=self.filters))
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


__all__ = ["ResNetEncoder"]
