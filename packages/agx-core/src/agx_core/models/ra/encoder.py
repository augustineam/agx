from __future__ import annotations

import keras

from typing import Sequence, Optional

from .base import BaseEncoder
from .layers import *
from .layers import _axis_channel


@keras.saving.register_keras_serializable(package="agx_core.models.ra")
class Encoder(BaseEncoder):
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
        latent_size: int = 512,
        name: str = "encoder",
        **kwargs,
    ):
        super(Encoder, self).__init__(latent_size=latent_size, name=name, **kwargs)

        self.filters = filters
        self.blocks: list[keras.layers.Layer] = [
            ConvBlock(filters[0], 5, padding="same", use_bias=False),
            keras.layers.AvgPool2D(2),
        ]

        for f in filters[1:]:
            self.blocks.extend([ResidualBlock(f), keras.layers.AvgPool2D(2)])

        self.blocks.append(ResidualBlock(filters[-1]))
        self.conv: keras.layers.Conv2D = None
        self.concat = keras.layers.Concatenate(_axis_channel())
        self.split = Split(2, axis=_axis_channel())

    def build(self, input_shape: Sequence[Sequence[int]]):
        x_shape, c_shape = input_shape

        for block in self.blocks:
            block.build(x_shape)
            x_shape = block.compute_output_shape(x_shape)

        if keras.config.image_data_format() == "channels_last":
            resolution = x_shape[1:3]
        else:
            resolution = x_shape[2:]
            
        # Map downsampled spatial features directly to 1x1 latent dimensions
        self.conv = keras.layers.Conv2D(
            2 * self.latent_size,
            kernel_size=resolution,
            padding="valid",
        )

        self.concat.build([x_shape, c_shape])
        x_shape = self.concat.compute_output_shape([x_shape, c_shape])

        self.conv.build(x_shape)
        x_shape = self.conv.compute_output_shape(x_shape)
        self.split.build(x_shape)

        super(Encoder, self).build(input_shape)
    
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
        config = super().get_config()
        config.update(
            dict(
                latent_size=self.latent_size,
                filters=self.filters,
            )
        )
        return config


__all__ = ["Encoder"]
