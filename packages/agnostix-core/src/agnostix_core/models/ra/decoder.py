from __future__ import annotations

import keras

from typing import Sequence, Optional

from .base import BaseDecoder
from .layers import *


@keras.saving.register_keras_serializable(package="agnostix_core.models.ra")
class Decoder(BaseDecoder):
    """Decoder mapping latent space representations back to images.

    Fully Convolutional Network (FCN) architecture:
    latent -> [concat cond] -> conv_transpose -> lrelu -> [resblock -> deconv]*N -> resblock -> conv -> sigmoid

    Args:
        filters: Filter sizes for each deconvolutional stage.
        target_shape: Spatial shape (height, width, channels) of the output image.
        name: Layer name.
    """

    def __init__(
        self,
        filters: Sequence[int] = [512, 512, 512, 256, 128, 64],
        target_shape: Sequence[int] = (256, 256, 1),
        name: str = "decoder",
        **kwargs,
    ):
        if target_shape is None:
            raise ValueError("target_shape must be provided")

        if len(target_shape) != 3:
            raise ValueError(
                "target_shape must be a sequence of length 3 (height, width, channels)"
            )

        super().__init__(target_shape=target_shape, name=name, **kwargs)

        out_ch = target_shape[-1]
        self.filters = filters

        # Calculate the initial resolution after the first conv transpose
        # This should be target_shape // (2 ** len(filters))
        strides = 2 ** len(filters)
        initial_resolution = (target_shape[0] // strides, target_shape[1] // strides)

        self.concat = keras.layers.Concatenate()

        # Map 1x1 latent feature to the lowest spatial resolution
        self.conv_transpose = keras.layers.Conv2DTranspose(
            filters[0],
            kernel_size=initial_resolution,
            strides=1,
            padding="valid",
        )
        self.lrelu = keras.layers.LeakyReLU(0.2)

        self.blocks: list[keras.layers.Layer] = []
        for f in filters:
            self.blocks.extend([ResidualBlock(f), DeConvBlock(f, 2, 2)])

        self.blocks.extend(
            [ResidualBlock(filters[-1]), keras.layers.Conv2D(out_ch, 5, padding="same")]
        )

    def call(
        self,
        inputs: Sequence[keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        x, c = inputs

        x = self.conv_transpose(x)
        x = self.lrelu(x)
        x = self.concat([x, c])

        for block in self.blocks:
            x = block(x, training=training)
        return keras.activations.sigmoid(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                filters=self.filters,
                target_shape=self.target_shape,
            )
        )
        return config

    def build_graph(self, inputs: Sequence[keras.KerasTensor]):
        x, c = inputs
        x = self.conv_transpose(x)
        x = self.lrelu(x)
        x = self.concat([x, c])

        for block in self.blocks:
            x = block(x)
        return keras.activations.sigmoid(x)


__all__ = ["Decoder"]
