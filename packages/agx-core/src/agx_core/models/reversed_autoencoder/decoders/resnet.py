from __future__ import annotations

import keras

from typing import Sequence, Optional

from agx_core.helpers import _channel_axis

from agx_core.models.reversed_autoencoder.base import BaseDecoder
from agx_core.models.reversed_autoencoder.layers import *
from agx_core.models.reversed_autoencoder._spatial import *


@keras.saving.register_keras_serializable(
    package="agx_core.models.reversed_autoencoder"
)
class ResNetDecoder(BaseDecoder):
    """ResNetDecoder mapping latent space representations back to images.

    Fully Convolutional Network (FCN) architecture:
    latent -> [concat cond] -> from_latent -> lrelu -> [resblock -> deconv]*N -> resblock -> conv -> sigmoid

    Args:
        filters: Filter sizes for each deconvolutional stage.
        target_shape: Spatial shape (height, width, channels) of the output image if 'channels_last'.
        name: Layer name.
    """

    def __init__(
        self,
        filters: Sequence[int] = [512, 512, 512, 256, 128, 64],
        output_activation: str = "tanh",
        name: str = "resnet_decoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        ch_dim = 0 if keras.config.image_data_format() == "channels_first" else -1
        self.out_ch = self.target_shape[ch_dim]
        self.filters = filters
        self.output_activation = output_activation

    def build(self, input_shape: Sequence[Sequence[int]]):
        x_shape, c_shape = input_shape

        # Compute the exact spatial trajectory the encoder would produce
        if keras.config.image_data_format() == "channels_last":
            h, w = self.target_shape[:2]
        else:
            h, w = self.target_shape[1:]

        trajectory = compute_spatial_trajectory(h, w, len(self.filters))
        output_paddings = compute_output_paddings(trajectory)

        # The bottleneck spatial size (what the encoder actually produces)
        bottleneck_h, bottleneck_w = trajectory[-1]

        self.concat = keras.layers.Concatenate(_channel_axis())

        # Map 1x1 latent feature to the lowest spatial resolution
        self.from_latent = keras.layers.Conv2DTranspose(
            self.filters[0],
            kernel_size=(bottleneck_h, bottleneck_w),
            strides=(1, 1),
            padding="valid",
        )
        self.lrelu = keras.layers.LeakyReLU(0.2)

        # Build upsampling blocks with correct output_padding per stage
        self.blocks: list[keras.layers.Layer] = []
        for f, out_pad in zip(self.filters, output_paddings):
            self.blocks.append(ResidualBlock(f))
            # Convert (0,0) padding to None for clean serialization
            op = tuple(out_pad) if any(p > 0 for p in out_pad) else None
            self.blocks.append(
                DeConvBlock(f, 2, strides=2, padding="valid", output_padding=op)
            )

        self.blocks.extend(
            [
                ResidualBlock(self.filters[-1]),
                keras.layers.Conv2D(self.out_ch, 5, padding="same"),
            ]
        )
        self.activation = keras.layers.Activation(self.output_activation)

        self.from_latent.build(x_shape)
        x_shape = self.from_latent.compute_output_shape(x_shape)

        self.concat.build([x_shape, c_shape])
        x_shape = self.concat.compute_output_shape([x_shape, c_shape])

        for block in self.blocks:
            block.build(x_shape)
            x_shape = block.compute_output_shape(x_shape)

        super().build(input_shape)

    def compute_output_shape(self, input_shape: Sequence[Sequence[int]]):
        x_shape, c_shape = input_shape
        x_shape = self.from_latent.compute_output_shape(x_shape)
        x_shape = self.concat.compute_output_shape([x_shape, c_shape])

        for block in self.blocks:
            x_shape = block.compute_output_shape(x_shape)
        return x_shape

    def call(
        self,
        inputs: Sequence[keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        x, c = inputs

        x = self.from_latent(x, training=training)
        x = self.lrelu(x)
        x = self.concat([x, c])

        for block in self.blocks:
            x = block(x, training=training)
        return self.activation(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                filters=self.filters,
                output_activation=self.output_activation,
            )
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


__all__ = ["ResNetDecoder"]
