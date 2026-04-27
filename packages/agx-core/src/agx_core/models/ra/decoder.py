from __future__ import annotations

import keras

from typing import Sequence, Optional

from .base import BaseDecoder
from .layers import _axis_channel
from .layers import *
from ._spatial import *


def _default_shape():
    if keras.config.image_data_format() == "channels_last":
        return (224, 224, 1)
    return (1, 224, 224)


@keras.saving.register_keras_serializable(package="agx_core.models.ra")
class Decoder(BaseDecoder):
    """Decoder mapping latent space representations back to images.

    Fully Convolutional Network (FCN) architecture:
    latent -> [concat cond] -> conv_transpose -> lrelu -> [resblock -> deconv]*N -> resblock -> conv -> sigmoid

    Args:
        filters: Filter sizes for each deconvolutional stage.
        target_shape: Spatial shape (height, width, channels) of the output image if 'channels_last'.
        name: Layer name.
    """

    def __init__(
        self,
        filters: Sequence[int] = [512, 512, 512, 256, 128, 64],
        target_shape: Sequence[int] = _default_shape(),
        output_activation: str = "tanh",
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

        ch_dim = 0 if keras.config.image_data_format() == "channels_first" else -1
        out_ch = target_shape[ch_dim]
        self.filters = filters
        self.output_activation = output_activation

        # Compute the exact spatial trajectory the encoder would produce
        if keras.config.image_data_format() == "channels_last":
            h, w = target_shape[:2]
        else:
            h, w = target_shape[1:]

        trajectory = compute_spatial_trajectory(h, w, len(filters))
        output_paddings = compute_output_paddings(trajectory)

        # The bottleneck spatial size (what the encoder actually produces)
        bottleneck_h, bottleneck_w = trajectory[-1]

        self.concat = keras.layers.Concatenate(_axis_channel())

        # Map 1x1 latent feature to the lowest spatial resolution
        self.conv_transpose = keras.layers.Conv2DTranspose(
            filters[0],
            kernel_size=(bottleneck_h, bottleneck_w),
            strides=1,
            padding="valid",
        )
        self.lrelu = keras.layers.LeakyReLU(0.2)

        # Build upsampling blocks with correct output_padding per stage
        self.blocks: list[keras.layers.Layer] = []
        for f, out_pad in zip(filters, output_paddings):
            self.blocks.append(ResidualBlock(f))
            # Convert (0,0) padding to None for clean serialization
            op = tuple(out_pad) if any(p > 0 for p in out_pad) else None
            self.blocks.append(
                DeConvBlock(f, 2, strides=2, padding="valid", output_padding=op)
            )

        self.blocks.extend(
            [ResidualBlock(filters[-1]), keras.layers.Conv2D(out_ch, 5, padding="same")]
        )
        self.activation = keras.layers.Activation(output_activation)

    def build(self, input_shape: Sequence[Sequence[int]]):
        x_shape, c_shape = input_shape
        
        self.conv_transpose.build(x_shape)
        x_shape = self.conv_transpose.compute_output_shape(x_shape)
        
        self.concat.build([x_shape, c_shape])
        x_shape = self.concat.compute_output_shape([x_shape, c_shape])

        for block in self.blocks:
            block.build(x_shape)
            x_shape = block.compute_output_shape(x_shape)
        
        super(Decoder, self).build(input_shape)

    def compute_output_shape(self, input_shape: Sequence[Sequence[int]]):
        x_shape, c_shape = input_shape
        x_shape = self.conv_transpose.compute_output_shape(x_shape)
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

        x = self.conv_transpose(x)
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
                target_shape=self.target_shape,
                output_activation=self.output_activation,
            )
        )
        return config


__all__ = ["Decoder"]
