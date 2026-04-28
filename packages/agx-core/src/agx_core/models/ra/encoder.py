from __future__ import annotations

import keras

from typing import Sequence, Optional

from agx_core.models.ra.base import BaseEncoder
from agx_core.models.ra.layers import *
from agx_core.models.ra.layers import _axis_channel
from agx_core.layers import Sequential


@keras.saving.register_keras_serializable(package="agx_core.models.ra")
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

        super(ResNetEncoder, self).build(input_shape)

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


@keras.saving.register_keras_serializable(package="agx_core.models.ra")
class Encoder(BaseEncoder):

    def __init__(
        self,
        backbone: Sequence[keras.layers.Layer],
        name="mbnetv3_encoder",
        **kwargs,
    ):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.backbone = backbone
        self.gray2rgb = Sequential(
            [
                keras.layers.Conv2D(
                    3, kernel_size=(3, 3), strides=(1, 1), padding="same"
                ),
                keras.layers.LayerNormalization(),
                keras.layers.Activation("tanh"),
            ]
        )
        self.concat = keras.layers.Concatenate(_axis_channel())
        self.to_latent = keras.layers.Conv2D(self.latent_size * 2, 1)
        self.split = Split(2, axis=_axis_channel())

    def build(self, input_shape: Sequence[Sequence[int]]):
        x_shape, c_shape = input_shape

        if keras.config.image_data_format() == "channels_last":
            spatial = slice(1, 3)
        else:
            spatial = slice(2, None)

        self.gray2rgb.build(x_shape)

        for block in self.backbone:
            h, w = x_shape[spatial]
            block.build(x_shape)
            x_shape = list(x_shape)
            x_shape[spatial] = [h // 2, w // 2]

        resolution = x_shape[spatial]
        if keras.config.image_data_format() == "channels_last":
            x_shape = (x_shape[0], *resolution, 576)
        else:
            x_shape = (x_shape[0], 576, *resolution)

        self._latent_spatial_res = tuple(resolution)
        self.concat.build([x_shape, c_shape])
        x_shape = self.concat.compute_output_shape([x_shape, c_shape])

        self.to_latent.build(x_shape)
        x_shape = self.to_latent.compute_output_shape(x_shape)

        self.split.build(x_shape)

        super(Encoder, self).build(input_shape)

    def train_backbone(self, train: bool):
        for layer in self.backbone:
            layer.trainable = train

    def call(
        self,
        inputs: Sequence[keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> Sequence[keras.KerasTensor]:
        x, c = inputs

        x = self.gray2rgb(x, training=training)
        embeddings = []
        for layer in self.backbone:
            x = layer(x, training=training)
            embeddings.append(x)

        x = self.concat([x, c])
        x = self.to_latent(x)

        mean, logvar = self.split(x)
        return (mean, logvar), embeddings

    def get_config(self):
        config = super(Encoder, self).get_config()
        backbone = [
            keras.saving.serialize_keras_object(layer) for layer in self.backbone
        ]
        config.update(dict(backbone=backbone))
        return config

    @classmethod
    def from_config(cls, config):
        backbone = [
            keras.saving.deserialize_keras_object(cfg) for cfg in config.pop("backbone")
        ]
        return cls(backbone, **config)


__all__ = ["Encoder", "ResNetEncoder"]
