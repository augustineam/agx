from __future__ import annotations

import keras

from typing import Sequence, Optional

from .base import BaseDecoder
from .layers import _axis_channel, _layer_norm_axis
from .layers import *
from ._spatial import *


def _default_shape():
    if keras.config.image_data_format() == "channels_last":
        return (224, 224, 1)
    return (1, 224, 224)


@keras.saving.register_keras_serializable(package="agx_core.models.ra")
class DecoderStage(keras.layers.Layer):
    """One resolution stage of the decoder.

    FiLM condition → [InvertedResidualBlock × N] → UpsampleRefine 2×

    Each stage refines features at a fixed spatial resolution, then
    upsamples to the next resolution. FiLM conditioning is applied
    before the residual blocks so the decoder can modulate its behavior
    based on product type, acquisition parameters, etc.

    Args:
        filters: Channel width for this stage.
        num_blocks: Number of inverted residual blocks.
        expand_ratio: Expansion ratio for inverted residual blocks.
        kernel_size: Depthwise conv kernel size (3 or 5).
        se_ratio: Squeeze-and-excite ratio.
        upsample: Whether to upsample at the end of this stage.
    """

    def __init__(
        self,
        filters: int,
        num_blocks: int = 2,
        expand_ratio: float = 4.0,
        kernel_size: int = 3,
        se_ratio: float = 0.25,
        upsample: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.num_blocks = num_blocks
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        self.se_ratio = se_ratio
        self._upsample = upsample

        # self.film = FiLM()
        self.blocks = [
            InvertedResidualBlock(
                filters,
                expand_ratio=expand_ratio,
                kernel_size=kernel_size,
                se_ratio=se_ratio,
            )
            for _ in range(num_blocks)
        ]
        self.up = UpsampleRefine(filters) if upsample else keras.layers.Identity()

        # to_rgb head for progressive training — produces a 1-channel
        # (or 3-channel) image from this stage's features. During
        # progressive growing, intermediate stages output through this.
        self.to_rgb = None  # built lazily when progressive training is used

    def build(self, input_shape):
        # feature_shape, _ = input_shape
        # self.film.build([feature_shape, cond_shape])
        x_shape = input_shape

        for block in self.blocks:
            block.build(x_shape)
            x_shape = block.compute_output_shape(x_shape)

        self.up.build(x_shape)
        x_shape = self.up.compute_output_shape(x_shape)

        self._output_shape = x_shape
        super(DecoderStage, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        # feature_shape, _ = input_shape
        x_shape = input_shape
        # Film doesn't change shape
        for block in self.blocks:
            x_shape = block.compute_output_shape(x_shape)
        x_shape = self.up.compute_output_shape(x_shape)
        return x_shape

    def call(self, x, training=None):
        # x, cond = inputs
        # x = self.film([x, cond])
        for block in self.blocks:
            x = block(x, training=training)
        x = self.up(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "num_blocks": self.num_blocks,
                "expand_ratio": self.expand_ratio,
                "kernel_size": self.kernel_size,
                "se_ratio": self.se_ratio,
                "upsample": self._upsample,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="agx_core.models.ra")
class Decoder(BaseDecoder):
    """MobileNetV3-symmetric decoder with FiLM conditioning.

    Architecture:
        Latent z (B, 1, 1, C)
            → ConvTranspose to bottleneck spatial size (e.g., 3×3)
            → [DecoderStage: FiLM → InvRes×N → Upsample 2×] × num_stages
            → Final InvRes refinement
            → Conv 3×3 → output activation

    Design choices vs. the original decoder:
        - Bilinear upsample + conv instead of ConvTranspose
          (eliminates checkerboard artifacts)
        - Inverted residual blocks with SE attention instead of
          standard ResBlocks (3× more expressive per parameter)
        - FiLM conditioning at every stage instead of concat at
          bottleneck (richer product-type / acquisition conditioning)
        - Supports progressive growing for stage-by-stage training

    Args:
        stage_config: List of dicts, one per upsampling stage. Each dict:
            filters (int): Channel width
            num_blocks (int): InvRes blocks at this resolution
            expand_ratio (float): InvRes expansion (default 4.0)
            kernel_size (int): DW conv kernel (default 3)
            se_ratio (float): SE reduction (default 0.25)
        target_shape: Output spatial shape (H, W, C) or (C, H, W).
        output_activation: Final activation ('tanh' or 'sigmoid').
    """

    # Sensible defaults matching MobileNetV3 Small's depth profile
    DEFAULT_STAGE_CONFIG = [
        {"filters": 256, "num_blocks": 1, "expand_ratio": 3.0, "kernel_size": 5},
        {"filters": 96, "num_blocks": 3, "expand_ratio": 3.0, "kernel_size": 5},
        {"filters": 48, "num_blocks": 2, "expand_ratio": 2.0, "kernel_size": 3},
        {"filters": 40, "num_blocks": 3, "expand_ratio": 2.0, "kernel_size": 3},
        {"filters": 24, "num_blocks": 2, "expand_ratio": 2.0, "kernel_size": 3},
        {"filters": 16, "num_blocks": 1, "expand_ratio": 1.0, "kernel_size": 3},
    ]

    def __init__(
        self,
        stage_config: Optional[list[dict]] = None,
        target_shape: Sequence[int] = _default_shape(),
        output_activation: str = "tanh",
        name: str = "decoder",
        **kwargs,
    ):
        if target_shape is None or len(target_shape) != 3:
            raise ValueError("target_shape must be (H, W, C) or (C, H, W)")

        super(Decoder, self).__init__(target_shape=target_shape, name=name, **kwargs)

        self.stage_config = stage_config or self.DEFAULT_STAGE_CONFIG
        self.output_activation = output_activation

        ch_dim = 0 if keras.config.image_data_format() == "channels_first" else -1
        out_ch = target_shape[ch_dim]

        self.concat = keras.layers.Concatenate(_axis_channel())
        # # --- Stem: channel expansion only (spatial already 7×7) ---
        # stem_filters = self.stage_config[0]["filters"]
        # self.stem_conv = keras.layers.Conv2D(
        #     stem_filters, 1, padding="same", use_bias=False
        # )
        # self.stem_norm = keras.layers.LayerNormalization(axis=_layer_norm_axis())
        # self.stem_act = keras.layers.Activation("relu6")

        num_stages = len(self.stage_config)
        # --- Decoder stages ---
        self.stages = []
        for i, cfg in enumerate(self.stage_config):
            is_last = i == (num_stages - 1)
            self.stages.append(
                DecoderStage(
                    filters=cfg["filters"],
                    num_blocks=cfg.get("num_blocks", 2),
                    expand_ratio=cfg.get("expand_ratio", 4.0),
                    kernel_size=cfg.get("kernel_size", 3),
                    se_ratio=cfg.get("se_ratio", 0.25),
                    upsample=not is_last,
                    name=f"stage_{i}",
                )
            )

        # --- Output head ---
        self.head_block = InvertedResidualBlock(
            self.stage_config[-1]["filters"],
            expand_ratio=2.0,
            kernel_size=3,
            se_ratio=0.25,
        )
        self.head_conv = keras.layers.Conv2D(out_ch, 3, padding="same", use_bias=True)
        self.activation = keras.layers.Activation(output_activation)

    def build(self, input_shape):
        x_shape, c_shape = input_shape

        self.concat.build([x_shape, c_shape])
        x_shape = self.concat.compute_output_shape([x_shape, c_shape])

        # self.stem_conv.build(x_shape)
        # x_shape = self.stem_conv.compute_output_shape(x_shape)
        # self.stem_norm.build(x_shape)

        for stage in self.stages:
            stage.build(x_shape)  # ([x_shape, c_shape])
            x_shape = stage.compute_output_shape(x_shape)  # ([x_shape, c_shape])

        self.head_block.build(x_shape)
        x_shape = self.head_block.compute_output_shape(x_shape)
        self.head_conv.build(x_shape)
        x_shape = self.head_conv.compute_output_shape(x_shape)

        super(Decoder, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        x_shape, c_shape = input_shape
        # x_shape = self.stem_conv.compute_output_shape(x_shape)
        x_shape = self.concat.compute_output_shape([x_shape, c_shape])
        for stage in self.stages:
            x_shape = stage.compute_output_shape(x_shape)  # ([x_shape, c_shape])
        x_shape = self.head_block.compute_output_shape(x_shape)
        return self.head_conv.compute_output_shape(x_shape)

    def call(self, inputs, training=None):
        x, cond = inputs

        # Stem: expand to spatial bottleneck
        # x = self.stem_conv(x)
        # x = self.stem_norm(x, training=training)
        # x = self.stem_act(x)
        x = self.concat([x, cond])

        # Progressive decode through stages
        for stage in self.stages:
            x = stage(x, training=training)  # ([x, cond], training=training)

        # Output head
        x = self.head_block(x, training=training)
        x = self.head_conv(x)
        return self.activation(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stage_config": self.stage_config,
                "target_shape": self.target_shape,
                "output_activation": self.output_activation,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="agx_core.models.ra")
class ResNetDecoder(BaseDecoder):
    """ResNetDecoder mapping latent space representations back to images.

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

        super(ResNetDecoder, self).build(input_shape)

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


__all__ = ["Decoder", "ResNetDecoder"]
