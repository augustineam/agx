from __future__ import annotations

import keras

from typing import Sequence, Optional

from .base import BaseDecoder
from .layers import _axis_channel, _layer_norm_axis
from .layers import *
from ._spatial import *


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
        dropout_rate: float = 0.0,
        **kwargs,
    ):
        super(DecoderStage, self).__init__(**kwargs)
        self.filters = filters
        self.num_blocks = num_blocks
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        self.se_ratio = se_ratio
        self._upsample = upsample
        self.dropout_rate = dropout_rate

        # SpatialDropout at stage entry — drops entire channels,
        # not individual pixels. Forces diverse feature usage.
        self.dropout = (
            keras.layers.SpatialDropout2D(dropout_rate) if dropout_rate > 0 else None
        )

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
        if self.dropout is not None:
            x = self.dropout(x, training=training)
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
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="agx_core.models.ra")
class Decoder(BaseDecoder):
    """MobileNetV3-Small-symmetric decoder.

    Architecture:
        [latent_z, cond] (B, 7, 7, C)
            → Concat → Conv 1×1 stem (mirrors encoder Top conv)
            → [DecoderStage: InvRes×N → Upsample 2×] × 5 stages
            → Final InvRes refinement at full resolution
            → Conv 3×3 → output activation

    Stage layout (exact mirror of MobileNetV3 Small encoder):
        Stem:    513ch → 96ch    Conv 1×1 + LN + h-swish    (encoder Top reversed)
        Stage 0:  7×7   → 14×14  3× InvRes k5 SE  96ch    (encoder S4 reversed)
        Stage 1: 14×14  → 28×28  5× InvRes k5 SE  48ch    (encoder S3 reversed)
        Stage 2: 28×28  → 56×56  2× InvRes k3     24ch    (encoder S2 reversed)
        Stage 3: 56×56  →112×112 1× InvRes k3 SE  16ch    (encoder S1 reversed)
        Stage 4:112×112 →224×224 1× InvRes k3     16ch    (encoder Stem reversed)
        Head:   224×224          InvRes + Conv 3×3 → out

    Design choices:
        - 1×1 stem projection narrows wide latent to stage width
          (prevents expand_ratio blow-up on first InvRes)
        - Bilinear upsample + conv instead of ConvTranspose
          (eliminates checkerboard artifacts)
        - Inverted residual blocks with SE attention instead of
          standard ResBlocks (3× more expressive per parameter)
        - ~1.4M params vs encoder's ~0.9M (1.5× ratio)

    Args:
        stage_config: List of dicts, one per upsampling stage. Each dict:
            filters (int): Channel width
            num_blocks (int): InvRes blocks at this resolution
            expand_ratio (float): InvRes expansion (default 4.0)
            kernel_size (int): DW conv kernel (default 3)
            se_ratio (float): SE reduction (default 0.25)
            upsample (bool): Whether to 2× upsample (default True)
        target_shape: Output spatial shape (H, W, C) or (C, H, W).
        output_activation: Final activation ('tanh' or 'sigmoid').
    """

    # Exact mirror of MobileNetV3 Small encoder (reversed).
    #
    # Encoder (forward):                         Decoder (reverse):
    #   Stem  224→112  Conv 3×3 s2    →16ch       Stage 4  112×112  head
    #   S1    112→56   1× InvRes k3   →16ch       Stage 3   56×56   1× InvRes k3  SE  →16ch
    #   S2     56→28   2× InvRes k3   →24ch       Stage 2   28×28   2× InvRes k3      →24ch
    #   S3     28→14   5× InvRes k5   →48ch       Stage 1   14×14   5× InvRes k5  SE  →48ch
    #   S4     14→7    3× InvRes k5   →96ch       Stage 0    7×7    3× InvRes k5  SE  →96ch
    #   Top    7→7     Conv 1×1       →576ch
    DEFAULT_STAGE_CONFIG = [
        # 7×7 — mirrors encoder Stage 4 (expanded_conv_8..10)
        {
            "filters": 96,
            "num_blocks": 3,
            "expand_ratio": 6.0,
            "kernel_size": 5,
            "se_ratio": 0.25,
            "dropout_rate": 0.2,
        },
        # 14×14 — mirrors encoder Stage 3 (expanded_conv_3..7)
        {
            "filters": 48,
            "num_blocks": 5,
            "expand_ratio": 4.0,
            "kernel_size": 5,
            "se_ratio": 0.25,
            "dropout_rate": 0.15,
        },
        # 28×28 — mirrors encoder Stage 2 (expanded_conv_1..2)
        {
            "filters": 24,
            "num_blocks": 2,
            "expand_ratio": 3.67,
            "kernel_size": 3,
            "se_ratio": 0.0,
            "dropout_rate": 0.1,
        },
        # 56×56 — mirrors encoder Stage 1 (expanded_conv_0)
        {
            "filters": 16,
            "num_blocks": 1,
            "expand_ratio": 1.0,
            "kernel_size": 3,
            "se_ratio": 0.25,
            "dropout_rate": 0.05,
        },
        # 112×112 — mirrors encoder Stem (no upsample, head only)
        {
            "filters": 16,
            "num_blocks": 1,
            "expand_ratio": 1.0,
            "kernel_size": 3,
            "se_ratio": 0.0,
            "dropout_rate": 0.0,
        },
    ]

    def __init__(
        self,
        stage_config: Optional[list[dict]] = None,
        output_activation: str = "tanh",
        name: str = "mbnetv3_decoder",
        **kwargs,
    ):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.stage_config = stage_config or self.DEFAULT_STAGE_CONFIG
        self.output_activation = output_activation

        ch_dim = 0 if keras.config.image_data_format() == "channels_first" else -1
        out_ch = self.target_shape[ch_dim]

        self.concat = keras.layers.Concatenate(_axis_channel())

        # --- Stem: mirrors encoder Top (Conv 1×1 96→576) in reverse ---
        # Projects wide latent+cond channels down to stage_0 width
        # before InvRes blocks see it (prevents expand_ratio blow-up).
        stem_filters = self.stage_config[0]["filters"]
        self.stem_conv = keras.layers.Conv2D(
            stem_filters, 1, padding="same", use_bias=False
        )
        self.stem_norm = keras.layers.LayerNormalization(axis=_layer_norm_axis())
        self.stem_act = keras.layers.Activation("hard_swish")

        # --- Decoder stages ---
        self.stages = []
        for i, cfg in enumerate(self.stage_config):
            self.stages.append(
                DecoderStage(
                    filters=cfg["filters"],
                    num_blocks=cfg.get("num_blocks", 2),
                    expand_ratio=cfg.get("expand_ratio", 4.0),
                    kernel_size=cfg.get("kernel_size", 3),
                    se_ratio=cfg.get("se_ratio", 0.25),
                    upsample=cfg.get("upsample", True),
                    dropout_rate=cfg.get("dropout_rate", 0.0),
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

        self.stem_conv.build(x_shape)
        x_shape = self.stem_conv.compute_output_shape(x_shape)
        self.stem_norm.build(x_shape)

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
        x_shape = self.concat.compute_output_shape([x_shape, c_shape])
        x_shape = self.stem_conv.compute_output_shape(x_shape)
        for stage in self.stages:
            x_shape = stage.compute_output_shape(x_shape)  # ([x_shape, c_shape])
        x_shape = self.head_block.compute_output_shape(x_shape)
        return self.head_conv.compute_output_shape(x_shape)

    def call(self, inputs, training=None):
        x, cond = inputs

        # Stem: project wide latent → narrow stage width
        x = self.concat([x, cond])
        x = self.stem_conv(x)
        x = self.stem_norm(x, training=training)
        x = self.stem_act(x)

        # Progressive decode through stages
        for stage in self.stages:
            x = stage(x, training=training)  # ([x, cond], training=training)

        # Output head
        x = self.head_block(x, training=training)
        x = self.head_conv(x)
        return self.activation(x)

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update(
            {
                "stage_config": self.stage_config,
                "output_activation": self.output_activation,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


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
        output_activation: str = "tanh",
        name: str = "resnet_decoder",
        **kwargs,
    ):
        super(ResNetDecoder, self).__init__(name=name, **kwargs)

        ch_dim = 0 if keras.config.image_data_format() == "channels_first" else -1
        out_ch = self.target_shape[ch_dim]
        self.filters = filters
        self.output_activation = output_activation

        # Compute the exact spatial trajectory the encoder would produce
        if keras.config.image_data_format() == "channels_last":
            h, w = self.target_shape[:2]
        else:
            h, w = self.target_shape[1:]

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
        config = super(ResNetDecoder, self).get_config()
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


__all__ = ["Decoder", "ResNetDecoder"]
