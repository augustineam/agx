from __future__ import annotations

import keras

from keras import layers

from agx_core.helpers import _channel_axis
from ._helpers import _depth


@keras.saving.register_keras_serializable(package="agx_core.models.mobilenet_v3")
class SqueezeExcite(layers.Layer):
    """Squeeze-and-Excite channel attention.

    Global pool → reduce (ReLU) → expand → hard-sigmoid gate.

    Args:
        se_ratio: Reduction ratio relative to input channels.
    """

    def __init__(self, se_ratio: float = 0.25, name="squeeze_and_excite", **kwargs):
        super().__init__(name=name, **kwargs)
        self.se_ratio = se_ratio

    def build(self, input_shape):
        ch = input_shape[_channel_axis()]
        se_ch = _depth(int(ch * self.se_ratio))

        self.pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.reduce = layers.Conv2D(se_ch, 1, padding="same")
        self.relu = layers.Activation("relu")
        self.expand = layers.Conv2D(ch, 1, padding="same")
        self.gate = layers.Activation("hard_sigmoid")
        self.mul = layers.Multiply()

        self.pool.build(input_shape)
        x_shape = self.pool.compute_output_shape(input_shape)
        self.reduce.build(x_shape)
        x_shape = self.reduce.compute_output_shape(x_shape)
        self.expand.build(x_shape)

        super().build(input_shape)

    def call(self, x, training=None):
        scale = self.pool(x)
        scale = self.reduce(scale, training=training)
        scale = self.relu(scale)
        scale = self.expand(scale, training=training)
        scale = self.gate(scale)
        return self.mul([x, scale])

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"se_ratio": self.se_ratio})
        return config


__all__ = [
    "SqueezeExcite",
]
