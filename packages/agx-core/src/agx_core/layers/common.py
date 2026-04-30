from __future__ import annotations

import keras

from keras import layers, ops

from typing import Sequence


@keras.saving.register_keras_serializable(package="agx_core.layers")
class Sequential(layers.Layer):
    """
    A keras Layer equivalent of torch.nn.Sequential.
    Composes an ordered list of layers into a single reusable Layer,
    not a Model — so it can be nested inside other layers or models freely.
    Fully serializable via keras.saving.
    """

    def __init__(self, layer_list: list[layers.Layer], name="sequential", **kwargs):
        super().__init__(name=name, **kwargs)
        self._layers = layer_list

    def build(self, input_shape):
        shape = input_shape
        for layer in self._layers:
            layer.build(shape)
            shape = layer.compute_output_shape(shape)
        super(Sequential, self).build(input_shape)

    def call(self, x, training=False):
        for layer in self._layers:
            x = layer(x, training=training)
        return x

    def compute_output_shape(self, input_shape):
        shape = input_shape
        for layer in self._layers:
            shape = layer.compute_output_shape(shape)
        return shape

    def get_config(self):
        base_config = super().get_config()
        layer_configs = [
            keras.saving.serialize_keras_object(layer) for layer in self._layers
        ]
        return {**base_config, "layer_list": layer_configs}

    @classmethod
    def from_config(cls, config):
        layer_list = [
            keras.saving.deserialize_keras_object(cfg)
            for cfg in config.pop("layer_list")
        ]
        return cls(layer_list=layer_list, **config)


@keras.saving.register_keras_serializable(package="agx_core.layers")
class Split(keras.layers.Layer):
    """Splits an input tensor along a specified axis.

    Args:
        num_or_size_splits: Number of equal splits or sizes of each split.
        axis: Axis to split along (default: -1 if 'channels_last', 1 otherwise).
    """

    def __init__(self, num_or_size_splits, axis=-1, name="split", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_or_size_splits = num_or_size_splits
        self.axis = axis

    def build(self, input_shape: Sequence[int]):
        self.input_spec = keras.layers.InputSpec(shape=(None, *input_shape[1:]))
        super(Split, self).build(input_shape)

    def compute_output_shape(self, input_shape: Sequence[int]):
        input_shape = list(input_shape)
        axis = self.axis if self.axis >= 0 else len(input_shape) + self.axis

        if isinstance(self.num_or_size_splits, int):
            # Equal splits
            split_size = input_shape[axis] // self.num_or_size_splits
            output_shapes = []
            for _ in range(self.num_or_size_splits):
                output_shape = input_shape.copy()
                output_shape[axis] = split_size
                output_shapes.append(tuple(output_shape))
        else:
            # Custom split sizes
            output_shapes = []
            for size in self.num_or_size_splits:
                output_shape = input_shape.copy()
                output_shape[axis] = size
                output_shapes.append(tuple(output_shape))

        return output_shapes

    def call(self, inputs):
        return ops.split(inputs, self.num_or_size_splits, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"num_or_size_splits": self.num_or_size_splits, "axis": self.axis}
        )
        return config


__all__ = ["Sequential", "Split"]
