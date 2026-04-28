import keras
from keras import layers


@keras.saving.register_keras_serializable(package="agx_core.layers")
class Sequential(layers.Layer):
    """
    A keras Layer equivalent of torch.nn.Sequential.
    Composes an ordered list of layers into a single reusable Layer,
    not a Model — so it can be nested inside other layers or models freely.
    Fully serializable via keras.saving.
    """

    def __init__(self, layer_list: list[layers.Layer], **kwargs):
        super().__init__(**kwargs)
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


__all__ = ["Sequential"]
