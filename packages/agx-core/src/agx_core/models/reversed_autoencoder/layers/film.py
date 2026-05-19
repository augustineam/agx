import keras

from keras import layers

from agx_core.layers import Sequential, Split
from agx_core.helpers import _channel_axis, _spatial_slice


@keras.saving.register_keras_serializable(
    package="agx_core.models.reversed_autoencoder"
)
class FiLM(layers.Layer):
    """Feature-wise Linear Modulation for conditioning.

    Applies per-channel affine transformation conditioned
    on an external vector:

        output = features * (1 + γ) + β

    where γ, β are predicted from the conditioning vector.
    Initializes to identity transform (γ=0, β=0).

    Args:
        None — adapts to input shapes at build time.
    """

    def __init__(
        self,
        name: str = "film_conditioning",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        features_shape, cond_shape = input_shape

        ch_axis = _channel_axis()

        feature_dim = features_shape[ch_axis]
        embed_dim = cond_shape[-1]

        proj_reshape = list(features_shape)
        proj_reshape[ch_axis] = 2 * feature_dim
        proj_reshape[_spatial_slice()] = [1, 1]

        self.projection = Sequential(
            layers.Dense(embed_dim, activation="relu"),
            layers.Dense(2 * feature_dim),
            layers.Reshape(tuple(proj_reshape[1:])),
            Split(2, axis=ch_axis),
            name="film_projection",
        )

        self.projection.build(cond_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs, training=None):
        features, cond = inputs

        b, g = self.projection(cond, training=training)

        return features * (1.0 + g) + b


__all__ = ["FiLM"]
