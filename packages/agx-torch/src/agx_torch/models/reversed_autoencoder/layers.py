from __future__ import annotations

import keras
import torch

from keras import ops

from typing import Sequence


@keras.saving.register_keras_serializable(package="agx_torch.models.reversed_autoencoder")
class Reparameterization(keras.layers.Layer):
    """Samples from latent space using the reparameterization trick: z = μ + σ * ε.

    Input: [z_mean, z_log_var]
    Output: Sampled latent tensor z.
    """

    def __init__(self, name="reparameterization", **kwargs):
        super().__init__(name=name, **kwargs)

    def build(self, input_shape: Sequence[Sequence[int]]):
        super().build(input_shape)

    def compute_output_shape(
        self, input_shape: Sequence[Sequence[int]]
    ) -> Sequence[int]:
        return input_shape[0]

    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = torch.rand_like(z_mean)
        return eps * ops.exp(z_log_var * 0.5) + z_mean

    def compute_output_shape(self, input_shape: Sequence[Sequence[int]]):
        return input_shape[0]


__all__ = [
    "Reparameterization",
]
