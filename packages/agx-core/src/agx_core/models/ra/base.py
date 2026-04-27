from __future__ import annotations


import keras
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, List


class BaseEncoder(keras.layers.Layer, ABC):

    def __init__(self, latent_size: int, name="base_encoder", **kwargs):
        self._latent_size = latent_size
        super(BaseEncoder, self).__init__(name=name, **kwargs)

    @property
    def latent_size(self):
        return self._latent_size

    def noise(self, batch_size) -> keras.KerasTensor:
        if keras.config.image_data_format() == "channels_last":
            return keras.random.normal((batch_size, 1, 1, self.latent_size))
        return keras.random.normal((batch_size, self.latent_size, 1, 1)) 

    def compute_output_shape(self, input_shape: Sequence[Sequence[int]]):
        # Calculate the output shape by passing through all blocks
        batch_size = input_shape[0][0]

        if keras.config.image_data_format() == "channels_last":
            mu_shape = (batch_size, 1, 1, self.latent_size)
        else:
            mu_shape = (batch_size, self.latent_size, 1, 1)

        return mu_shape, mu_shape

    @abstractmethod
    def call(
        self, inputs: Sequence[keras.KerasTensor], training: bool | None = None
    ) -> Tuple[Tuple[keras.KerasTensor, keras.KerasTensor], List[keras.KerasTensor]]:
        """
        Contract:
        Inputs: [image, conditional]
        Outputs: ((mean, logvar), embeddings)
        """
        pass


class BaseDecoder(keras.layers.Layer):

    def __init__(self, target_shape: Sequence[int], name="base_decoder", **kwargs):
        self.target_shape = target_shape
        super(BaseDecoder, self).__init__(name=name, **kwargs)

    @abstractmethod
    def call(
        self, inputs: Sequence[keras.KerasTensor], training: bool | None = None
    ) -> keras.KerasTensor:
        """
        Contract:
        Inputs: [latent_z, conditional]
        Outputs: reconstructed_image
        """
        pass


__all__ = [
    "BaseEncoder",
    "BaseDecoder",
]
