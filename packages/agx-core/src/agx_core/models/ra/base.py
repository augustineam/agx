from __future__ import annotations


import keras
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, List, Optional


class BaseEncoder(keras.layers.Layer, ABC):

    def __init__(
        self,
        latent_size: int,
        latent_spatial_res: Optional[Tuple[int]] = None,
        name="base_encoder",
        **kwargs,
    ):
        self._latent_size = latent_size
        self._latent_spatial_res = latent_spatial_res
        super(BaseEncoder, self).__init__(name=name, **kwargs)

    @property
    def latent_size(self):
        return self._latent_size

    @property
    def latent_spatial_res(self):
        return self._latent_spatial_res

    def noise(self, batch_size) -> keras.KerasTensor:
        if self.latent_spatial_res is None:
            raise ValueError(
                "Make sure to build the encoder or to set the correct latent spatial dimension"
            )

        if keras.config.image_data_format() == "channels_last":
            return keras.random.normal(
                (batch_size, *self.latent_spatial_res, self.latent_size)
            )
        return keras.random.normal(
            (batch_size, self.latent_size, *self.latent_spatial_res)
        )

    def compute_output_shape(self, input_shape: Sequence[Sequence[int]]):
        # Calculate the output shape by passing through all blocks
        batch_size = input_shape[0][0]

        if self.latent_spatial_res is None:
            raise ValueError(
                "Make sure to build the encoder or to set the correct latent spatial dimension"
            )

        if keras.config.image_data_format() == "channels_last":
            mu_shape = (batch_size, *self.latent_spatial_res, self.latent_size)
        else:
            mu_shape = (batch_size, self.latent_size, *self.latent_spatial_res)

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
