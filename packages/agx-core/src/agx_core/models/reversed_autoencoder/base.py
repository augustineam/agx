from __future__ import annotations


import keras
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, List, Optional


class BaseEncoder(keras.layers.Layer, ABC):

    def __init__(
        self,
        latent_size: int = 512,
        latent_spatial_res: Optional[Tuple[int]] = None,
        name="base_encoder",
        **kwargs,
    ):
        self._latent_size = latent_size
        self._latent_spatial_res = latent_spatial_res
        super().__init__(name=name, **kwargs)

    @property
    def latent_size(self):
        return self._latent_size

    @property
    def latent_spatial_res(self):
        return self._latent_spatial_res

    def training_enabled(self, training: bool):
        self.trainable = training

    @abstractmethod
    def call(
        self, inputs: Sequence[keras.KerasTensor], training: bool | None = None
    ) -> Tuple[Tuple[keras.KerasTensor], keras.KerasTensor]:
        """
        Contract:
        Inputs: [image, conditional]
        Outputs: (mean, logvar), *embeddings
        """
        pass

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                latent_size=self.latent_size, latent_spatial_res=self.latent_spatial_res
            )
        )
        return config


class BaseDecoder(keras.layers.Layer):

    def __init__(
        self,
        target_shape: Sequence[int] = (224, 224, 1),
        name="base_decoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        if target_shape is None or len(target_shape) != 3:
            raise ValueError("target_shape must be (H, W, C) or (C, H, W)")

        self.target_shape = target_shape

    def training_enabled(self, training: bool):
        self.trainable = training

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

    def get_config(self):
        config = super().get_config()
        config.update(dict(target_shape=self.target_shape))
        return config


__all__ = [
    "BaseEncoder",
    "BaseDecoder",
]
