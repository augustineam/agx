import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf

from typing import Dict, Any

from agx_core.models.reversed_autoencoder.base import BaseEncoder, BaseDecoder
from agx_core.models.reversed_autoencoder.layers import Reparameterization
from agx_core.models.reversed_autoencoder import ReversedAutoencoderBase


@keras.saving.register_keras_serializable(package="agx_tf.models.reversed_autoencoder")
class ReversedAutoencoder(ReversedAutoencoderBase):

    encoder: BaseEncoder
    decoder: BaseDecoder

    def __init__(
        self,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        reparameterize=Reparameterization(),
        beta_kld: float = 0.25,
        expelbo_temp: float = 1.0,
        lambda_embed: float = 1.0,
        name: str = "reversed_autoencoder",
        **kwargs,
    ):
        super().__init__(
            encoder,
            decoder,
            reparameterize,
            beta_kld=beta_kld,
            expelbo_temp=expelbo_temp,
            lambda_embed=lambda_embed,
            name=name,
            **kwargs,
        )

    def train_encoder(self, real, noise, condition):
        if self.train_encoder_enabled:
            self.encoder.trainable = True
            self.decoder.trainable = False

            if self.freeze_backbone and hasattr(self.encoder, "train_backbone"):
                self.encoder.train_backbone(False)

            with tf.GradientTape() as tape:
                loss, aux_outputs, metric_updates = self.compute_encoder_loss(
                    real, noise, condition
                )

            grads = tape.gradient(loss, self.encoder.trainable_variables)
            self.enc_optimizer.apply_gradients(
                zip(grads, self.encoder.trainable_variables)
            )

            self.update_step_metrics(metric_updates)
        else:
            self.encoder.trainable = False
            self.decoder.trainable = False

            _, aux_outputs, metric_updates = self.compute_encoder_loss(
                real, noise, condition
            )
            self.update_step_metrics(metric_updates)

        return aux_outputs

    def train_decoder(self, real, noise, condition, z_real, embeds_real, kld_real):
        if self.train_decoder_enabled:
            self.encoder.trainable = False
            self.decoder.trainable = True

            with tf.GradientTape() as tape:
                loss, metric_updates = self.compute_decoder_loss(
                    real, noise, condition, z_real, embeds_real, kld_real
                )

            grads = tape.gradient(loss, self.decoder.trainable_variables)
            self.dec_optimizer.apply_gradients(
                zip(grads, self.decoder.trainable_variables)
            )

            self.update_step_metrics(metric_updates)
        else:
            self.encoder.trainable = False
            self.decoder.trainable = False

            _, metric_updates = self.compute_decoder_loss(
                real, noise, condition, z_real, embeds_real, kld_real
            )
            self.update_step_metrics(metric_updates)

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        encoder = keras.saving.deserialize_keras_object(config.pop("encoder"))
        decoder = keras.saving.deserialize_keras_object(config.pop("decoder"))
        reparameterize = keras.saving.deserialize_keras_object(
            config.pop("reparameterize")
        )
        return cls(encoder, decoder, reparameterize, **config)
