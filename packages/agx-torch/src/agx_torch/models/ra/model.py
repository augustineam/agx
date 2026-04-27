import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch

from typing import Optional

from agx_core.models.ra.base import BaseEncoder, BaseDecoder
from .layers import Reparameterization

from agx_core.models.ra import ReversedAutoencoderBase


@keras.saving.register_keras_serializable(package="agx_torch.models.ra")
class ReversedAutoencoder(ReversedAutoencoderBase, torch.nn.Module):

    encoder: BaseEncoder
    decoder: BaseDecoder

    def __init__(
        self,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        scale: Optional[float] = None,
        name: str = "reversed_autoencoder",
        **kwargs,
    ):
        super(ReversedAutoencoder, self).__init__(
            encoder, decoder, Reparameterization(), scale=scale, name=name, **kwargs
        )

    def train_encoder(self, real, noise, condition):
        if self.train_encoder_enable:
            self.encoder.trainable = True
            self.decoder.trainable = False

            if self.freeze_backbone and hasattr(self.encoder, "train_backbone"):
                self.encoder.train_backbone(False)

            self.zero_grad()
            loss, aux_outputs, metric_updates = self.compute_encoder_loss(
                real, noise, condition
            )
            loss.backward()

            # Access .module if DDP-wrapped, otherwise use directly
            enc = (
                self.encoder.module if hasattr(self.encoder, "module") else self.encoder
            )

            trainable_vars = enc.trainable_variables
            grads = [v.value.grad for v in trainable_vars]

            with torch.no_grad():
                self.optimizer.enc.apply(grads, trainable_vars)

            self.update_step_metrics(metric_updates)
        else:
            with torch.no_grad():
                _, aux_outputs, metric_updates = self.compute_encoder_loss(
                    real, noise, condition
                )
            self.update_step_metrics(metric_updates)
        return aux_outputs

    def train_decoder(self, real, noise, condition, z_real, embeds_real, kld_real):
        if self.train_decoder_enabled:
            self.encoder.trainable = False
            self.decoder.trainable = True

            self.zero_grad()
            loss, metric_updates = self.compute_decoder_loss(
                real, noise, condition, z_real, embeds_real, kld_real
            )
            loss.backward()

            dec = (
                self.decoder.module if hasattr(self.decoder, "module") else self.decoder
            )

            trainable_vars = dec.trainable_variables
            grads = [v.value.grad for v in trainable_vars]

            with torch.no_grad():
                self.optimizer.dec.apply(grads, trainable_vars)

            self.update_step_metrics(metric_updates)
        else:
            with torch.no_grad():
                _, metric_updates = self.compute_decoder_loss(
                    real, noise, condition, z_real, embeds_real, kld_real
                )
            self.update_step_metrics(metric_updates)
