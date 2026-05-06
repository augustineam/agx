import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch

from keras import ops
from torch.utils.checkpoint import checkpoint

from typing import Dict, Any

from agx_core.models.reversed_autoencoder import ReversedAutoencoderBase
from agx_core.models.reversed_autoencoder.base import BaseEncoder, BaseDecoder
from agx_torch.models.reversed_autoencoder.layers import Reparameterization


@keras.saving.register_keras_serializable(
    package="agx_torch.models.reversed_autoencoder"
)
class ReversedAutoencoder(ReversedAutoencoderBase, torch.nn.Module):

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
        super(ReversedAutoencoder, self).__init__(
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

            self.zero_grad()
            loss, metric_updates = checkpoint(
                self.compute_encoder_loss,
                real,
                noise,
                condition,
                use_reentrant=False,
            )
            loss.backward()

            # Access .module if DDP-wrapped, otherwise use directly
            enc = (
                self.encoder.module if hasattr(self.encoder, "module") else self.encoder
            )

            trainable_vars = enc.trainable_variables
            grads = [v.value.grad for v in trainable_vars]

            with torch.no_grad():
                self.enc_optimizer.apply(grads, trainable_vars)

            self.update_step_metrics(metric_updates)
        else:
            with torch.no_grad():
                _, metric_updates = self.compute_encoder_loss(
                    real,
                    noise,
                    condition,
                    use_reentrant=False,
                )
            self.update_step_metrics(metric_updates)

    def train_decoder(self, real, noise, condition):
        if self.train_decoder_enabled:
            self.encoder.trainable = False
            self.decoder.trainable = True

            self.zero_grad()
            loss, metric_updates = self.compute_decoder_loss(real, noise, condition)
            loss.backward()

            dec = (
                self.decoder.module if hasattr(self.decoder, "module") else self.decoder
            )

            trainable_vars = dec.trainable_variables
            grads = [v.value.grad for v in trainable_vars]

            with torch.no_grad():
                self.dec_optimizer.apply(grads, trainable_vars)

            self.update_step_metrics(metric_updates)
        else:
            with torch.no_grad():
                _, metric_updates = self.compute_decoder_loss(real, noise, condition)
            self.update_step_metrics(metric_updates)

    # def train_step(self, data):
    #     (batch_real, batch_cond), _ = data

    #     batch_size = ops.shape(batch_real)[0]
    #     batch_noise = self.noise(batch_size)
    #     batch_real = self.resize_progressive_output(batch_real)

    #     self.train_encoder(batch_real, batch_noise, batch_cond)

    #     # Force PyTorch to release cached blocks before decoder allocation
    #     torch.cuda.empty_cache()

    #     self.train_decoder(batch_real, batch_noise, batch_cond)

    #     return self.get_metrics_result()

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        encoder = keras.saving.deserialize_keras_object(config.pop("encoder"))
        decoder = keras.saving.deserialize_keras_object(config.pop("decoder"))
        reparameterize = keras.saving.deserialize_keras_object(
            config.pop("reparameterize")
        )
        return cls(encoder, decoder, reparameterize, **config)
