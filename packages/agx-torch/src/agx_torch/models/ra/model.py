import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch

from agx_core.models.ra import ReversedAutoencoderBase

@keras.saving.register_keras_serializable(package="agx_torch.models.ra")
class ReversedAutoencoder(ReversedAutoencoderBase):

    def train_encoder(self, real, noise, condition):
        self.encoder.trainable = True
        self.decoder.trainable = False

        self.zero_grad()
        loss, aux_outputs, metric_updates = self.compute_encoder_loss(real, noise, condition)
        loss.backward()

        # Access .module if DDP-wrapped, otherwise use directly
        enc = self.encoder.module if hasattr(self.encoder, "module") else self.encoder
        trainable_weights = enc.trainable_variables
        grads = [v.value.grad for v in trainable_weights]

        with torch.no_grad():
            self.optimizer.enc.apply(grads, trainable_weights)

        self.update_step_metrics(metric_updates)
        return aux_outputs

    def train_decoder(self, real, noise, condition, z_real, embeds_real, kld_real):
        self.encoder.trainable = False
        self.decoder.trainable = True

        self.zero_grad()
        loss, metric_updates = self.compute_decoder_loss(real, noise, condition, z_real, embeds_real, kld_real)
        loss.backward()

        dec = self.decoder.module if hasattr(self.decoder, "module") else self.decoder
        trainable_weights = dec.trainable_variables
        grads = [v.value.grad for v in trainable_weights]

        with torch.no_grad():
            self.optimizer.dec.apply(grads, trainable_weights)

        self.update_step_metrics(metric_updates)
