import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
import gc

from torch.utils.checkpoint import checkpoint

from keras import ops
from typing import Dict, Any, Optional

from agx_core.models.reversed_autoencoder import ReversedAutoencoderBase
from agx_core.models.reversed_autoencoder.base import BaseEncoder, BaseDecoder
from agx_torch.models.reversed_autoencoder.layers import Reparameterization


def _detach_clone(tensor):
    """Detach AND clone — severs both graph AND storage reference."""
    return tensor.detach().clone()


def _detach_clone_list(tensors):
    """Detach+clone a list of tensors."""
    return [t.detach().clone() for t in tensors]


@keras.saving.register_keras_serializable(
    package="agx_torch.models.reversed_autoencoder"
)
class ReversedAutoencoder(ReversedAutoencoderBase, torch.nn.Module):
    """PyTorch backend implementation of the Reversed Autoencoder.

    Supports optional multi-GPU training via ``place_on_devices()``:
    encoder and its optimizer live on one GPU, decoder and its optimizer
    on another. Tensors are moved across the device boundary as needed.
    When ``place_on_devices()`` is not called, everything stays on the
    default device (single-GPU mode) and the base class's default
    orchestration is used.
    """

    encoder: BaseEncoder
    decoder: BaseDecoder

    def __init__(
        self,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        reparameterize=Reparameterization(),
        beta_kld: float = 0.25,
        enc_expkld_temp: float = 1.0,
        dec_expelbo_temp: float = 1.0,
        spatial_temperature: float = 1.0,
        lambda_embed: float = 1.0,
        alpha_ssim: float = 0.3,
        ssim_kwargs: Optional[Dict[str, Any]] = None,
        name: str = "reversed_autoencoder",
        **kwargs,
    ):
        super().__init__(
            encoder,
            decoder,
            reparameterize,
            beta_kld=beta_kld,
            enc_expkld_temp=enc_expkld_temp,
            dec_expelbo_temp=dec_expelbo_temp,
            spatial_temperature=spatial_temperature,
            lambda_embed=lambda_embed,
            alpha_ssim=alpha_ssim,
            ssim_kwargs=ssim_kwargs,
            name=name,
            **kwargs,
        )

    def _apply_encoder_gradients(self):
        trainable_vars = self.encoder.trainable_variables
        grads = [
            v.value.grad if isinstance(v, keras.Variable) else v for v in trainable_vars
        ]
        with torch.no_grad():
            self.enc_optimizer.apply(grads, trainable_vars)

    def _apply_decoder_gradients(self):
        trainable_vars = self.decoder.trainable_variables
        grads = [
            v.value.grad if isinstance(v, keras.Variable) else v for v in trainable_vars
        ]
        with torch.no_grad():
            self.dec_optimizer.apply(grads, trainable_vars)

    def train_step(self, data):

        (batch_real, batch_cond), _ = data

        batch_size = batch_real.shape[0]
        batch_noise = self.noise(batch_size)
        batch_real = self.resize_progressive_output(batch_real)

        # 1. Train VAE Collaborative
        self.zero_grad()

        loss_1, z_real, embeds_real, metrics_1 = self.train_collaborative(
            batch_real, batch_cond, training=True
        )
        loss_1.backward()

        self._apply_encoder_gradients()
        self._apply_decoder_gradients()
        self.update_step_metrics(metrics_1)

        z_real = _detach_clone(z_real)
        embeds_real = _detach_clone_list(embeds_real)
        del loss_1, metrics_1

        self.zero_grad()
        if self.train_decoder_enabled:
            # 2. Train Decoder on fake path
            loss_2, fake, metrics_2 = self.train_decoder_fake_path(
                batch_noise, batch_cond, training=True
            )
            loss_2.backward()

            fake = _detach_clone(fake)
            del loss_2

            # 3. Train Decoder on reconstruction path
            loss_3, rec, metrics_3 = self.train_decoder_rec_path(
                z_real, embeds_real, batch_cond, training=True
            )
            loss_3.backward()
            self._apply_decoder_gradients()

            rec = _detach_clone(rec)
            del loss_3
        else:
            with torch.no_grad():
                _, fake, metrics_2 = self.train_decoder_fake_path(
                    batch_noise, batch_cond, training=False
                )
                _, rec, metrics_3 = self.train_decoder_rec_path(
                    z_real, embeds_real, batch_cond, training=False
                )

        self.update_step_metrics(metrics_2 | metrics_3)
        del metrics_2, metrics_3, z_real, embeds_real

        self.zero_grad()
        if self.train_encoder_enabled:
            # 4. Train Encoder as critic
            loss_4, metrics_4 = self.train_encoder_critic(
                fake, rec, batch_cond, training=True
            )
            loss_4.backward()
            self._apply_encoder_gradients()
        else:
            with torch.no_grad():
                loss_4, metrics_4 = self.train_encoder_critic(
                    fake, rec, batch_cond, training=False
                )

        self.zero_grad()
        self.update_step_metrics(metrics_4)
        del loss_4, metrics_4, fake, rec

        return self.get_metrics_result()

    def test_step(self, data):

        with torch.no_grad():
            (batch_real, batch_cond), _ = data

            batch_real = self.resize_progressive_output(batch_real)

            loss, z_real, embeds_real, metrics = self.train_collaborative(
                batch_real, batch_cond, training=False
            )

        self.update_step_metrics(metrics)

        return_metrics = {
            name: result
            for name, result in self.get_metrics_result().items()
            if name in metrics
        }
        del loss, z_real, embeds_real, metrics
        return return_metrics

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        encoder = keras.saving.deserialize_keras_object(config.pop("encoder"))
        decoder = keras.saving.deserialize_keras_object(config.pop("decoder"))
        reparameterize = keras.saving.deserialize_keras_object(
            config.pop("reparameterize")
        )
        return cls(encoder, decoder, reparameterize, **config)
