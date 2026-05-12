import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch

from keras import ops
from typing import Dict, Any, Optional

from agx_core.models.reversed_autoencoder import ReversedAutoencoderBase
from agx_core.models.reversed_autoencoder.base import BaseEncoder, BaseDecoder
from agx_torch.models.reversed_autoencoder.layers import Reparameterization


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
        diff_kld_rec_weight: float = 0.7,
        spatial_temperature: float = 1.0,
        lambda_embed: float = 1.0,
        alpha_ssim: Optional[float] = None,
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
            diff_kld_rec_weight=diff_kld_rec_weight,
            spatial_temperature=spatial_temperature,
            lambda_embed=lambda_embed,
            alpha_ssim=alpha_ssim,
            ssim_kwargs=ssim_kwargs,
            name=name,
            **kwargs,
        )

    def train_step(self, data):

        def apply_encoder_gradients():
            trainable_vars = self.encoder.trainable_variables
            grads = [
                v.value.grad if isinstance(v, keras.Variable) else v
                for v in trainable_vars
            ]
            with torch.no_grad():
                self.enc_optimizer.apply(grads, trainable_vars)

        def apply_decoder_gradients():
            trainable_vars = self.decoder.trainable_variables
            grads = [
                v.value.grad if isinstance(v, keras.Variable) else v
                for v in trainable_vars
            ]
            with torch.no_grad():
                self.dec_optimizer.apply(grads, trainable_vars)

        (batch_real, batch_cond), _ = data

        batch_size = ops.shape(batch_real)[0]
        batch_noise = self.noise(batch_size)
        batch_real = self.resize_progressive_output(batch_real)

        # 1. Train VAE Collaborative
        self.zero_grad()

        loss, z_real, embeds_real, metrics_1 = self.train_collaborative(
            batch_real, batch_cond, training=True
        )
        loss.backward()

        apply_encoder_gradients()
        apply_decoder_gradients()

        if self.train_decoder_enabled:
            # 2. Train Decoder on fake path
            self.zero_grad()
            loss, fake, metrics_2 = self.train_decoder_fake_path(
                batch_noise, batch_cond, training=True
            )
            loss.backward()

            # 3. Train Decoder on reconstruction path
            loss, rec, metrics_3 = self.train_decoder_rec_path(
                z_real, embeds_real, batch_cond, training=True
            )
            loss.backward()
            apply_decoder_gradients()
        else:
            with torch.no_grad():
                _, fake, metrics_2 = self.train_decoder_fake_path(
                    batch_noise, batch_cond, training=False
                )
                _, rec, metrics_3 = self.train_decoder_rec_path(
                    z_real, embeds_real, batch_cond, training=False
                )

        if self.train_encoder_enabled:
            # 4. Train Encoder as critic
            self.zero_grad()
            loss, metrics_4 = self.train_encoder_critic(
                fake, rec, batch_cond, training=True
            )
            loss.backward()
            apply_encoder_gradients()
        else:
            with torch.no_grad():
                _, metrics_4 = self.train_encoder_critic(
                    fake, rec, batch_cond, training=False
                )

        metrics = metrics_1 | metrics_2 | metrics_3 | metrics_4
        diff_kld = (
            self.diff_kld_rec_weight * metrics["kld_rec"]
            + (1 - self.diff_kld_rec_weight) * metrics["kld_fake"]
            - metrics["kld_real"]
        )

        metrics.update(dict(diff_kld=diff_kld))
        self.update_step_metrics(metrics)

        return self.get_metrics_result()

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        encoder = keras.saving.deserialize_keras_object(config.pop("encoder"))
        decoder = keras.saving.deserialize_keras_object(config.pop("decoder"))
        reparameterize = keras.saving.deserialize_keras_object(
            config.pop("reparameterize")
        )
        return cls(encoder, decoder, reparameterize, **config)
