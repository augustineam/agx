import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch

from typing import Dict, Any, Optional

from agx_core.models.reversed_autoencoder import ReversedAutoencoderBase
from agx_core.models.reversed_autoencoder.layers import CompositeConditionEncoder
from agx_core.models.reversed_autoencoder.base import BaseEncoder, BaseDecoder
from agx_torch.models.reversed_autoencoder.layers import Reparameterization


@keras.saving.register_keras_serializable(
    package="agx_torch.models.reversed_autoencoder"
)
class ReversedAutoencoder(ReversedAutoencoderBase):
    """PyTorch backend implementation of the Reversed Autoencoder.

    Supports optional multi-GPU training via ``place_on_devices()``:
    encoder and its optimizer live on one GPU, decoder and its optimizer
    on another. Tensors are moved across the device boundary as needed.
    When ``place_on_devices()`` is not called, everything stays on the
    default device (single-GPU mode) and the base class's default
    orchestration is used.
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        reparameterize=Reparameterization(),
        condition: Optional[CompositeConditionEncoder] = None,
        beta_kld: float = 0.25,
        enc_expkld_temp: float = 1.0,
        dec_expelbo_temp: float = 1.0,
        spatial_temp: float = 1.0,
        lambda_embed: float = 1.0,
        alpha_ssim: float = 0.3,
        ssim_kwargs: Optional[Dict[str, Any]] = None,
        z_fake_interp: Optional[Dict[str, Any]] = None,
        name: str = "reversed_autoencoder",
        **kwargs,
    ):
        super().__init__(
            encoder,
            decoder,
            reparameterize,
            condition=condition,
            beta_kld=beta_kld,
            enc_expkld_temp=enc_expkld_temp,
            dec_expelbo_temp=dec_expelbo_temp,
            spatial_temp=spatial_temp,
            lambda_embed=lambda_embed,
            alpha_ssim=alpha_ssim,
            ssim_kwargs=ssim_kwargs,
            z_fake_interp=z_fake_interp,
            name=name,
            **kwargs,
        )

    def _apply_encoder_gradients(self, include_condition: bool = False):
        train_vars = [v for v in self.encoder.trainable_weights]

        if (
            include_condition
            and self.condition is not None
            and self.cond_optimizer is None
        ):
            train_vars += [v for v in self.condition.trainable_weights]

        grads = [v.value.grad for v in train_vars]
        with torch.no_grad():
            self.enc_optimizer.apply(grads, train_vars)

    def _apply_decoder_gradients(self):
        grads = [v.value.grad for v in self.decoder.trainable_weights]
        with torch.no_grad():
            self.dec_optimizer.apply(grads, self.decoder.trainable_weights)

    def _apply_condition_gradients(self):
        if self.condition is None or self.cond_optimizer is None:
            return

        grads = [v.value.grad for v in self.condition.trainable_weights]
        with torch.no_grad():
            self.cond_optimizer.apply(grads, self.condition.trainable_weights)

    def train_step(self, data):
        (batch_real, batch_cond), _ = data

        batch_real = self.resize_progressive_output(batch_real)

        # 1. Train VAE Collaborative
        loss_1, z_real, embeds_real, metrics_1 = self.train_collaborative(
            batch_real, batch_cond, training=True
        )

        torch.nn.Module.zero_grad(self)
        loss_1.backward()

        self._apply_encoder_gradients(include_condition=True)
        self._apply_decoder_gradients()
        self._apply_condition_gradients()
        self.update_step_metrics(metrics_1)

        if self.train_decoder_enabled:
            # 2. Train Decoder on fake path
            loss_2, fake, metrics_2 = self.train_decoder_fake_path(
                z_real, batch_cond, training=True
            )

            torch.nn.Module.zero_grad(self)
            loss_2.backward()

            # 3. Train Decoder on reconstruction path
            loss_3, rec, metrics_3 = self.train_decoder_rec_path(
                z_real, embeds_real, batch_cond, training=True
            )
            loss_3.backward()

            self._apply_decoder_gradients()
        else:
            with torch.no_grad():
                _, fake, metrics_2 = self.train_decoder_fake_path(
                    z_real, batch_cond, training=False
                )
                _, rec, metrics_3 = self.train_decoder_rec_path(
                    z_real, embeds_real, batch_cond, training=False
                )

        self.update_step_metrics(metrics_2 | metrics_3)

        if self.train_encoder_enabled:
            # 4. Train Encoder as critic
            loss_4, metrics_4 = self.train_encoder_critic(
                fake, rec, batch_cond, training=True
            )

            torch.nn.Module.zero_grad(self)
            loss_4.backward()

            self._apply_encoder_gradients(include_condition=False)
        else:
            with torch.no_grad():
                _, metrics_4 = self.train_encoder_critic(
                    fake, rec, batch_cond, training=False
                )

        self.update_step_metrics(metrics_4)

        torch.nn.Module.zero_grad(self)

        return self.get_metrics_result()

    def test_step(self, data):

        with torch.no_grad():
            (batch_real, batch_cond), _ = data

            batch_real = self.resize_progressive_output(batch_real)
            *_, metrics = self.train_collaborative(
                batch_real, batch_cond, training=False
            )
            self.update_step_metrics(metrics)

        return {m.name: m.result() for m in self.test_metrics}

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        encoder = keras.saving.deserialize_keras_object(config.pop("encoder"))
        decoder = keras.saving.deserialize_keras_object(config.pop("decoder"))
        reparameterize = keras.saving.deserialize_keras_object(
            config.pop("reparameterize")
        )
        cond_config = config.pop("condition")
        condition = (
            None
            if cond_config is None
            else keras.saving.deserialize_keras_object(cond_config)
        )

        return cls(
            encoder,
            decoder,
            reparameterize,
            condition,
            **config,
        )
