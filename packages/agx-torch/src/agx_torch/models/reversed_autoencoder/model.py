import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch

from keras import ops
from keras.src.backend.torch.core import device_scope
from typing import Dict, Any, Optional

from agx_core.models.reversed_autoencoder import ReversedAutoencoderBase
from agx_core.models.reversed_autoencoder.model import kl_divergence
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
        # None = single-GPU mode (no tensor movement)
        self._enc_device: Optional[torch.device] = None
        self._dec_device: Optional[torch.device] = None

    # ─── Device Placement ─────────────────────────────────────────────

    def place_on_devices(self, enc_device: str = "cuda:0", dec_device: str = "cuda:1"):
        """Place encoder and decoder on separate GPUs.

        Call after build(). When set, train_step automatically moves
        tensors across the device boundary. Optimizer states follow
        the network they belong to.

        Single-GPU usage: simply don't call this method.

        Args:
            enc_device: Device string for encoder (e.g. "cuda:0").
            dec_device: Device string for decoder (e.g. "cuda:1").
        """
        self._enc_device = torch.device(enc_device)
        self._dec_device = torch.device(dec_device)

        self.encoder.to(self._enc_device)
        self.reparameterize.to(self._enc_device)
        self.decoder.to(self._dec_device)

        self._build_optimizers_if_needed()
        return self

    @property
    def _multi_gpu(self) -> bool:
        """True when encoder and decoder are on different devices."""
        return self._enc_device is not None and self._enc_device != self._dec_device

    def _build_optimizers_if_needed(self):
        """Build optimizers with state tensors on the correct device.

        Keras optimizer.build() creates momentum/velocity buffers via
        ``convert_to_tensor``, which places them on the Keras default
        device (cuda:0). In multi-GPU mode we wrap each build in the
        appropriate ``device_scope`` so that per-parameter state (momentum,
        velocity) lives alongside its network's parameters.

        The optimizer's scalar variables (iteration, learning_rate) are
        created during ``__init__`` before any device scope, so they
        remain on the default device.  After building, we relocate them
        via ``.data`` assignment to satisfy PyTorch's ``torch._foreach_*``
        same-device requirements.
        """
        if not self._multi_gpu:
            super()._build_optimizers_if_needed()
            return

        if self.enc_optimizer is not None and not self.enc_optimizer.built:
            with device_scope(str(self._enc_device)):
                self.enc_optimizer.build(self.encoder.variables)

        if self.enc_optimizer is not None:
            # Relocate scalar optimizer state (iteration, lr) to dec_device
            for v in self.enc_optimizer.variables:
                if v.value.device != self._enc_device:
                    v.value.data = v.value.data.to(self._enc_device)

        if self.dec_optimizer is not None and not self.dec_optimizer.built:
            with device_scope(str(self._dec_device)):
                self.dec_optimizer.build(self.decoder.variables)

        if self.dec_optimizer is not None:
            # Relocate scalar optimizer state (iteration, lr) to dec_device
            for v in self.dec_optimizer.variables:
                if v.value.device != self._dec_device:
                    v.value.data = v.value.data.to(self._dec_device)

    def _to_enc(self, *tensors):
        """Move tensors to encoder device. No-op in single-GPU mode."""
        if not self._multi_gpu:
            return tensors[0] if len(tensors) == 1 else tensors
        out = tuple(
            t.to(self._enc_device, non_blocking=True) if t is not None else None
            for t in tensors
        )
        return out[0] if len(out) == 1 else out

    def _to_dec(self, *tensors):
        """Move tensors to decoder device. No-op in single-GPU mode."""
        if not self._multi_gpu:
            return tensors[0] if len(tensors) == 1 else tensors
        out = tuple(
            t.to(self._dec_device, non_blocking=True) if t is not None else None
            for t in tensors
        )
        return out[0] if len(out) == 1 else out

    def noise(self, batch_size):
        """Generate latent noise on the decoder device."""
        if self._latent_shape is None:
            raise ValueError(
                "Make sure to build the encoder or to set the correct latent space shape"
            )
        shape = list(self._latent_shape)
        shape[0] = batch_size
        if self._multi_gpu:
            return torch.randn(tuple(shape), device=self._dec_device)
        return keras.random.normal(tuple(shape))

    def train_encoder(self, real, noise, condition):
        if self.train_encoder_enabled:
            self.encoder.training_enabled(True)
            self.decoder.training_enabled(False)

            self.zero_grad()
            loss, metric_updates = self.compute_encoder_loss(real, noise, condition)
            loss.backward()

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
                    real, noise, condition, training=False
                )
            self.update_step_metrics(metric_updates)

    def train_decoder(self, real, noise, condition):
        if self.train_decoder_enabled:
            self.encoder.training_enabled(False)
            self.decoder.training_enabled(True)

            self.zero_grad()
            loss, metric_updates = self.compute_decoder_loss(real, noise, condition)
            loss.backward()

            dec = (
                self.decoder.module if hasattr(self.decoder, "module") else self.decoder
            )
            trainable_vars = dec.trainable_variables
            grads = [v.value.grad for v in trainable_vars]

            if self._multi_gpu:
                with torch.no_grad(), device_scope(str(self._dec_device)):
                    self.dec_optimizer.apply(grads, trainable_vars)
            else:
                with torch.no_grad():
                    self.dec_optimizer.apply(grads, trainable_vars)

            self.update_step_metrics(metric_updates)
        else:
            with torch.no_grad():
                _, metric_updates = self.compute_decoder_loss(
                    real, noise, condition, training=False
                )
            self.update_step_metrics(metric_updates)

    # ─── Multi-GPU Forward Orchestration ─────────────────────────────
    #
    # Override base class orchestration only when multi-GPU is active.
    # Single-GPU falls through to the base class defaults.

    def compute_encoder_loss(self, real, noise, condition, training: bool = True):
        """Encoder loss with multi-GPU device orchestration.

        Encoder forward passes (with grad) run on enc_device.
        Decoder forward passes (no grad) run on dec_device.
        Loss math delegated to _encoder_loss_from_outputs on enc_device.

        Falls through to base class implementation in single-GPU mode.

        Note:
            Decoder calls are wrapped in ``device_scope(dec_device)`` to
            prevent Keras's ``convert_to_tensor`` from moving activations
            back to the default device (cuda:0).
        """
        if not self._multi_gpu:
            return super().compute_encoder_loss(
                real, noise, condition, training=training
            )

        real_e, cond_e = self._to_enc(real, condition)
        noise_d, cond_d = self._to_dec(noise, condition)

        with torch.no_grad(), device_scope(str(self._dec_device)):
            fake_d = self.decoder([noise_d, cond_d], training=False)
        fake_e = self._to_enc(fake_d.detach())

        (mean_real, logvar_real), _ = self.encoder([real_e, cond_e], training=training)
        z_real = self.reparameterize([mean_real, logvar_real])

        z_real_d = self._to_dec(z_real.detach())
        with torch.no_grad(), device_scope(str(self._dec_device)):
            rec_real_d = self.decoder([z_real_d, cond_d], training=False)
        rec_real_e = self._to_enc(rec_real_d.detach())

        (mean_rec, logvar_rec), _ = self.encoder(
            [ops.stop_gradient(rec_real_e), cond_e], training=training
        )
        z_rec = self.reparameterize([mean_rec, logvar_rec])

        z_rec_d = self._to_dec(z_rec.detach())
        with torch.no_grad(), device_scope(str(self._dec_device)):
            rec_rec_d = self.decoder([z_rec_d, cond_d], training=False)
        rec_rec_e = self._to_enc(rec_rec_d.detach())

        (mean_fake, logvar_fake), _ = self.encoder(
            [ops.stop_gradient(fake_e), cond_e], training=training
        )
        z_fake = self.reparameterize([mean_fake, logvar_fake])

        z_fake_d = self._to_dec(z_fake.detach())
        with torch.no_grad(), device_scope(str(self._dec_device)):
            rec_fake_d = self.decoder([z_fake_d, cond_d], training=False)
        rec_fake_e = self._to_enc(rec_fake_d.detach())

        return self._encoder_loss_from_outputs(
            real_e,
            fake_e,
            rec_real_e,
            rec_rec_e,
            rec_fake_e,
            mean_real,
            logvar_real,
            mean_rec,
            logvar_rec,
            mean_fake,
            logvar_fake,
        )

    def compute_decoder_loss(self, real, noise, condition, training: bool = True):
        """Decoder loss with multi-GPU device orchestration.

        Decoder forward passes (with grad) run on dec_device.
        Encoder forward passes (no grad, frozen critic) run on enc_device.
        Loss math delegated to _decoder_loss_from_outputs on dec_device.

        Falls through to base class implementation in single-GPU mode.

        Note:
            Decoder calls and the final loss computation are wrapped in
            ``device_scope(dec_device)`` to prevent Keras's
            ``convert_to_tensor`` from moving activations back to the
            default device (cuda:0).
        """
        if not self._multi_gpu:
            return super().compute_decoder_loss(real, noise, condition)

        real_e, cond_e = self._to_enc(real, condition)
        noise_d, cond_d = self._to_dec(noise, condition)
        real_d = self._to_dec(real)

        # Frozen encoder pass: z_real, embeds_real, kld_real
        with torch.no_grad(), device_scope(str(self._enc_device)):
            (mean_real, logvar_real), embeds_real = self.encoder(
                [real_e, cond_e], training=False
            )
            z_real = self.reparameterize([mean_real, logvar_real])
            kld_real = ops.mean(kl_divergence(mean_real, logvar_real), axis=[1, 2])

        z_real_d = self._to_dec(z_real.detach())
        kld_real_d = self._to_dec(kld_real.detach())

        # Decoder forward passes (with grad)
        with device_scope(str(self._dec_device)):
            fake = self.decoder([noise_d, cond_d], training=training)
            rec_real = self.decoder([z_real_d, cond_d], training=training)

        # Encoder critic: rec_real → embeds_rec, z_rec
        # NO torch.no_grad() here — gradients must flow through the
        # encoder (as a frozen differentiable feature extractor) back to
        # rec_real so that embedding_loss provides signal to the decoder.
        rec_real_e = self._to_enc(rec_real)
        (mean_rec, logvar_rec), embeds_rec = self.encoder(
            [rec_real_e, cond_e], training=False
        )
        z_rec = self.reparameterize([mean_rec, logvar_rec])

        z_rec_d = self._to_dec(z_rec.detach())
        with device_scope(str(self._dec_device)):
            rec_rec = self.decoder([z_rec_d, cond_d], training=training)

        # Encoder critic: fake → z_fake
        fake_e = self._to_enc(fake)
        with torch.no_grad(), device_scope(str(self._enc_device)):
            (mean_fake, logvar_fake), _ = self.encoder([fake_e, cond_e], training=False)
            z_fake = self.reparameterize([mean_fake, logvar_fake])

        z_fake_d = self._to_dec(z_fake.detach())
        with device_scope(str(self._dec_device)):
            rec_fake = self.decoder([z_fake_d, cond_d], training=training)

        # Move encoder outputs to dec_device for loss computation
        embeds_real_d = [self._to_dec(e.detach()) for e in embeds_real]
        embeds_rec_d = [self._to_dec(e) for e in embeds_rec]
        mean_rec_d, logvar_rec_d = self._to_dec(mean_rec.detach(), logvar_rec.detach())
        mean_fake_d, logvar_fake_d = self._to_dec(
            mean_fake.detach(), logvar_fake.detach()
        )

        with device_scope(str(self._dec_device)):
            return self._decoder_loss_from_outputs(
                real_d,
                rec_real,
                rec_rec,
                rec_fake,
                fake,
                mean_rec_d,
                logvar_rec_d,
                mean_fake_d,
                logvar_fake_d,
                embeds_real_d,
                embeds_rec_d,
                kld_real_d,
            )

    def test_step(self, data):
        """Evaluation step with multi-GPU device orchestration.

        All forward passes run on their respective devices (encoder on
        enc_device, decoder on dec_device). Loss math runs on enc_device
        for simplicity (all intermediate results are moved there).

        Falls through to base class implementation in single-GPU mode.
        """
        if not self._multi_gpu:
            return super().test_step(data)

        (real, cond), _ = data
        batch_size = ops.shape(real)[0]
        noise = self.noise(batch_size)  # on dec_device
        real = self.resize_progressive_output(real)

        real_e, cond_e = self._to_enc(real, cond)
        noise_d, cond_d = self._to_dec(noise, cond)

        with torch.no_grad():
            # Decoder generates fake
            with device_scope(str(self._dec_device)):
                fake_d = self.decoder([noise_d, cond_d], training=False)
            fake_e = self._to_enc(fake_d)

            # Encoder on real
            (mean_real, logvar_real), embeds_real = self.encoder(
                [real_e, cond_e], training=False
            )
            z_real = self.reparameterize([mean_real, logvar_real])
            kld_real = ops.mean(kl_divergence(mean_real, logvar_real), axis=[1, 2])

            # Decoder reconstructs real
            z_real_d = self._to_dec(z_real)
            with device_scope(str(self._dec_device)):
                rec_real_d = self.decoder([z_real_d, cond_d], training=False)
            rec_real_e = self._to_enc(rec_real_d)

            # Encoder on rec_real
            (mean_rec, logvar_rec), embeds_rec = self.encoder(
                [rec_real_e, cond_e], training=False
            )
            z_rec = self.reparameterize([mean_rec, logvar_rec])

            # Decoder reconstructs rec
            z_rec_d = self._to_dec(z_rec)
            with device_scope(str(self._dec_device)):
                rec_rec_d = self.decoder([z_rec_d, cond_d], training=False)
            rec_rec_e = self._to_enc(rec_rec_d)

            # Encoder on fake
            (mean_fake, logvar_fake), _ = self.encoder([fake_e, cond_e], training=False)
            z_fake = self.reparameterize([mean_fake, logvar_fake])

            # Decoder reconstructs fake
            z_fake_d = self._to_dec(z_fake)
            with device_scope(str(self._dec_device)):
                rec_fake_d = self.decoder([z_fake_d, cond_d], training=False)
            rec_fake_e = self._to_enc(rec_fake_d)

        # All loss math on enc_device (everything is already there)
        _, enc_metrics = self._encoder_loss_from_outputs(
            real_e,
            fake_e,
            rec_real_e,
            rec_rec_e,
            rec_fake_e,
            mean_real,
            logvar_real,
            mean_rec,
            logvar_rec,
            mean_fake,
            logvar_fake,
        )

        _, dec_metrics = self._decoder_loss_from_outputs(
            real_e,
            rec_real_e,
            rec_rec_e,
            rec_fake_e,
            fake_e,
            mean_rec,
            logvar_rec,
            mean_fake,
            logvar_fake,
            embeds_real,
            embeds_rec,
            kld_real,
        )

        self.update_step_metrics(enc_metrics)
        self.update_step_metrics(dec_metrics)

        return self.get_metrics_result()

    # ─── Serialization ────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        encoder = keras.saving.deserialize_keras_object(config.pop("encoder"))
        decoder = keras.saving.deserialize_keras_object(config.pop("decoder"))
        reparameterize = keras.saving.deserialize_keras_object(
            config.pop("reparameterize")
        )
        return cls(encoder, decoder, reparameterize, **config)
