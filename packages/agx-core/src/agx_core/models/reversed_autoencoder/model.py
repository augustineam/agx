import keras

from keras import Model, metrics, ops, optimizers
from typing import Sequence, Optional, Dict, Any

from .base import BaseEncoder, BaseDecoder
from .layers import Reparameterization, CompositeConditionEncoder
from .losses import *

from agx_core.helpers import _channel_axis


@keras.saving.register_keras_serializable(
    package="agx_core.models.reversed_autoencoder"
)
class ReversedAutoencoderBase(Model):
    """Reversed Autoencoder orchestrating adversarial training between encoder and decoder.

    The encoder and decoder are instantiated externally and passed to this model.
    Call ``compile(enc_optimizer, dec_optimizer)`` before training — no ``loss``
    argument is needed because the training step computes losses internally.
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        reparameterize=Reparameterization(),
        condition: Optional[CompositeConditionEncoder] = None,
        beta_kld: float = 1.0,
        enc_expkld_temp: float = 1.0,
        dec_expelbo_temp: float = 1.0,
        lambda_embed: float = 1.0,
        spatial_temp: float = 1.0,
        alpha_ssim: float = 0.3,
        ssim_kwargs: Optional[Dict[str, Any]] = None,
        z_fake_interp: Optional[Dict[str, Any]] = None,
        name: str = "reversed_autoencoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.reparameterize = reparameterize
        self.condition = condition

        self.beta_kld = beta_kld
        self.enc_expkld_temp = enc_expkld_temp
        self.dec_expelbo_temp = dec_expelbo_temp
        self.spatial_temp = spatial_temp
        self.lambda_embed = lambda_embed
        self.alpha_ssim = alpha_ssim
        self.z_fake_interp = (
            z_fake_interp
            if z_fake_interp is not None
            else dict(
                mode="manifold",  # | "perturbed" | "slerp"
                manifold_op="roll",  # | "shuffle"
                perturbed_sigma=0.2,
            )
        )
        self.ssim_kwargs = (
            ssim_kwargs
            if ssim_kwargs is not None
            else dict(
                max_val=1.0,
                filter_size=11,
                filter_sigma=1.5,
                k1=0.01,
                k2=0.03,
            )
        )

        # Two independent optimizers — set via compile()
        self.optimizer: Optional[keras.Optimizer] = None
        self._dec_optimizer: Optional[keras.Optimizer] = None
        self._cond_optimizer: Optional[keras.Optimizer] = None

        # Turn-taking flags — controlled by AdversarialEquilibriumCallback
        self.train_encoder_enabled = True
        self.train_decoder_enabled = True

        self.loss_rec_tracker = metrics.Mean("loss_rec")
        self.loss_embed_tracker = metrics.Mean("loss_embed")
        self.kld_real_tracker = metrics.Mean("kld_real")
        self.kld_rec_tracker = metrics.Mean("kld_rec")
        self.kld_fake_tracker = metrics.Mean("kld_fake")
        self.elbo_real_tracker = metrics.Mean("elbo_real")
        self.elbo_rec_tracker = metrics.Mean("elbo_rec")
        self.elbo_fake_tracker = metrics.Mean("elbo_fake")
        self.expkld_rec_tracker = metrics.Mean("expkld_rec")
        self.expkld_fake_tracker = metrics.Mean("expkld_fake")
        self.expelbo_rec_tracker = metrics.Mean("expelbo_rec")
        self.expelbo_fake_tracker = metrics.Mean("expelbo_fake")

    @property
    def metrics(self):
        """Return list of all tracked metrics.

        Returns:
            List of metric trackers for monitoring training progress
        """
        return [
            self.loss_rec_tracker,
            self.loss_embed_tracker,
            self.kld_real_tracker,
            self.kld_rec_tracker,
            self.kld_fake_tracker,
            self.elbo_real_tracker,
            self.elbo_rec_tracker,
            self.elbo_fake_tracker,
            self.expkld_rec_tracker,
            self.expkld_fake_tracker,
            self.expelbo_rec_tracker,
            self.expelbo_fake_tracker,
        ]

    @property
    def test_metrics(self):
        """Return list of test tracked metrics.

        Returns:
            List of metric trackers for monitoring validation progress
        """
        return [
            self.loss_rec_tracker,
            self.kld_real_tracker,
            self.elbo_real_tracker,
        ]

    @property
    def enc_optimizer(self):
        return self.optimizer

    @enc_optimizer.setter
    def enc_optimizer(self, optimizer: keras.Optimizer):
        self.optimizer = optimizer

    @property
    def dec_optimizer(self):
        return self._dec_optimizer

    @dec_optimizer.setter
    def dec_optimizer(self, optimizer: keras.Optimizer):
        self._dec_optimizer = optimizer

    @property
    def cond_optimizer(self):
        return self._cond_optimizer

    @cond_optimizer.setter
    def cond_optimizer(self, optimizer: Optional[keras.Optimizer]):
        self._cond_optimizer = optimizer

    def build(self, input_shape: Sequence[Sequence[int]]):
        x_shape, c_shape = input_shape

        # Stash for get_build_config before any shape mutation.
        self._img_shape = tuple(x_shape)
        self._cond_shape = tuple(c_shape)

        if self.condition:
            self.condition.build(c_shape)
            c_shape = self.condition.compute_output_shape(c_shape)

        self.encoder.build([x_shape, c_shape])
        latent_shape, *_ = self.encoder.compute_output_shape([x_shape, c_shape])

        self._latent_shape = latent_shape[0]

        self.reparameterize.build(latent_shape)
        x_shape = self.reparameterize.compute_output_shape(latent_shape)

        self.decoder.build([x_shape, c_shape])
        x_shape = self.decoder.compute_output_shape([x_shape, c_shape])

        # Pre-build SSIM kernel (never changes, no gradients needed)
        ch_axis = _channel_axis()
        num_channels = x_shape[ch_axis]
        self._ssim_kwargs = self.ssim_kwargs.copy()

        self._ssim_kernel = self._build_ssim_kernel(
            self._ssim_kwargs.pop("filter_size", 11),
            self._ssim_kwargs.pop("filter_sigma", 1.5),
            num_channels,
        )

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        x_shape, c_shape = input_shape

        if self.condition:
            c_shape = self.condition.compute_output_shape(c_shape)

        latent_shape, *features_shape = self.encoder.compute_output_shape(
            [x_shape, c_shape]
        )
        z_shape = self.reparameterize.compute_output_shape(latent_shape)
        rec_shape = self.decoder.compute_output_shape([z_shape, c_shape])

        return rec_shape, latent_shape, *features_shape

    def _build_ssim_kernel(self, filter_size, filter_sigma, num_channels):
        """Build and cache the SSIM Gaussian kernel (no gradients)."""
        coords = ops.arange(filter_size, dtype="float32") - (filter_size - 1) / 2.0
        g = ops.exp(
            -(coords[:, None] ** 2 + coords[None, :] ** 2) / (2.0 * filter_sigma**2)
        )
        kernel = g / ops.sum(g)  # (kH, kW)

        if keras.config.image_data_format() == "channels_last":
            kernel = ops.expand_dims(kernel, axis=[-2, -1])  # (kH, kW, 1, 1)
        else:
            kernel = ops.expand_dims(kernel, axis=[0, 1])  # (1, 1, kH, kW)

        kernel = ops.tile(kernel, [1, 1, num_channels, 1])
        return ops.stop_gradient(kernel)  # Ensure no grad tracking

    def get_build_config(self):
        """Persist the two input shapes so build_from_config can rebuild optimizers."""
        return {
            "img_shape": self._img_shape,
            "cond_shape": self._cond_shape,
        }

    def build_from_config(self, config):
        """Mark the model as built (sub-layers are already restored).

        During deserialization the sub-layers (encoder, decoder, reparameterize)
        are reconstructed and built by their own ``from_config`` /
        ``build_from_config`` calls before this method is reached.  Re-calling
        ``build()`` would attempt to add new variables to already-locked layer
        trackers and raise a ``ValueError``.

        Optimizer building is intentionally deferred to ``compile_from_config``
        which runs *after* this in the Keras deserialization sequence.
        """
        self._img_shape = tuple(config["img_shape"])
        self._cond_shape = tuple(config["cond_shape"])

        if self.condition:
            self.condition.build(self._cond_shape)
            c_shape = self.condition.compute_output_shape(self._cond_shape)
        else:
            c_shape = self._cond_shape

        x_shape, _ = self.encoder.compute_output_shape([self._img_shape, c_shape])
        self._latent_shape = x_shape[0]

        ch_axis = _channel_axis()
        num_channels = self._img_shape[ch_axis]
        self._ssim_kwargs = self.ssim_kwargs.copy()

        self._ssim_kernel = self._build_ssim_kernel(
            self._ssim_kwargs.pop("filter_size", 11),
            self._ssim_kwargs.pop("filter_sigma", 1.5),
            num_channels,
        )

        self.built = True

    def _build_optimizers_if_needed(self):
        """Build enc/dec optimizers against *all* sub-layer variables.

        Uses ``layer.variables`` (not ``trainable_variables``) so that the
        optimizer slot count is independent of the transient ``trainable``
        flag.  During adversarial training the flag is toggled every step,
        and ``test_step`` sets both to ``False``.  If the model is saved in
        that state the config persists ``trainable: false``, causing
        ``trainable_variables`` to return an empty list on reload.  Building
        against ``variables`` guarantees the optimizer always creates the
        correct number of slots and saved momentum / adaptive-rate state can
        be restored without a mismatch.

        Safe to call multiple times — Keras optimizers are idempotent once
        built. Called from both compile() (live training path) and
        build_from_config() (deserialization path) to guarantee the optimizers
        always have the correct number of slot variables before Keras tries to
        restore saved state into them.
        """
        if self.enc_optimizer is not None and not self.enc_optimizer.built:
            # If condition encoder is set and condition optimizer wasn't provided, encoder optimizer will be used,
            if self.condition is not None and self.cond_optimizer is None:
                self.enc_optimizer.build(
                    self.encoder.variables + self.condition.variables
                )
            else:
                self.enc_optimizer.build(self.encoder.variables)
        if self.dec_optimizer is not None and not self.dec_optimizer.built:
            self.dec_optimizer.build(self.decoder.variables)

        if (
            self.condition is not None
            and self.cond_optimizer is not None
            and not self.cond_optimizer.built
        ):
            self.cond_optimizer.build(self.condition.variables)

    def compile(
        self,
        enc_optimizer: keras.Optimizer,
        dec_optimizer: keras.Optimizer,
        cond_optimizer: Optional[keras.Optimizer] = None,
        **kwargs,
    ):
        """Compile the model with separate optimizers for encoder and decoder.

        No ``loss`` argument is accepted — losses are computed inside
        ``train_step`` / ``test_step`` and are not routed through the
        standard Keras loss machinery.

        Args:
            enc_optimizer: Optimizer used to update the encoder's weights.
            dec_optimizer: Optimizer used to update the decoder's weights.
        """
        # Pass optimizer=None so the base Trainer does not create a phantom
        # rmsprop optimizer that would be saved/loaded alongside our real ones.
        super().compile(optimizer=enc_optimizer, **kwargs)
        self.dec_optimizer = dec_optimizer
        self.cond_optimizer = cond_optimizer

        if self.built:
            self._build_optimizers_if_needed()

    def compile_from_config(self, config: Dict[str, Any]):
        """Restore both optimizers from a serialized config produced by get_compile_config.

        Keras deserialization order (``serialization_lib.py``):
          1. ``from_config()``  — sub-layers recursively built
          2. ``build_from_config()`` — only if ``not instance.built``
          3. ``compile_from_config()`` — this method
          4. ``_load_state()`` — weight & optimizer-state restoration

        By step 3 the sub-layers are fully reconstructed and their
        ``.variables`` lists are populated, so we can build the optimizers
        eagerly here.  This ensures the optimizer slot counts match the
        saved state *before* step 4 tries to load them.
        """
        enc_optimizer = optimizers.deserialize(config["enc_optimizer"])
        dec_optimizer = optimizers.deserialize(config["dec_optimizer"])
        cond_optimizer = config.get("cond_optimizer")
        cond_optimizer = (
            None if cond_optimizer is None else optimizers.deserialize(cond_optimizer)
        )

        self.compile(enc_optimizer, dec_optimizer, cond_optimizer)
        self._build_optimizers_if_needed()

    def get_compile_config(self):
        """Serialize the optimizer pair so that compile_from_config can restore them."""
        return {
            "enc_optimizer": optimizers.serialize(self.enc_optimizer),
            "dec_optimizer": optimizers.serialize(self.dec_optimizer),
            "cond_optimizer": (
                optimizers.serialize(self.cond_optimizer)
                if self.cond_optimizer
                else None
            ),
        }

    def reconstruction_loss(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
    ):
        """Combined pixel + structural reconstruction loss.

        Returns spatial map (B, H, W) suitable for mean-reduction to logpx_z.

        Args:
            alpha_ssim: Blend weight. 0 = pure MSE (original behavior).
                        Recommended for X-rays: 0.15–0.35
        """
        mse = mse_weighted(y_true, y_pred, self.spatial_temp)  # (B, H, W)

        if self.alpha_ssim <= 0.0:
            return mse

        ssim = ssim_loss(
            y_true, y_pred, self._ssim_kernel, **self._ssim_kwargs
        )  # (B, H, W)

        return (1.0 - self.alpha_ssim) * mse + self.alpha_ssim * ssim

    def call(
        self,
        inputs: Sequence[keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass through the autoencoder.

        Args:
            inputs: List containing [input_images, conditioning] tensors
            training: Whether the model is in training mode

        Returns:
            Reconstructed images
        """

        x, c = inputs
        x = self.resize_progressive_output(x)

        if self.condition:
            c = self.condition(c, training=training)

        mulogvar, *features = self.encoder([x, c], training=training)
        z = self.reparameterize(mulogvar)
        return (
            self.decoder([z, ops.stop_gradient(c)], training=training),
            mulogvar,
            *features,
        )

    def resize_progressive_output(self, image: keras.KerasTensor):
        from .decoders.mobilenet_v3 import MobileNetV3SmallProgressiveDecoder

        if not isinstance(self.decoder, MobileNetV3SmallProgressiveDecoder):
            return image
        if self.decoder.is_fully_grown:
            return image
        return ops.image.resize(image, self.decoder.current_output_size())

    def train_collaborative(
        self,
        real: keras.KerasTensor,
        cond: keras.KerasTensor,
        training: bool = True,
    ):
        self.encoder.training_enabled(training)
        self.decoder.training_enabled(training)

        if self.condition:
            self.condition.trainable = training or True
            cond = self.condition(cond, training=training)

        mu_logvar, *embeds = self.encoder([real, cond], training=training)
        z = self.reparameterize(mu_logvar)
        rec = self.decoder([z, ops.stop_gradient(cond)], training=training)

        loss_rec = ops.mean(
            self.reconstruction_loss(real, rec),
            axis=[1, 2],
        )
        kld = ops.mean(kl_divergence(*mu_logvar), axis=[1, 2])
        elbo_real = -loss_rec - self.beta_kld * kld

        loss = -ops.mean(elbo_real)

        metrics = dict(
            loss_rec=loss_rec,
            kld_real=kld,
            elbo_real=elbo_real,
        )

        return loss, ops.stop_gradient(z), embeds, metrics

    def train_decoder_fake_path(
        self,
        z_real: keras.KerasTensor,
        cond: keras.KerasTensor,
        training: bool = True,
    ):
        self.encoder.training_enabled(False)
        self.decoder.training_enabled(training)

        if self.condition:
            self.condition.trainable = False
            cond = ops.stop_gradient(self.condition(cond, training=False))

        z_fake = self.sample_z_fake(z_real)

        fake = self.decoder([z_fake, cond], training=training)

        mu_logvar, *_ = self.encoder([fake, cond], training=False)
        z = self.reparameterize(mu_logvar)
        rec = self.decoder([z, cond], training=training)

        logpx_z = -ops.mean(
            self.reconstruction_loss(ops.stop_gradient(fake), rec),
            axis=[1, 2],
        )
        kld = ops.mean(kl_divergence(*mu_logvar), axis=[1, 2])
        elbo = logpx_z - self.beta_kld * kld

        expelbo = ops.exp(-self.dec_expelbo_temp * elbo)
        loss = ops.mean(expelbo)

        metrics = dict(
            elbo_fake=elbo,
            expelbo_fake=expelbo,
        )

        return loss, ops.stop_gradient(fake), metrics

    def train_decoder_rec_path(
        self,
        z_real: keras.KerasTensor,
        embeds_real: keras.KerasTensor,
        cond: keras.KerasTensor,
        training: bool = True,
    ):
        self.encoder.training_enabled(False)
        self.decoder.training_enabled(training)

        if self.condition:
            self.condition.trainable = False
            cond = ops.stop_gradient(self.condition(cond, training=False))

        rec = self.decoder([z_real, cond], training=training)

        mu_logvar, *embeds_rec = self.encoder([rec, cond], training=False)
        z_rec = self.reparameterize(mu_logvar)
        rec_rec = self.decoder([z_rec, cond], training=training)

        logpx_z = -ops.mean(
            self.reconstruction_loss(ops.stop_gradient(rec), rec_rec),
            axis=[1, 2],
        )
        kld = ops.mean(kl_divergence(*mu_logvar), axis=[1, 2])
        elbo = logpx_z - self.beta_kld * kld

        embed_loss = embedding_loss(embeds_real, embeds_rec)
        expelbo = ops.exp(-self.dec_expelbo_temp * elbo)

        loss = ops.mean(expelbo + embed_loss)

        metrics = dict(
            elbo_rec=elbo,
            expelbo_rec=expelbo,
            loss_embed=embed_loss,
        )

        return loss, ops.stop_gradient(rec), metrics

    def train_encoder_critic(
        self,
        fake: keras.KerasTensor,
        rec: keras.KerasTensor,
        cond: keras.KerasTensor,
        training: bool = True,
    ):
        self.encoder.training_enabled(training)
        self.decoder.training_enabled(False)

        if self.condition:
            self.condition.trainable = False
            cond = ops.stop_gradient(self.condition(cond, training=False))

        mu_logvar_fake, *_ = self.encoder([fake, cond], training=training)
        mu_logvar_rec, *_ = self.encoder([rec, cond], training=training)

        kld_fake = ops.mean(kl_divergence(*mu_logvar_fake), axis=[1, 2])
        kld_rec = ops.mean(kl_divergence(*mu_logvar_rec), axis=[1, 2])

        expkld_fake = ops.exp(-self.enc_expkld_temp * kld_fake)
        expkld_rec = ops.exp(-self.enc_expkld_temp * kld_rec)

        loss = ops.mean(expkld_fake + expkld_rec)

        metrics = dict(
            kld_rec=kld_rec,
            kld_fake=kld_fake,
            expkld_fake=expkld_fake,
            expkld_rec=expkld_rec,
        )

        return loss, metrics

    def update_step_metrics(self, metric_updates):
        """Helper to cleanly update all tracked metrics from the returned dictionaries."""
        for name, value in metric_updates.items():
            getattr(self, f"{name}_tracker").update_state(value)

    def train_step(self, data):
        """Perform one training step with adversarial encoder-decoder training.

        Args:
            data: Training batch containing (images, conditions, sample_weights)

        Returns:
            Dictionary of metric results
        """
        raise NotImplementedError(
            "Gradient application must be implemented by backend subclasses."
        )

    def test_step(self, data):
        """Perform evaluation step computing all metrics without training.

        Args:
            data: Validation batch containing (images, conditions, sample_weights)

        Returns:
            Dictionary of evaluation metrics
        """

        (batch_real, batch_cond), _ = data

        batch_real = self.resize_progressive_output(batch_real)

        *_, metrics = self.train_collaborative(batch_real, batch_cond, training=False)

        self.update_step_metrics(metrics)
        return {metric.name: metric.result() for metric in self.test_metrics}

    def sample_z_fake(self, z: keras.KerasTensor):
        mode = self.z_fake_interp.get("mode", "perturbed")

        if mode == "manifold":
            z_fake = self._sample_z_manifold(z)
        elif mode == "perturbed":
            z_fake = self._sample_z_perturbed(z)
        elif mode == "slerp":
            z_fake = self._sample_z_slerp(z)
        else:
            raise ValueError(f"Unknown interpolation mode: {mode}")

        return z_fake

    def _sample_manifold(self, z: keras.KerasTensor):
        manifold_op = self.z_fake_interp.get("manifold_op", "roll")
        if manifold_op == "roll":
            z_b = ops.roll(z, shift=1, axis=0)
        else:
            z_b = keras.random.shuffle(z, axis=0)
        return ops.stop_gradient(z_b)

    def _sample_z_manifold(self, z: keras.KerasTensor):
        z_b = self._sample_manifold(z)
        batch_size = ops.shape(z)[0]
        t = keras.random.uniform((batch_size, 1, 1, 1))
        z_interp = (1 - t) * z + t * z_b
        return ops.stop_gradient(z_interp)

    def _sample_z_perturbed(self, z: keras.KerasTensor):
        sigma = self.z_fake_interp.get("perturbed_sigma", 0.2)
        z_perturbed = z + keras.random.normal(ops.shape(z)) * sigma
        return ops.stop_gradient(z_perturbed)

    def _sample_z_slerp(self, z: keras.KerasTensor):
        z_b = self._sample_manifold(z)
        batch_size = ops.shape(z)[0]
        t = keras.random.uniform((batch_size, 1, 1, 1))
        theta = ops.arccos(ops.sum(z * z_b, axis=_channel_axis(), keepdims=True))
        z_interp = (ops.sin((1 - t) * theta) * z + ops.sin(t * theta) * z_b) / ops.sin(
            theta
        )
        return ops.stop_gradient(z_interp)

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                encoder=keras.saving.serialize_keras_object(self.encoder),
                decoder=keras.saving.serialize_keras_object(self.decoder),
                reparameterize=keras.saving.serialize_keras_object(self.reparameterize),
                condition=(
                    None
                    if self.condition is None
                    else keras.saving.serialize_keras_object(self.condition)
                ),
                beta_kld=self.beta_kld,
                enc_expkld_temp=self.enc_expkld_temp,
                dec_expelbo_temp=self.dec_expelbo_temp,
                lambda_embed=self.lambda_embed,
                spatial_temp=self.spatial_temp,
                alpha_ssim=self.alpha_ssim,
                ssim_kwargs=self.ssim_kwargs,
                z_fake_interp=self.z_fake_interp,
            )
        )
        return config


__all__ = ["ReversedAutoencoderBase"]
