import keras


from keras import metrics, Model, ops
from typing import Sequence, Optional, Dict, Any

from .base import BaseEncoder, BaseDecoder
from .layers import Reparameterization

from agx_core.helpers import _channel_axis


def ssim_loss(
    y_true: keras.KerasTensor,
    y_pred: keras.KerasTensor,
    kernel: keras.KerasTensor,
    max_val: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> keras.KerasTensor:
    """
    Structural Similarity Index (SSIM) loss via `ops`.

    Computes `1 - SSIM(y_true, y_pred)` so that minimizing the loss
    maximizes structural similarity.

    Args:
        y_true: Ground-truth images, shape `(B, H, W, C)`.
        y_pred: Predicted images, shape `(B, H, W, C)`.
        max_val: The dynamic range of the images (1.0 for normalized, 255.0 for uint8).
        filter_size: Side length of the Gaussian kernel.
        filter_sigma: Standard deviation of the Gaussian kernel.
        k1: Stability constant for the luminance term.
        k2: Stability constant for the contrast/structure term.

    Returns:
        Scalar loss value: `1 - mean(SSIM)`.
    """
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    def _apply_filter(x: keras.KerasTensor) -> keras.KerasTensor:
        return ops.depthwise_conv(
            x, kernel, strides=(1, 1), padding="same", dilation_rate=(1, 1)
        )

    mu_x = _apply_filter(y_true)
    mu_y = _apply_filter(y_pred)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = _apply_filter(y_true * y_true) - mu_x_sq
    sigma_y_sq = _apply_filter(y_pred * y_pred) - mu_y_sq
    sigma_xy = _apply_filter(y_true * y_pred) - mu_xy

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

    ssim_map = numerator / denominator
    return 1.0 - ops.mean(ssim_map, axis=_channel_axis())


def kl_divergence(mean, logvar, cap=10.0):
    """Closed-form KLD from N(mean, exp(logvar)) to N(0, I), per spatial position.

    With a spatial latent (B, H, W, C), this returns a (B, H, W) map where
    each position measures how "unusual" the encoding is at that location.

    KLD = 0.5 * Σ_c (μ² + exp(logvar) - logvar - 1) / C

    This replaces the Monte Carlo estimate (log q(z|x) - log p(z)) which
    requires sampling z and has non-zero variance. The closed form is
    exact, lower variance, and cheaper to compute.

    Returns:
        KLD map of shape (B, H, W) — mean over channels, spatial preserved.
    """
    kld = 0.5 * ops.mean(
        ops.square(mean) + ops.exp(logvar) - logvar - 1.0,
        axis=_channel_axis(),
    )
    return ops.minimum(kld, cap)


def embedding_loss(
    teacher_embedds: Sequence[keras.KerasTensor],
    student_embedds: Sequence[keras.KerasTensor],
):
    """Feature consistency loss (MSE + Cosine Similarity) between two sets of embeddings."""
    total_loss = 0
    scale = 1.0 / len(teacher_embedds)
    for teacher_feature, student_feature in zip(teacher_embedds, student_embedds):
        mse = ops.mean(
            ops.square(teacher_feature - student_feature), axis=_channel_axis()
        )
        # [B, H, W]
        cosine_similarity = keras.losses.cosine_similarity(
            teacher_feature, student_feature, axis=_channel_axis()
        )
        total_loss += ops.mean(0.5 * mse + (1 + cosine_similarity), axis=[1, 2])
    return scale * total_loss


def mse_weighted(y_true, y_pred, spatial_temperature: float = 0.0):
    """Per-pixel MSE with optional spatial curriculum weighting.

    When spatial_temperature > 0, hard-to-reconstruct regions receive
    exponentially more gradient attention. Critical for information-sparse
    images (single-channel X-rays) where uniform background dominates
    the spatial mean.

    Args:
        y_true: Ground truth [B, H, W, C]
        y_pred: Prediction [B, H, W, C]
        spatial_temperature: Curriculum sharpness. 0 = uniform (original behavior).
            Recommended range for X-rays: 2.0-10.0

    Returns:
        Error map [B, H, W]. When temperature > 0, the map is spatially
        reweighted so that hard regions contribute more to the downstream mean.
    """
    error = ops.mean(ops.square(y_true - y_pred), axis=_channel_axis())  # [B, H, W]

    if spatial_temperature <= 0.0:
        return error

    # Exponential weighting: hard pixels (high error) get more attention
    # stop_gradient prevents second-order effects (don't optimize weights themselves)
    weights = ops.exp(spatial_temperature * ops.stop_gradient(error))
    # Normalize per-sample so overall loss magnitude is preserved
    weights = weights / ops.mean(weights, axis=[1, 2], keepdims=True)

    return weights * error


@keras.saving.register_keras_serializable(
    package="agx_core.models.reversed_autoencoder"
)
class ReversedAutoencoderBase(Model):
    """Reversed Autoencoder orchestrating adversarial training between encoder and decoder.

    The encoder and decoder are instantiated externally and passed to this model.
    Call ``compile(enc_optimizer, dec_optimizer)`` before training — no ``loss``
    argument is needed because the training step computes losses internally.
    """

    enc_optimizer: keras.optimizers.Optimizer
    dec_optimizer: keras.optimizers.Optimizer
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
        super().__init__(name=name, **kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.reparameterize = reparameterize
        self.beta_kld = beta_kld
        self.enc_expkld_temp = enc_expkld_temp
        self.dec_expelbo_temp = dec_expelbo_temp
        self.spatial_temperature = spatial_temperature
        self.lambda_embed = lambda_embed
        self.alpha_ssim = alpha_ssim
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
        self.optimizer: Optional[keras.optimizers.Optimizer] = None
        self._dec_optimizer: Optional[keras.optimizers.Optimizer] = None

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
    def dec_optimizer(self):
        return self._dec_optimizer

    @dec_optimizer.setter
    def dec_optimizer(self, optimizer: keras.Optimizer):
        self._dec_optimizer = optimizer

    @property
    def enc_optimizer(self):
        return self.optimizer

    @enc_optimizer.setter
    def enc_optimizer(self, optimizer: keras.Optimizer):
        self.optimizer = optimizer

    def noise(self, batch_size) -> keras.KerasTensor:
        if self._latent_shape is None:
            raise ValueError(
                "Make sure to build the encoder or to set the correct latent space shape"
            )
        shape = list(self._latent_shape)
        shape[0] = batch_size
        noise = keras.random.normal(tuple(shape))
        return ops.stop_gradient(noise)

    def build(self, input_shape: Sequence[Sequence[int]]):
        x_shape, c_shape = input_shape

        # Stash for get_build_config before any shape mutation.
        self._img_shape = tuple(x_shape)
        self._cond_shape = tuple(c_shape)

        self.encoder.build([x_shape, c_shape])
        x_shape, _ = self.encoder.compute_output_shape([x_shape, c_shape])

        self._latent_shape = x_shape[0]

        self.reparameterize.build(x_shape)
        x_shape = self.reparameterize.compute_output_shape(x_shape)

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

        x_shape, _ = self.encoder.compute_output_shape(
            [self._img_shape, self._cond_shape]
        )
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
            self.enc_optimizer.build(self.encoder.variables)
        if self.dec_optimizer is not None and not self.dec_optimizer.built:
            self.dec_optimizer.build(self.decoder.variables)

    def compile(
        self,
        enc_optimizer: keras.Optimizer,
        dec_optimizer: keras.Optimizer,
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

        if self.built:
            self._build_optimizers_if_needed()

    def compile_from_config(self, config):
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
        enc_optimizer = keras.optimizers.deserialize(config["enc_optimizer"])
        dec_optimizer = keras.optimizers.deserialize(config["dec_optimizer"])
        self.compile(enc_optimizer, dec_optimizer)
        self._build_optimizers_if_needed()

    def get_compile_config(self):
        """Serialize the optimizer pair so that compile_from_config can restore them."""
        return {
            "enc_optimizer": keras.optimizers.serialize(self.enc_optimizer),
            "dec_optimizer": keras.optimizers.serialize(self.dec_optimizer),
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
        mse = mse_weighted(y_true, y_pred, self.spatial_temperature)  # (B, H, W)

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

        latent_params, _ = self.encoder([x, c], training=training)
        z = self.reparameterize(latent_params)
        return self.decoder([z, c], training=training)

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
        training: Optional[bool] = None,
    ):
        self.encoder.training_enabled(training or True)
        self.decoder.training_enabled(training or True)

        (mean, logvar), embeds = self.encoder([real, cond], training=training)
        z = self.reparameterize([mean, logvar])
        rec = self.decoder([z, cond], training=training)

        loss_rec = ops.mean(
            self.reconstruction_loss(real, rec),
            axis=[1, 2],
        )
        kld = ops.mean(kl_divergence(mean, logvar), axis=[1, 2])
        elbo_real = -loss_rec - self.beta_kld * kld

        loss = -ops.mean(elbo_real)

        self.loss_rec_tracker.update_state(ops.stop_gradient(loss_rec))
        self.kld_real_tracker.update_state(ops.stop_gradient(kld))
        self.elbo_real_tracker.update_state(ops.stop_gradient(elbo_real))

        return loss, ops.stop_gradient(z), [ops.stop_gradient(e) for e in embeds]

    def train_decoder_fake_path(
        self,
        z_real: keras.KerasTensor,
        cond: keras.KerasTensor,
        training: Optional[bool] = None,
    ):
        self.encoder.training_enabled(False)
        self.decoder.training_enabled(training or True)

        z_perturbed = ops.stop_gradient(
            z_real + keras.random.normal(ops.shape(z_real)) * 0.1
        )

        fake = self.decoder([z_perturbed, cond], training=training)

        (mean, logvar), _ = self.encoder([fake, cond], training=False)
        z = self.reparameterize([mean, logvar])
        rec = self.decoder([z, cond], training=training)

        logpx_z = -ops.mean(
            self.reconstruction_loss(ops.stop_gradient(fake), rec),
            axis=[1, 2],
        )
        kld = ops.mean(kl_divergence(mean, logvar), axis=[1, 2])
        elbo = logpx_z - self.beta_kld * kld

        expelbo = ops.exp(-self.dec_expelbo_temp * elbo)
        loss = ops.mean(expelbo)

        self.elbo_fake_tracker.update_state(ops.stop_gradient(elbo))
        self.expelbo_fake_tracker.update_state(ops.stop_gradient(expelbo))

        return loss, ops.stop_gradient(fake)

    def train_decoder_rec_path(
        self,
        z_real: keras.KerasTensor,
        embeds_real: keras.KerasTensor,
        cond: keras.KerasTensor,
        training: Optional[bool] = None,
    ):
        self.encoder.training_enabled(False)
        self.decoder.training_enabled(training or True)

        rec = self.decoder([z_real, cond], training=training)

        (mean, logvar), embeds_rec = self.encoder([rec, cond], training=False)
        z_rec = self.reparameterize([mean, logvar])
        rec_rec = self.decoder([z_rec, cond], training=training)

        logpx_z = -ops.mean(
            self.reconstruction_loss(ops.stop_gradient(rec), rec_rec),
            axis=[1, 2],
        )
        kld = ops.mean(kl_divergence(mean, logvar), axis=[1, 2])
        elbo = logpx_z - self.beta_kld * kld

        embed_loss = embedding_loss(embeds_real, embeds_rec)
        expelbo = ops.exp(-self.dec_expelbo_temp * elbo)

        loss = ops.mean(expelbo + embed_loss)

        self.elbo_rec_tracker.update_state(ops.stop_gradient(elbo))
        self.expelbo_rec_tracker.update_state(ops.stop_gradient(expelbo))
        self.loss_embed_tracker.update_state(ops.stop_gradient(embed_loss))

        return loss, ops.stop_gradient(rec)

    def train_encoder_critic(
        self,
        fake: keras.KerasTensor,
        rec: keras.KerasTensor,
        cond: keras.KerasTensor,
        training: Optional[bool] = None,
    ):
        self.encoder.training_enabled(training or True)
        self.decoder.training_enabled(False)

        (mean_fake, logvar_fake), _ = self.encoder([fake, cond], training=training)
        (mean_rec, logvar_rec), _ = self.encoder([rec, cond], training=training)

        kld_fake = ops.mean(kl_divergence(mean_fake, logvar_fake), axis=[1, 2])
        kld_rec = ops.mean(kl_divergence(mean_rec, logvar_rec), axis=[1, 2])

        expkld_fake = ops.exp(-self.enc_expkld_temp * kld_fake)
        expkld_rec = ops.exp(-self.enc_expkld_temp * kld_rec)

        loss = ops.mean(expkld_fake + expkld_rec)

        self.kld_rec_tracker.update_state(ops.stop_gradient(kld_rec))
        self.kld_fake_tracker.update_state(ops.stop_gradient(kld_fake))
        self.expkld_fake_tracker.update_state(ops.stop_gradient(expkld_fake))
        self.expkld_rec_tracker.update_state(ops.stop_gradient(expkld_rec))

        return loss

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

        output = self.train_collaborative(batch_real, batch_cond, training=False)
        del output

        return {metric.name: metric.result() for metric in self.test_metrics}

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                encoder=keras.saving.serialize_keras_object(self.encoder),
                decoder=keras.saving.serialize_keras_object(self.decoder),
                reparameterize=keras.saving.serialize_keras_object(self.reparameterize),
                beta_kld=self.beta_kld,
                enc_expkld_temp=self.enc_expkld_temp,
                dec_expelbo_temp=self.dec_expelbo_temp,
                spatial_temperature=self.spatial_temperature,
                lambda_embed=self.lambda_embed,
                alpha_ssim=self.alpha_ssim,
                ssim_kwargs=self.ssim_kwargs,
            )
        )
        return config


__all__ = ["ReversedAutoencoderBase"]
