import keras

from typing import Sequence, Optional

from keras import metrics, Model, ops

from .base import BaseEncoder, BaseDecoder
from .layers import Reparameterization, _axis_channel


def kl_divergence(mean, logvar):
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
    return 0.5 * ops.mean(
        ops.square(mean) + ops.exp(logvar) - logvar - 1.0,
        axis=_axis_channel(),
    )


# def embedding_loss(
#     teacher_embedds: Sequence[keras.KerasTensor],
#     student_embedds: Sequence[keras.KerasTensor],
# ):
#     """Feature consistency loss with depth-weighted layers.

#     Deeper layers (closer to bottleneck resolution) get higher weight
#     because their information is preservable through the VAE bottleneck.
#     Shallow layers get lower weight because their high-resolution
#     spatial detail is fundamentally lost in the bottleneck — demanding
#     their reconstruction wastes gradient energy on an impossible target.
#     """
#     n_layers = len(teacher_embedds)
#     total_loss = 0

#     for i, (teacher_feature, student_feature) in enumerate(
#         zip(teacher_embedds, student_embedds)
#     ):
#         # Exponential weighting: deepest layer = 1.0, shallowest ≈ 0
#         # For 5 layers: weights ≈ [0.06, 0.12, 0.25, 0.50, 1.00]
#         depth_weight = 2.0 ** (i - n_layers + 1)

#         mse = ops.mean(
#             ops.square(teacher_feature - student_feature), axis=_axis_channel()
#         )
#         cosine_similarity = keras.losses.cosine_similarity(
#             teacher_feature, student_feature, axis=_axis_channel()
#         )
#         layer_loss = ops.mean(0.5 * mse + (1 - cosine_similarity), axis=[1, 2])
#         total_loss += depth_weight * layer_loss

#     # Normalize by sum of weights so lambda_embed remains interpretable
#     weight_sum = sum(2.0 ** (i - n_layers + 1) for i in range(n_layers))
#     return total_loss / weight_sum


def embedding_loss(
    teacher_embedds: Sequence[keras.KerasTensor],
    student_embedds: Sequence[keras.KerasTensor],
):
    """Feature consistency loss (MSE + Cosine Similarity) between two sets of embeddings."""
    total_loss = 0
    scale = 1.0 / len(teacher_embedds)
    for teacher_feature, student_feature in zip(teacher_embedds, student_embedds):
        mse = ops.mean(
            ops.square(teacher_feature - student_feature), axis=_axis_channel()
        )
        # [B, H, W]
        cosine_similarity = keras.losses.cosine_similarity(
            teacher_feature, student_feature, axis=_axis_channel()
        )
        total_loss += ops.mean(0.5 * mse + (1 - cosine_similarity), axis=[1, 2])
    return scale * total_loss


def pixel_mse(y_true, y_pred):
    return ops.mean(ops.square(y_true - y_pred), axis=_axis_channel())


@keras.saving.register_keras_serializable(package="agx_core.models.ra")
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
        expelbo_temp: float = 1.0,
        lambda_embed: float = 1.0,
        freeze_backbone: bool = True,
        name: str = "reversed_autoencoder",
        **kwargs,
    ):
        super(ReversedAutoencoderBase, self).__init__(name=name, **kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.reparameterize = reparameterize
        self.freeze_backbone = freeze_backbone
        self.beta_kld = beta_kld
        self.expelbo_temp = expelbo_temp
        self.lambda_embed = lambda_embed

        # Two independent optimizers — set via compile()
        self.enc_optimizer: Optional[keras.optimizers.Optimizer] = None
        self.dec_optimizer: Optional[keras.optimizers.Optimizer] = None

        # Turn-taking flags — controlled by AdversarialEquilibriumCallback
        self.train_encoder_enabled = True
        self.train_decoder_enabled = True

        self.loss_enc_tracker = metrics.Mean("loss_enc")
        self.loss_dec_tracker = metrics.Mean("loss_dec")
        self.loss_rec_tracker = metrics.Mean("loss_rec")
        self.loss_embed_tracker = metrics.Mean("loss_embed")
        self.kld_real_tracker = metrics.Mean("kld_real")
        self.kld_rec_tracker = metrics.Mean("kld_rec")
        self.kld_fake_tracker = metrics.Mean("kld_fake")
        self.diff_kld_tracker = metrics.Mean("diff_kld")
        self.elbo_real_tracker = metrics.Mean("elbo_real")
        self.elbo_rec_tracker = metrics.Mean("elbo_rec")
        self.elbo_fake_tracker = metrics.Mean("elbo_fake")
        self.expelbo_rec_tracker = metrics.Mean("expelbo_rec")
        self.expelbo_fake_tracker = metrics.Mean("expelbo_fake")

    @property
    def metrics(self):
        """Return list of all tracked metrics.

        Returns:
            List of metric trackers for monitoring training progress
        """
        return [
            self.loss_enc_tracker,
            self.loss_dec_tracker,
            self.loss_rec_tracker,
            self.loss_embed_tracker,
            self.kld_real_tracker,
            self.kld_rec_tracker,
            self.kld_fake_tracker,
            self.diff_kld_tracker,
            self.elbo_real_tracker,
            self.elbo_rec_tracker,
            self.elbo_fake_tracker,
            self.expelbo_rec_tracker,
            self.expelbo_fake_tracker,
        ]

    def noise(self, batch_size) -> keras.KerasTensor:
        if self._latent_shape is None:
            raise ValueError(
                "Make sure to build the encoder or to set the correct latent space shape"
            )
        shape = list(self._latent_shape)
        shape[0] = batch_size
        return keras.random.normal(tuple(shape))

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

        super(ReversedAutoencoderBase, self).build(input_shape)

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
        enc_optimizer: keras.optimizers.Optimizer,
        dec_optimizer: keras.optimizers.Optimizer,
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
        super(ReversedAutoencoderBase, self).compile(optimizer=None, **kwargs)
        self.enc_optimizer = enc_optimizer
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
        latent_params, _ = self.encoder([x, c], training=training)
        z = self.reparameterize(latent_params)
        return self.decoder([z, c], training=training)

    def compute_encoder_loss(self, real, noise, condition):
        """Framework-agnostic forward pass and loss computation for the encoder."""
        fake = self.decoder([noise, condition], training=False)

        # Get the latent representations, embeddings and reconstructions for real samples
        (mean_real, logvar_real), embeds_real = self.encoder(
            [real, condition], training=True
        )
        z_real = self.reparameterize([mean_real, logvar_real])
        rec_real = self.decoder([z_real, condition], training=False)

        # Get the latent representations, embeddings and reconstructions for reconstructed samples
        (mean_rec, logvar_rec), _ = self.encoder(
            [ops.stop_gradient(rec_real), condition], training=True
        )
        z_rec = self.reparameterize([mean_rec, logvar_rec])
        rec_rec = self.decoder([z_rec, condition], training=False)

        # Get the latent representations and reconstructions for fake samples
        (mean_fake, logvar_fake), _ = self.encoder(
            [ops.stop_gradient(fake), condition], training=True
        )
        z_fake = self.reparameterize([mean_fake, logvar_fake])
        rec_fake = self.decoder([z_fake, condition], training=False)

        # For real samples, we want to maximize the ELBO by minimizing the negative
        # ELBO for the real samples `elbo_real`.

        # [N], [N], [N], [N], [N]
        logpx_z_real = -ops.mean(pixel_mse(real, rec_real), axis=[1, 2])
        kld_real = ops.mean(kl_divergence(mean_real, logvar_real), axis=[1, 2])
        elbo_real = logpx_z_real - self.beta_kld * kld_real

        # For fake and reconstructed samples, we want to minimize the ELBO
        # for the fake and reconstructed samples `elbo_fake` and `elbo_rec`,
        # respectively, so that the encoder learns to discriminate the
        # reconstructed and fake samples.

        # [N], [N], [N], [N]
        logpx_z_rec = -ops.mean(
            pixel_mse(ops.stop_gradient(rec_real), rec_rec), axis=[1, 2]
        )
        kld_rec = ops.mean(kl_divergence(mean_rec, logvar_rec), axis=[1, 2])
        elbo_rec = logpx_z_rec - self.beta_kld * kld_rec

        # [N], [N], [N], [N]
        logpx_z_fake = -ops.mean(
            pixel_mse(ops.stop_gradient(fake), rec_fake), axis=[1, 2]
        )
        kld_fake = ops.mean(kl_divergence(mean_fake, logvar_fake), axis=[1, 2])
        elbo_fake = logpx_z_fake - self.beta_kld * kld_fake

        # Exponential curriculum weighting: focus training on hardest-to-discriminate samples
        # Good fakes (higher ELBO, closer to 0) get exponentially higher loss contribution
        # Bad fakes (lower ELBO, more negative) get exponentially lower loss contribution
        # This creates adaptive curriculum learning where the encoder progressively focuses
        # on subtler discrimination tasks as the decoder improves

        # [N], [N]
        expelbo_rec = ops.exp(self.expelbo_temp * elbo_rec)
        expelbo_fake = ops.exp(self.expelbo_temp * elbo_fake)

        loss = ops.mean(-elbo_real + 0.5 * (expelbo_rec + expelbo_fake))

        # Track KLD gap between fake and real samples as an adversarial
        # equilibrium diagnostic:
        #   diff_kld = kld_fake - kld_real
        #
        #   > 0 (large): Encoder dominates — fakes need far more specialized
        #                encodings than reals. Decoder is underperforming.
        #   > 0 (small): Approaching equilibrium — decoder produces fakes
        #                that are nearly as "normal" as reals to the encoder.
        #   ≈ 0:         Nash equilibrium — encoder cannot distinguish fakes
        #                from reals via KLD alone. Decoder has succeeded.
        #   < 0:         Pathological — encoder finds fakes more normal than
        #                reals. Indicates training instability.

        aux_outputs = (
            ops.stop_gradient(z_real),
            [ops.stop_gradient(embed) for embed in embeds_real],
            ops.stop_gradient(kld_real),
        )

        metric_updates = dict(
            loss_enc=loss,
            loss_rec=-logpx_z_real,
            kld_real=kld_real,
            kld_rec=kld_rec,
            kld_fake=kld_fake,
            elbo_real=elbo_real,
            elbo_rec=elbo_rec,
            elbo_fake=elbo_fake,
            expelbo_rec=expelbo_rec,
            expelbo_fake=expelbo_fake,
            diff_kld=kld_fake - kld_real,
        )

        return loss, aux_outputs, metric_updates

    def compute_decoder_loss(
        self,
        real: keras.KerasTensor,
        noise: keras.KerasTensor,
        condition: keras.KerasTensor,
        z_real: keras.KerasTensor,
        embeds_real: keras.KerasTensor,
        kld_real: keras.KerasTensor,
    ):
        """Train the decoder to generate realistic samples and fool the encoder.

        The decoder is trained to:
        - Maximize ELBO on real samples to minimize the reconstruction loss
        - Maximize ELBO on reconstructed and fake samples (fool the encoder)
        - Minimize embedding loss to maintain feature consistency

        Args:
            real: Batch of real input images
            noise: Random noise for generating fake samples
            condition: Conditioning information
            z_real: Latent codes from real samples (from encoder training)
            embeds_real: Feature embeddings from real samples
            kld_real: KL divergence from real samples
        """
        # Generate fake samples and reconstruct real samples
        fake = self.decoder([noise, condition], training=True)
        rec_real = self.decoder([ops.stop_gradient(z_real), condition], training=True)

        # Get latent representations and embeddings for reconstructed samples
        (mean_rec, logvar_rec), embeds_rec = self.encoder(
            [rec_real, condition], training=False
        )
        z_rec = self.reparameterize([mean_rec, logvar_rec])
        rec_rec = self.decoder([ops.stop_gradient(z_rec), condition], training=True)

        # Get latent representations for fake samples
        (mean_fake, logvar_fake), _ = self.encoder([fake, condition], training=False)
        z_fake = self.reparameterize([mean_fake, logvar_fake])
        rec_fake = self.decoder([ops.stop_gradient(z_fake), condition], training=True)

        # Embedding loss (decoder only): measures feature consistency between
        # E(I) and E(I') in the encoder's native feature space.
        #
        # The encoder acts as a frozen perceptual critic — its intermediate
        # features define the space where we measure reconstruction fidelity.
        # By training only the decoder with this signal, we preserve the
        # encoder's anomaly sensitivity: it remains free to produce divergent
        # embeddings for anomalous inputs, while the decoder is pushed to
        # always reconstruct "healthy" images that re-encode faithfully.
        #
        # At inference, anomaly score = distance(E(I), E(I')), which is
        # high precisely because the encoder was never trained to suppress
        # that difference.
        # [N]
        embeds_real = [ops.stop_gradient(embed) for embed in embeds_real]
        embed_loss = embedding_loss(embeds_real, embeds_rec)

        # Here we want to maximize the ELBOs for the real and fake samples,
        # to guide the decoder to generate realistic samples.

        # [N], [N]
        logpx_z_real = -ops.mean(pixel_mse(real, rec_real), axis=[1, 2])
        elbo_real = logpx_z_real - self.beta_kld * ops.stop_gradient(kld_real)

        # [N], [N], [N], [N], [N]
        logpx_z_rec = -ops.mean(
            pixel_mse(ops.stop_gradient(rec_real), rec_rec), axis=[1, 2]
        )
        kld_rec = ops.mean(kl_divergence(mean_rec, logvar_rec), axis=[1, 2])
        elbo_rec = logpx_z_rec - self.beta_kld * kld_rec

        # [N], [N], [N], [N], [N]
        logpx_z_fake = -ops.mean(
            pixel_mse(ops.stop_gradient(fake), rec_fake), axis=[1, 2]
        )
        kld_fake = ops.mean(kl_divergence(mean_fake, logvar_fake), axis=[1, 2])
        elbo_fake = logpx_z_fake - self.beta_kld * kld_fake

        loss = ops.mean(
            -elbo_real - 0.5 * (elbo_rec + elbo_fake) + self.lambda_embed * embed_loss
        )

        return loss, dict(loss_dec=loss, loss_embed=embed_loss)

    def train_encoder(
        self,
        real: keras.KerasTensor,
        noise: keras.KerasTensor,
        condition: keras.KerasTensor,
    ):
        """Train the encoder to distinguish real from fake samples.

        The encoder is trained to:
        - Maximize ELBO on real samples (minimize negative ELBO)
        - Minimize ELBO on reconstructed and fake samples (discriminate against them)
        - Minimize embedding loss to maintain feature consistency

        Args:
            real: Batch of real input images
            noise: Random noise for generating fake samples
            condition: Conditioning information

        Returns:
            Tuple of (latent_codes_real, embeddings_real, kld_real)
        """
        raise NotImplementedError(
            "Gradient application must be implemented by backend subclasses."
        )

    def train_decoder(
        self,
        real: keras.KerasTensor,
        noise: keras.KerasTensor,
        condition: keras.KerasTensor,
        z_real: keras.KerasTensor,
        embeds_real: keras.KerasTensor,
        kld_real: keras.KerasTensor,
    ):
        """Train the decoder to generate realistic samples and fool the encoder.

        The decoder is trained to:
        - Maximize ELBO on real samples to minimize the reconstruction loss
        - Maximize ELBO on reconstructed and fake samples (fool the encoder)
        - Minimize embedding loss to maintain feature consistency

        Args:
            real: Batch of real input images
            noise: Random noise for generating fake samples
            condition: Conditioning information
            z_real: Latent codes from real samples (from encoder training)
            embeds_real: Feature embeddings from real samples
            kld_real: KL divergence from real samples
        """
        raise NotImplementedError(
            "Gradient application must be implemented by backend subclasses."
        )

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

        (batch_real, batch_cond), _ = data

        batch_size = ops.shape(batch_real)[0]
        batch_noise = self.noise(batch_size)

        z_real, embeds_real, kld_real = self.train_encoder(
            batch_real, batch_noise, batch_cond
        )
        self.train_decoder(
            batch_real, batch_noise, batch_cond, z_real, embeds_real, kld_real
        )

        return self.get_metrics_result()

    def test_step(self, data):
        """Perform evaluation step computing all metrics without training.

        Args:
            data: Validation batch containing (images, conditions, sample_weights)

        Returns:
            Dictionary of evaluation metrics
        """

        (real, cond), _ = data

        batch_size = ops.shape(real)[0]
        noise = self.noise(batch_size)

        # Generate fake samples
        fake = self.decoder([noise, cond], training=False)

        # Get latent representations, embeddings and reconstruction for real samples
        (mean_real, logvar_real), embeds_real = self.encoder(
            [real, cond], training=False
        )
        z_real = self.reparameterize([mean_real, logvar_real])
        rec_real = self.decoder([z_real, cond], training=False)

        # Get mu, logvar parameters of the latent representations and embeddings for the real reconstructed samples
        (mean_rec, logvar_rec), embeds_rec = self.encoder(
            [rec_real, cond], training=False
        )
        z_rec = self.reparameterize([mean_rec, logvar_rec])
        rec_rec = self.decoder([z_rec, cond], training=False)

        # Get mu, logvar parameters of the latent representations of the fake samples
        (mean_fake, logvar_fake), _ = self.encoder([fake, cond], training=False)
        z_fake = self.reparameterize([mean_fake, logvar_fake])
        rec_fake = self.decoder([z_fake, cond], training=False)

        # Compute embedding loss between the real and reconstructed samples
        loss_embed = embedding_loss(embeds_real, embeds_rec)

        # [N], [N], [N], [N], [N]
        kld_real = ops.mean(kl_divergence(mean_real, logvar_real), axis=[1, 2])
        kld_rec = ops.mean(kl_divergence(mean_rec, logvar_rec), axis=[1, 2])
        kld_fake = ops.mean(kl_divergence(mean_fake, logvar_fake), axis=[1, 2])

        logpx_z_real = -ops.mean(pixel_mse(real, rec_real), axis=[1, 2])
        elbo_real = logpx_z_real - self.beta_kld * kld_real

        logpx_z_rec = -ops.mean(
            pixel_mse(ops.stop_gradient(rec_real), rec_rec), axis=[1, 2]
        )
        elbo_rec = logpx_z_rec - self.beta_kld * kld_rec

        logpx_z_fake = -ops.mean(
            pixel_mse(ops.stop_gradient(fake), rec_fake), axis=[1, 2]
        )
        elbo_fake = logpx_z_fake - self.beta_kld * kld_fake

        # [N], [N]
        expelbo_rec = ops.exp(self.expelbo_temp * elbo_rec)
        expelbo_fake = ops.exp(self.expelbo_temp * elbo_fake)

        loss_enc = ops.mean(-elbo_real + 0.5 * (expelbo_rec + expelbo_fake))
        loss_dec = ops.mean(
            -elbo_real - 0.5 * (elbo_rec + elbo_fake) + self.lambda_embed * loss_embed
        )

        self.elbo_real_tracker.update_state(elbo_real)
        self.elbo_rec_tracker.update_state(elbo_rec)
        self.elbo_fake_tracker.update_state(elbo_fake)
        self.expelbo_rec_tracker.update_state(expelbo_rec)
        self.expelbo_fake_tracker.update_state(expelbo_fake)
        self.loss_enc_tracker.update_state(loss_enc)
        self.loss_dec_tracker.update_state(loss_dec)
        self.loss_rec_tracker.update_state(-logpx_z_real)
        self.loss_embed_tracker.update_state(loss_embed)
        self.kld_real_tracker.update_state(kld_real)
        self.kld_rec_tracker.update_state(kld_rec)
        self.kld_fake_tracker.update_state(kld_fake)
        self.diff_kld_tracker.update_state(kld_fake - kld_real)

        return dict(
            elbo_real=self.elbo_real_tracker.result(),
            elbo_rec=self.elbo_rec_tracker.result(),
            elbo_fake=self.elbo_fake_tracker.result(),
            expelbo_rec=self.expelbo_rec_tracker.result(),
            expelbo_fake=self.expelbo_fake_tracker.result(),
            loss_enc=self.loss_enc_tracker.result(),
            loss_dec=self.loss_dec_tracker.result(),
            loss_rec=self.loss_rec_tracker.result(),
            loss_embed=self.loss_embed_tracker.result(),
            diff_kld=self.diff_kld_tracker.result(),
            kld_real=self.kld_real_tracker.result(),
            kld_rec=self.kld_rec_tracker.result(),
            kld_fake=self.kld_fake_tracker.result(),
        )

    def get_config(self):
        config = super(ReversedAutoencoderBase, self).get_config()
        config.update(
            dict(
                encoder=keras.saving.serialize_keras_object(self.encoder),
                decoder=keras.saving.serialize_keras_object(self.decoder),
                reparameterize=keras.saving.serialize_keras_object(self.reparameterize),
                beta_kld=self.beta_kld,
                expelbo_temp=self.expelbo_temp,
                lambda_embed=self.lambda_embed,
                freeze_backbone=self.freeze_backbone,
            )
        )
        return config


__all__ = ["ReversedAutoencoderBase"]
