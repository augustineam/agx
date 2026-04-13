import keras
import numpy as np

from typing import Sequence, Optional, Dict, Any

from keras import metrics, Model, ops

from .base import BaseEncoder, BaseDecoder
from .layers import Reparameterization
from .optimizer import RAOptimizer


def log_normal_pdf(sample, mean, logvar, axis=1):
    """Computes log PDF of samples under a normal distribution."""
    log2pi = ops.log(2.0 * np.pi)
    return ops.sum(
        -0.5 * (ops.square(sample - mean) * ops.exp(-logvar) + logvar + log2pi),
        axis=axis,
    )


def embedding_loss(
    teacher_embedds: Sequence[keras.KerasTensor],
    student_embedds: Sequence[keras.KerasTensor],
):
    """Feature consistency loss (MSE + Cosine Similarity) between two sets of embeddings."""
    total_loss = 0
    scale = 1.0 / len(teacher_embedds)
    for teacher_feature, student_feature in zip(teacher_embedds, student_embedds):
        mse = keras.losses.mean_squared_error(teacher_feature, student_feature)
        cosine_similarity = keras.losses.cosine_similarity(
            teacher_feature, student_feature
        )
        total_loss += ops.mean(0.5 * mse + (1 - cosine_similarity), axis=[1, 2])
    return scale * total_loss


@keras.saving.register_keras_serializable(package="agx_core.models.ra")
class ReversedAutoencoder(Model):
    """Reversed Autoencoder orchestrating adversarial training between encoder and decoder.

    The encoder and decoder are instantiated externally and passed to this model.
    """

    optimizer: RAOptimizer

    def __init__(
        self,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        scale: Optional[float] = None,
        name: str = "reversed_autoencoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.encoder = encoder
        self.decoder = decoder

        self.reparameterize = Reparameterization()

        # Adaptive scaling factor to maintain consistent gradient energy across image resolutions
        self.scale = scale

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

    def build(self, input_shape: Sequence[Sequence[int]]):
        x_shape, c_shape = input_shape

        self.encoder.build([x_shape, c_shape])

        # Calculate dynamic scale if not provided based on spatial dimension
        if self.scale is None:
            self.scale = 32 / np.prod(x_shape[1:3]) ** 0.5

        self.built = True

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
        (mean_rec, logvar_rec), embeds_rec = self.encoder(
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

        # The embedding loss is to make sure that the embeddings of the real
        # samples and the reconstructed samples are close to each other,
        # which means that the features of the real samples and the reconstructed
        # samples are close to each other.

        # [N]
        embed_loss = self.scale * embedding_loss(embeds_real, embeds_rec)

        # For real samples, we want to maximize the ELBO by minimizing the negative
        # ELBO for the real samples `elbo_real`.

        # [N], [N], [N], [N], [N]
        logpx_z_real = -self.scale * ops.sum(self.loss(real, rec_real), axis=[1, 2])
        logpz_real = log_normal_pdf(z_real, 0.0, 0.0, axis=[1, 2, 3])
        logqz_x_real = log_normal_pdf(z_real, mean_real, logvar_real, axis=[1, 2, 3])
        kld_real = self.scale * (logqz_x_real - logpz_real)
        elbo_real = logpx_z_real - kld_real

        # For fake and reconstructed samples, we want to minimize the ELBO
        # for the fake and reconstructed samples `elbo_fake` and `elbo_rec`,
        # respectively, so that the encoder learns to discriminate the
        # reconstructed and fake samples.

        # [N], [N], [N], [N]
        logpx_z_rec = -self.scale * ops.sum(
            self.loss(ops.stop_gradient(rec_real), rec_rec), axis=[1, 2]
        )
        logpz_rec = log_normal_pdf(z_rec, 0.0, 0.0, axis=[1, 2, 3])
        logqz_x_rec = log_normal_pdf(z_rec, mean_rec, logvar_rec, axis=[1, 2, 3])
        kld_rec = self.scale * (logqz_x_rec - logpz_rec)
        elbo_rec = logpx_z_rec - kld_rec

        # [N], [N], [N], [N]
        logpx_z_fake = -self.scale * ops.sum(
            self.loss(ops.stop_gradient(fake), rec_fake), axis=[1, 2]
        )
        logpz_fake = log_normal_pdf(z_fake, 0.0, 0.0, axis=[1, 2, 3])
        logqz_x_fake = log_normal_pdf(z_fake, mean_fake, logvar_fake, axis=[1, 2, 3])
        kld_fake = self.scale * (logqz_x_fake - logpz_fake)
        elbo_fake = logpx_z_fake - kld_fake

        # Exponential curriculum weighting: focus training on hardest-to-discriminate samples
        # Good fakes (higher ELBO, closer to 0) get exponentially higher loss contribution
        # Bad fakes (lower ELBO, more negative) get exponentially lower loss contribution
        # This creates adaptive curriculum learning where the encoder progressively focuses
        # on subtler discrimination tasks as the decoder improves

        # ELBO values are negative (-∞ to 0), exp(scale * elbo) weights by difficulty
        # scale is adapted by image size to maintain consistent gradient energy

        # [N], [N]
        expelbo_rec = -elbo_rec * ops.exp(self.scale * elbo_rec)
        expelbo_fake = -elbo_fake * ops.exp(self.scale * elbo_fake)

        loss = ops.mean(-elbo_real + 0.5 * (expelbo_rec + expelbo_fake) + embed_loss)

        # Keep track of the KL divergence between the real and fake samples
        # to see who's winning the game between the encoder and decoder.
        # Encoder is winning when real KLD < fake KLD
        # Decoder is winning when real KLD >= fake KLD
        # In order to generate realistic samples, the decoder should move this
        # difference close to zero.

        aux_outputs = (z_real, embeds_real, kld_real)

        metric_updates = dict(
            loss_enc=loss,
            loss_embed=embed_loss,
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

        # [N]
        embeds_real = [ops.stop_gradient(embed) for embed in embeds_real]
        embed_loss = self.scale * embedding_loss(embeds_real, embeds_rec)

        # Here we want to maximize the ELBOs for the real and fake samples,
        # to guide the decoder to generate realistic samples.

        # [N], [N]
        logpx_z_real = -self.scale * ops.sum(self.loss(real, rec_real), axis=[1, 2])
        elbo_real = logpx_z_real - ops.stop_gradient(kld_real)

        # [N], [N], [N], [N], [N]
        logpx_z_rec = -self.scale * ops.sum(
            self.loss(ops.stop_gradient(rec_real), rec_rec), axis=[1, 2]
        )
        logpz_rec = log_normal_pdf(z_rec, 0.0, 0.0, axis=[1, 2, 3])
        logqz_x_rec = log_normal_pdf(z_rec, mean_rec, logvar_rec, axis=[1, 2, 3])
        kld_rec = self.scale * (logqz_x_rec - logpz_rec)
        elbo_rec = logpx_z_rec - kld_rec

        # [N], [N], [N], [N], [N]
        logpx_z_fake = -self.scale * ops.sum(
            self.loss(ops.stop_gradient(fake), rec_fake), axis=[1, 2]
        )
        logpz_fake = log_normal_pdf(z_fake, 0.0, 0.0, axis=[1, 2, 3])
        logqz_x_fake = log_normal_pdf(z_fake, mean_fake, logvar_fake, axis=[1, 2, 3])
        kld_fake = self.scale * (logqz_x_fake - logpz_fake)
        elbo_fake = logpx_z_fake - kld_fake

        loss = ops.mean(-elbo_real - 0.5 * (elbo_rec + elbo_fake) + embed_loss)

        return loss, dict(loss_dec=loss)

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

        batch_real, batch_cond, _ = keras.utils.unpack_x_y_sample_weight(data)
        # batch_real = ops.reshape(batch_real, (-1, *self.input_shape))

        batch_size = ops.shape(batch_real)[0]
        batch_noise = self.encoder.noise(batch_size)

        z_real, embeds_real, kld_real = self.train_encoder(
            batch_real, batch_noise, batch_cond
        )
        self.train_decoder(
            batch_real, batch_noise, batch_cond, z_real, embeds_real, kld_real
        )

        # Update optimizer iterations to match encoder iterations
        # so that callbacks can work properly accessing optimizer.iterations
        if hasattr(self.optimizer, "_iterations"):
            self.optimizer.assign(
                self.optimizer._iterations, self.optimizer.enc.iterations
            )
        else:
            self.optimizer.iterations = self.optimizer.enc.iterations

        return self.get_metrics_result()

    def test_step(self, data):
        """Perform evaluation step computing all metrics without training.

        Args:
            data: Validation batch containing (images, conditions, sample_weights)

        Returns:
            Dictionary of evaluation metrics
        """

        self.encoder.trainable = False
        self.decoder.trainable = False

        real, cond, _ = keras.utils.unpack_x_y_sample_weight(data)
        # real = ops.reshape(real, (-1, *self.input_shape))

        batch_size = ops.shape(real)[0]
        noise = self.encoder.noise(batch_size)

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
        loss_embed = self.scale * embedding_loss(embeds_real, embeds_rec)

        # [N], [N], [N], [N], [N]
        logpx_z_real = -self.scale * ops.sum(self.loss(real, rec_real), axis=[1, 2])
        logpz_real = log_normal_pdf(z_real, 0.0, 0.0, axis=[1, 2, 3])
        logqz_x_real = log_normal_pdf(z_real, mean_real, logvar_real, axis=[1, 2, 3])
        kld_real = self.scale * (logqz_x_real - logpz_real)
        elbo_real = logpx_z_real - kld_real

        # [N], [N], [N], [N], [N]
        logpx_z_rec = -self.scale * ops.sum(
            self.loss(ops.stop_gradient(rec_real), rec_rec), axis=[1, 2]
        )
        logpz_rec = log_normal_pdf(z_rec, 0.0, 0.0, axis=[1, 2, 3])
        logqz_x_rec = log_normal_pdf(z_rec, mean_rec, logvar_rec, axis=[1, 2, 3])
        kld_rec = self.scale * (logqz_x_rec - logpz_rec)
        elbo_rec = logpx_z_rec - kld_rec

        # [N], [N], [N], [N], [N]
        logpx_z_fake = -self.scale * ops.sum(
            self.loss(ops.stop_gradient(fake), rec_fake), axis=[1, 2]
        )
        logpz_fake = log_normal_pdf(z_fake, 0.0, 0.0, axis=[1, 2, 3])
        logqz_x_fake = log_normal_pdf(z_fake, mean_fake, logvar_fake, axis=[1, 2, 3])
        kld_fake = self.scale * (logqz_x_fake - logpz_fake)
        elbo_fake = logpx_z_fake - kld_fake

        self.elbo_real_tracker.update_state(elbo_real)
        self.elbo_rec_tracker.update_state(elbo_rec)
        self.elbo_fake_tracker.update_state(elbo_fake)
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
            loss_rec=self.loss_rec_tracker.result(),
            loss_embed=self.loss_embed_tracker.result(),
            diff_kld=self.diff_kld_tracker.result(),
            kld_real=self.kld_real_tracker.result(),
            kld_rec=self.kld_rec_tracker.result(),
            kld_fake=self.kld_fake_tracker.result(),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                encoder=keras.saving.serialize_keras_object(self.encoder),
                decoder=keras.saving.serialize_keras_object(self.decoder),
                scale=self.scale,
            )
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        config["encoder"] = keras.saving.deserialize_keras_object(config.pop("encoder"))
        config["decoder"] = keras.saving.deserialize_keras_object(config.pop("decoder"))
        return cls(**config)

    def build_graph(self, input_shape: Sequence[Sequence[int]]):
        """Build a Keras functional model for visualization and export.

        Args:
            input_shape: Shape of input images (without batch dimension)

        Returns:
            Keras Model instance representing the autoencoder architecture
        """

        x_shape, c_shape = input_shape
        x = keras.Input(shape=x_shape, name="image")
        c = keras.Input(shape=c_shape, name="condition")

        mu, logvar = self.encoder.build_graph([x, c])
        z = self.reparameterize([mu, logvar])
        y = self.decoder.build_graph([z, c])

        return keras.Model(inputs=[x, c], outputs=y, name=self.name)


__all__ = ["ReversedAutoencoder"]
