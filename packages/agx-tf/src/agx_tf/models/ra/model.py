import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf

from agx_core.models.reversed_autoencoder import ReversedAutoencoderBase


@keras.saving.register_keras_serializable(package="agx_tf.models.reversed_autoencoder")
class ReversedAutoencoder(ReversedAutoencoderBase):

    def train_encoder(self, real, noise, condition):
        self.encoder.trainable = True
        self.decoder.trainable = False

        with tf.GradientTape() as tape:
            loss, aux_outputs, metric_updates = self.compute_encoder_loss(
                real, noise, condition
            )

        grads = tape.gradient(loss, self.encoder.trainable_variables)
        self.enc_optimizer.apply_gradients(zip(grads, self.encoder.trainable_variables))

        self.update_step_metrics(metric_updates)
        return aux_outputs

    def train_decoder(self, real, noise, condition, z_real, embeds_real, kld_real):
        self.encoder.trainable = False
        self.decoder.trainable = True

        with tf.GradientTape() as tape:
            loss, metric_updates = self.compute_decoder_loss(
                real, noise, condition, z_real, embeds_real, kld_real
            )

        grads = tape.gradient(loss, self.decoder.trainable_variables)
        self.dec_optimizer.apply_gradients(zip(grads, self.decoder.trainable_variables))

        self.update_step_metrics(metric_updates)
