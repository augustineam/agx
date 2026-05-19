from __future__ import annotations

import keras
import warnings

from keras import layers, ops

from agx_core.layers import Sequential, Split
from agx_core.helpers import _channel_axis


@keras.saving.register_keras_serializable(
    package="agx_core.models.reversed_autoencoder"
)
class CompositeConditionEncoder(layers.Layer):
    """General-purpose condition encoder for categorical + continuous fields.

    Encodes N categorical fields and M continuous fields into a single
    fixed-size vector via per-field embeddings/projections + learned fusion.

    Contract:
        Input:  (B, num_categorical + num_continuous)
        Output: (B, embed_dim) float tensor (LayerNorm'd)

    The embed_dim is the API contract with downstream FiLM layers.

    Args:
        vocab_sizes: List of vocabulary sizes for categorical fields.
            Example: [20, 4] → product_id (20), view_id (4)
        embed_dim: Output dimension. MUST match FiLM expectations.
        field_embed_dim: Per-field embedding dim (auto if None).

    Transfer learning:
        - Swap vocab_sizes and num_continuous freely
        - Keep embed_dim identical
        - FiLM layers stay frozen
    """

    def __init__(
        self,
        vocab_sizes: list[int],
        embed_dim: int = 64,
        field_embed_dim: int | None = None,
        name: str = "condition_encoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.vocab_sizes = list(vocab_sizes)
        self.embed_dim = embed_dim
        self.field_embed_dim = field_embed_dim

    def build(self, input_shape):
        batch_dim, ch_dim = input_shape

        self.num_categorical = len(self.vocab_sizes)
        self.num_continuous = ch_dim - self.num_categorical

        if self.num_continuous < 0:
            warnings.warn(
                f"Vocabulary size provided for {self.num_categorical} categories, but the conditional vector has {ch_dim} dimensions. "
                f"The first {ch_dim} vocabulary size(s) will be used."
            )
            self.num_continuous = 0
            self.num_categorical = ch_dim

        size_splits = [1] * self.num_categorical
        if self.num_continuous > 0:
            size_splits += [self.num_continuous]

        self.split = Split(size_splits)
        self.split.build(input_shape)

        num_fields_total = len(size_splits)

        # Per-field dim for categorical embeddings
        per_field_dim = self.field_embed_dim or max(
            8, self.embed_dim // num_fields_total
        )

        # Categorical: one embedding table per field
        self.field_embeddings = []
        for i, vocab_size in enumerate(self.vocab_sizes[: self.num_categorical]):
            emb = layers.Embedding(vocab_size, per_field_dim, name=f"field_{i}_embed")
            emb.build((batch_dim,))
            self.field_embeddings.append(emb)

        # Continuous: project to same scale as categorical embeddings
        if self.num_continuous > 0:
            self.continuous_proj = Sequential(
                layers.Dense(per_field_dim, activation="relu"),
                layers.Dense(per_field_dim),
                name="continuous_proj",
            )
            self.continuous_proj.build((batch_dim, self.num_continuous))

        # Fuse all fields → fixed embed_dim
        self.fuse = Sequential(
            layers.Concatenate(),
            layers.Dense(self.embed_dim, activation="relu"),
            layers.Dense(self.embed_dim),
            layers.LayerNormalization(),
            name="fuse",
        )
        self.fuse.build([(batch_dim, per_field_dim) for _ in range(num_fields_total)])

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.embed_dim)

    def call(self, condition, training=None):
        """
        Args:
            condition: (B, num_categorical + num_continuous)

        Returns:
            (B, embed_dim) conditioning vector
        """
        fields = self.split(condition)

        parts = []
        # Embed each categorical field
        for field, emb in zip(fields, self.field_embeddings):
            parts.append(emb(ops.squeeze(field, axis=-1), training=training))

        if self.num_continuous > 0:
            parts.append(self.continuous_proj(fields[-1], training=training))

        return self.fuse(parts, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            vocab_sizes=self.vocab_sizes,
            embed_dim=self.embed_dim,
            field_embed_dim=self.field_embed_dim,
        )
        return config


__all__ = ["CompositeConditionEncoder"]
