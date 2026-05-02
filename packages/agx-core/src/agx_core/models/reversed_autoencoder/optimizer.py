# RAOptimizer has been removed.
#
# The two optimizers (encoder / decoder) are now stored directly on
# ReversedAutoencoderBase as ``enc_optimizer`` and ``dec_optimizer``.
# Pass them to ``model.compile(enc_optimizer=..., dec_optimizer=...)``.
# Serialization is handled via ``get_compile_config`` / ``compile_from_config``.
