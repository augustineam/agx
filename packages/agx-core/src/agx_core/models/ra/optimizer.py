import keras
import warnings

from typing import Any

def _flatten_optimizer_config(optimizer_config: dict[str, Any], prefix: str) -> dict:
    """Flatten optimizer configuration for MLflow tracking."""
    flattened = {}

    # Add top-level keys
    for key in ["module", "class_name", "registered_name"]:
        if key in optimizer_config:
            flattened[f"{prefix}/{key}"] = optimizer_config[key]

    # Flatten config dict
    if "config" in optimizer_config:
        config: dict[str, Any] = optimizer_config["config"]
        for key, value in config.items():
            flattened[f"{prefix}/config/{key}"] = value

    return flattened


def _unflatten_optimizer_config(flattened_config: dict[str, Any], prefix: str) -> dict:
    """Static version of unflatten method for use in from_config."""
    optimizer_config = {
        "module": None,
        "class_name": None,
        "config": {},
        "registered_name": None,
    }

    prefix_with_slash = f"{prefix}/"

    for key, value in flattened_config.items():
        if key.startswith(prefix_with_slash):
            # Remove prefix
            relative_key = key[len(prefix_with_slash) :]

            if relative_key in ["module", "class_name", "registered_name"]:
                optimizer_config[relative_key] = value
            elif relative_key.startswith("config/"):
                config_key = relative_key[7:]  # Remove 'config/'
                optimizer_config["config"][config_key] = value

    return optimizer_config


@keras.saving.register_keras_serializable(package="kssaiml.models.reversed_autoencoder")
class RAOptimizer(keras.optimizers.Optimizer):
    """A custom optimizer wrapper for Reversed Autoencoder training.

    This class extends keras.optimizers.Optimizer and manages two separate
    optimizers internally: one for the encoder and one for the decoder.
    It properly handles state saving/loading for checkpoint compatibility.
    """

    def __init__(
        self,
        enc_optimizer: keras.optimizers.Optimizer,
        dec_optimizer: keras.optimizers.Optimizer,
        name: str = "RAOptimizer",
        **kwargs,
    ):
        """Initialize the RAOptimizer with encoder and decoder optimizers.

        Args:
            enc_optimizer: The optimizer to use for training the encoder network.
            dec_optimizer: The optimizer to use for training the decoder network.
            name: Name for this optimizer instance.
        """
        super().__init__(0.0, name=name, **kwargs)
        self._enc_optimizer = enc_optimizer
        self._dec_optimizer = dec_optimizer

    @property
    def enc(self):
        """Get the encoder optimizer."""
        return self._enc_optimizer

    @property
    def dec(self):
        """Get the decoder optimizer."""
        return self._dec_optimizer

    @property
    def variables(self):
        """Return all optimizer variables from both internal optimizers."""
        return self._enc_optimizer.variables + self._dec_optimizer.variables

    def build(self, var_list):
        """Build the optimizer by building both internal optimizers."""

        # NOTE: We need to make sure each optimizer is built with the correct variables.
        # This function is called when the optimizer is being loaded from a state file,
        # and before load_own_variables is called.

        enc_vars = [var for var in var_list if var.path.startswith("reversed_autoencoder/encoder")]
        dec_vars = [var for var in var_list if var.path.startswith("reversed_autoencoder/decoder")]

        self._enc_optimizer.build(enc_vars)
        self._dec_optimizer.build(dec_vars)

        self.built = True

    def update_step(self, gradient, variable):
        """This method is required by the base class but not used directly."""
        # This is called by the base apply_gradients, but we override that method
        raise NotImplementedError("update_step should not be called directly.")

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        """Apply gradients using the appropriate optimizer based on variable names."""
        # This method should be called directly from your training step
        # The model's train_step handles the separation of encoder/decoder gradients
        raise NotImplementedError("apply_gradients should not be called directly.")

    def get_config(self):
        """Return the flattened configuration of the optimizer."""
        enc_config = keras.optimizers.serialize(self._enc_optimizer)
        dec_config = keras.optimizers.serialize(self._dec_optimizer)

        flattened_config = {}
        flattened_config.update(_flatten_optimizer_config(enc_config, "enc_optimizer"))
        flattened_config.update(_flatten_optimizer_config(dec_config, "dec_optimizer"))

        return flattened_config

    @classmethod
    def from_config(cls, config):
        """Create optimizer from flattened configuration."""
        # Separate encoder and decoder configs
        enc_keys = {k: v for k, v in config.items() if k.startswith("enc_optimizer/")}
        dec_keys = {k: v for k, v in config.items() if k.startswith("dec_optimizer/")}

        # Get remaining config (name, etc.)
        remaining_config = {
            k: v
            for k, v in config.items()
            if not k.startswith("enc_optimizer/") and not k.startswith("dec_optimizer/")
        }

        # Unflatten optimizer configs
        enc_config = _unflatten_optimizer_config(enc_keys, "enc_optimizer")
        dec_config = _unflatten_optimizer_config(dec_keys, "dec_optimizer")

        # Create optimizers
        enc_optimizer = keras.optimizers.deserialize(enc_config)
        dec_optimizer = keras.optimizers.deserialize(dec_config)

        return cls(enc_optimizer, dec_optimizer, **remaining_config)

    def set_weights(self, weights):
        """Set the weights of both internal optimizers."""
        print("Setting optimizer weights")
        return super().set_weights(weights)

    def save_own_variables(self, store):
        """Get the state of this optimizer object."""
        for i, variable in enumerate(self._enc_optimizer.variables):
            store[f"enc_{i}"] = variable.numpy()

        for i, variable in enumerate(self._dec_optimizer.variables):
            store[f"dec_{i}"] = variable.numpy()

    def load_own_variables(self, store: dict[str, Any]):
        """Set the state of this optimizer object."""
        # Separate encoder and decoder keys
        enc_vars: dict[str, Any] = {}
        dec_vars: dict[str, Any] = {}

        for key, value in store.items():
            if key.startswith("enc_"):
                # Remove the "enc_" prefix to get the original
                original_key = key[4:]
                enc_vars[original_key] = value
            elif key.startswith("dec_"):
                # Remove the "dec_" prefix to get the
                original_key = key[4:]
                dec_vars[original_key] = value

        if len(enc_vars.keys()) != len(self._enc_optimizer.variables):
            msg = (
                f"Skipping variable loading for encoder optimizer '{self._enc_optimizer.name}', "
                f"because it has {len(self._enc_optimizer.variables)} variables whereas "
                f"the saved optimizer has {len(enc_vars.keys())} variables. "
            )
            if len(self._enc_optimizer.variables) == 0:
                msg += (
                    "This is likely because the optimizer has not been "
                    "called/built yet."
                )
            warnings.warn(msg, stacklevel=2)
        else:
            for i, variable in enumerate(self._enc_optimizer.variables):
                variable.assign(enc_vars[str(i)])

        if len(dec_vars.keys()) != len(self._dec_optimizer.variables):
            msg = (
                f"Skipping variable loading for decoder optimizer '{self._dec_optimizer.name}', "
                f"because it has {len(self._dec_optimizer.variables)} variables whereas "
                f"the saved optimizer has {len(dec_vars.keys())} variables. "
            )
            if len(self._dec_optimizer.variables) == 0:
                msg += (
                    "This is likely because the optimizer has not been "
                    "called/built yet."
                )
            warnings.warn(msg, stacklevel=2)
        else:
            for i, variable in enumerate(self._dec_optimizer.variables):
                variable.assign(dec_vars[str(i)])