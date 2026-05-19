from __future__ import annotations

import keras
from keras import ops

from typing import Optional


@keras.saving.register_keras_serializable(
    package="agx_core.models.reversed_autoencoder"
)
class AdversarialEquilibriumCallback(keras.callbacks.Callback):
    """Dynamically pauses encoder or decoder training to maintain adversarial equilibrium.

    Monitors an exponential moving average (EMA) of `diff_kld` (= kld_fake - kld_real)
    and applies hysteresis thresholds to decide turn-taking:

        diff_kld (EMA)
        ──────────────────────────────────────────────────
        ▲
        │   encoder dominant     ← pause encoder, train decoder only
        │ ─ ─ ─ upper ─ ─ ─ ─   (e.g. +2.0)
        │
        │   healthy zone         ← train both
        │
        │ ─ ─ ─ lower ─ ─ ─ ─   (e.g. -0.5)
        │   encoder collapsing   ← pause decoder, train encoder only
        ▼

    Once a threshold is breached, the paused component stays paused for at least
    `min_pause_steps` steps, then resumes only when the EMA re-enters the healthy
    zone (hysteresis prevents oscillation).

    Args:
        upper_threshold: EMA diff_kld above which encoder training is paused.
            Indicates encoder is dominant and decoder needs to catch up.
        lower_threshold: EMA diff_kld below which decoder training is paused.
            Indicates encoder is collapsing and needs recovery time.
        ema_momentum: Smoothing factor for exponential moving average.
            Higher values = smoother, slower response (default: 0.99).
        min_pause_steps: Minimum number of steps a component stays paused
            once a threshold is breached, preventing rapid oscillation.
        warmup_steps: Minimum number of steps the model trains before any
            equilibrium is applied.
        verbose: Whether to log state transitions.
    """

    def __init__(
        self,
        upper_threshold: float = 2.0,
        lower_threshold: float = -0.5,
        diff_kld_rec_weight: float = 0.7,
        ema_momentum: float = 0.99,
        min_pause_steps: int = 50,
        warmup_pause_steps: int = 500,
        verbose: bool = True,
    ):
        super().__init__()
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.diff_kld_rec_weight = diff_kld_rec_weight
        self.ema_momentum = ema_momentum
        self.min_pause_steps = min_pause_steps
        self.warmup_pause_steps = warmup_pause_steps
        self.verbose = verbose

        # State
        self._ema: Optional[float] = None
        self._steps_paused: int = 0
        self._state: str = "both"  # "both" | "encoder_only" | "decoder_only" | "warmup"

    def on_train_begin(self, logs=None):
        self._ema = None
        self._steps_paused = 0
        self._state = "warmup"
        # Only collaborative step is trained
        self.model.train_encoder_enabled = False
        self.model.train_decoder_enabled = False

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}

        keys = ["kld_real", "kld_rec", "kld_fake"]
        exists = all(key in logs for key in keys)

        if not exists:
            return

        kld_real = logs.get("kld_real")
        kld_rec = logs.get("kld_rec")
        kld_fake = logs.get("kld_fake")

        diff_kld = (
            self.diff_kld_rec_weight * kld_rec
            + (1 - self.diff_kld_rec_weight) * kld_fake
            - kld_real
        )
        logs["diff_kld"] = diff_kld

        # Convert from tensor if needed
        diff_kld = float(diff_kld)

        # Update EMA
        if self._ema is None:
            self._ema = diff_kld
        else:
            self._ema = (
                self.ema_momentum * self._ema + (1 - self.ema_momentum) * diff_kld
            )

        prev_state = self._state

        if self._state != "both":
            self._steps_paused += 1

        # Transition logic with hysteresis
        if self._state == "both":
            if self._ema > self.upper_threshold:
                # Encoder is dominant — pause it, let decoder catch up
                self._state = "decoder_only"
                self._steps_paused = 0
                self.model.train_encoder_enabled = False
                self.model.train_decoder_enabled = True

            elif self._ema < self.lower_threshold:
                # Encoder is collapsing — pause decoder, let encoder recover
                self._state = "encoder_only"
                self._steps_paused = 0
                self.model.train_encoder_enabled = True
                self.model.train_decoder_enabled = False

        elif self._state == "decoder_only":
            # Resume encoder only after min pause AND EMA back in healthy zone
            if (
                self._steps_paused >= self.min_pause_steps
                and self._ema <= self.upper_threshold
            ):
                self._state = "both"
                self.model.train_encoder_enabled = True
                self.model.train_decoder_enabled = True

        elif self._state == "encoder_only":
            # Resume decoder only after min pause AND EMA back in healthy zone
            if (
                self._steps_paused >= self.min_pause_steps
                and self._ema >= self.lower_threshold
            ):
                self._state = "both"
                self.model.train_encoder_enabled = True
                self.model.train_decoder_enabled = True
        elif self._state == "warmup":
            if self._steps_paused >= self.warmup_pause_steps:
                self._state = "both"

        if self.verbose and self._state != prev_state:
            print(
                f"\n[Equilibrium] State: {prev_state} → {self._state} "
                f"(EMA diff_kld={self._ema:.4f}, steps_paused={self._steps_paused})"
            )

    def get_config(self):
        return dict(
            upper_threshold=self.upper_threshold,
            lower_threshold=self.lower_threshold,
            ema_momentum=self.ema_momentum,
            min_pause_steps=self.min_pause_steps,
            warmup_pause_steps=self.warmup_pause_steps,
            verbose=self.verbose,
        )


@keras.saving.register_keras_serializable(
    package="agx_core.models.reversed_autoencoder"
)
class BackboneThawCallback(keras.callbacks.Callback):
    """Progressively thaws the pretrained encoder backbone when training plateaus.

    Monitors val_loss_rec plateau to decide when to unfreeze, then applies
    a discriminative learning rate (backbone LR << head LR) to prevent
    catastrophic forgetting of pretrained features.

    Args:
        monitor: Metric to watch for plateau (default: "val_loss_rec").
        patience: Epochs of no improvement before thawing.
        min_delta: Minimum change to qualify as improvement.
        backbone_lr_factor: Factor to scale backbone LR relative to head LR.
            e.g., 0.01 means backbone trains at 1/100th the head's rate.
        verbose: Whether to log when thawing occurs.
    """

    def __init__(
        self,
        monitor: str = "val_loss_rec",
        patience: int = 15,
        min_delta: float = 1e-4,
        backbone_lr_factor: float = 0.01,
        verbose: bool = True,
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.backbone_lr_factor = backbone_lr_factor
        self.verbose = verbose

        self._best: Optional[float] = None
        self._wait: int = 0
        self._thawed: bool = False

    def on_train_begin(self, logs=None):
        self._best = None
        self._wait = 0
        self._thawed = False

    def on_epoch_end(self, epoch, logs=None):
        if self._thawed or logs is None:
            return

        current = logs.get(self.monitor)
        if current is None:
            return

        current = float(current)

        if self._best is None or current < self._best - self.min_delta:
            self._best = current
            self._wait = 0
        else:
            self._wait += 1

        if self._wait >= self.patience:
            self._thaw_backbone()
            self._thawed = True

    def _thaw_backbone(self):
        from .encoders.mobilenet_v3 import MobileNetV3SmallEncoder

        encoder = self.model.encoder

        if not isinstance(encoder, MobileNetV3SmallEncoder):
            if self.verbose:
                print(
                    f"\n[Thaw] Encoder is {type(encoder).__name__}, "
                    "not a backbone encoder — skipping."
                )
            return

        encoder.freeze_backbone = False
        encoder.training_enabled(True)

        if self.verbose:
            print(f"\n[Thaw] Backbone unfrozen (LR factor: {self.backbone_lr_factor}x)")
            print(
                f"[Thaw] Triggered after {self._wait} epochs without "
                f"improvement on {self.monitor} (best={self._best:.6f})"
            )

    @property
    def thawed(self) -> bool:
        return self._thawed

    def get_config(self):
        return dict(
            monitor=self.monitor,
            patience=self.patience,
            min_delta=self.min_delta,
            backbone_lr_factor=self.backbone_lr_factor,
            verbose=self.verbose,
        )


@keras.saving.register_keras_serializable(
    package="agx_core.models.reversed_autoencoder"
)
class ProgressiveGrowingCallback(keras.callbacks.Callback):
    """Drives progressive growing of the decoder during training.

    The decoder starts at low resolution (stage 0) and progressively
    activates higher-resolution stages. Each new stage is faded in
    over ``fade_in_steps`` training steps by linearly ramping alpha
    from 0→1. Once alpha reaches 1.0, the stage is stabilized for
    ``stabilize_steps`` steps before the next grow() call.

    Training images must be downscaled to match the decoder's current
    output resolution. The callback exposes ``current_output_size``
    so the data pipeline (or a wrapping tf.data.Dataset.map) can
    resize targets on the fly.

    Timeline for a 5-stage decoder:

        Steps:  |-- fade 0 --|-- stab 0 --|-- fade 1 --|-- stab 1 --| ...
        Stage:       0 (α→1)      0 (α=1)      1 (α→1)      1 (α=1)
        Res:        14×14         14×14         28×28         28×28

    The final stage completes its stabilization phase and the callback
    then sets the decoder to full-network mode (is_fully_grown=True),
    at which point the standard head_block + head_conv path takes over
    from the per-stage to_rgb heads.

    Integration with ReversedAutoencoderBase:
        The VAE's loss functions (pixel MSE, KLD, embedding loss) all
        operate on whatever resolution the decoder currently outputs.
        The caller is responsible for downscaling the real images and
        conditioning tensors to match ``decoder.current_output_size()``
        before passing them to the model's train_step.

    Args:
        fade_in_steps: Training steps to linearly ramp alpha 0→1 for
            each new stage.
        stabilize_steps: Training steps to hold alpha=1.0 after fade-in
            completes, letting the new stage's weights settle before
            the next grow.
        verbose: Whether to log grow / fade-in events.
    """

    def __init__(
        self,
        fade_in_steps: int = 5000,
        stabilize_steps: int = 5000,
        fade_in_expelbo_factor: float = 0.25,
        verbose: bool = True,
    ):
        super().__init__()
        self.fade_in_steps = fade_in_steps
        self.stabilize_steps = stabilize_steps
        self.fade_in_expelbo_factor = fade_in_expelbo_factor
        self.verbose = verbose

        # Internal counters
        self._phase: str = "stabilize"
        self._phase_step: int = 0
        self._total_steps: int = 0
        self._base_dec_expelbo_temp: float | None = None

    @property
    def decoder(self) -> "MobileNetV3SmallProgressiveDecoder":
        from .decoders.mobilenet_v3 import MobileNetV3SmallProgressiveDecoder

        model = self.model
        if isinstance(model, MobileNetV3SmallProgressiveDecoder):
            return model
        if hasattr(model, "decoder") and isinstance(
            model.decoder, MobileNetV3SmallProgressiveDecoder
        ):
            return model.decoder
        raise TypeError(
            f"ProgressiveGrowingCallback requires a MobileNetV3SmallProgressiveDecoder, "
            f"got {type(getattr(model, 'decoder', model)).__name__}"
        )

    @property
    def encoder(self) -> "MobileNetV3SmallProgressiveEncoder":
        from .encoders.mobilenet_v3 import MobileNetV3SmallProgressiveEncoder

        model = self.model
        if isinstance(model, MobileNetV3SmallProgressiveEncoder):
            return model
        if hasattr(model, "encoder") and isinstance(
            model.encoder, MobileNetV3SmallProgressiveEncoder
        ):
            return model.encoder
        raise TypeError(
            f"ProgressiveGrowingCallback requires a MobileNetV3SmallProgressiveEncoder, "
            f"got {type(getattr(model, 'encoder', model)).__name__}"
        )

    def on_train_begin(self, logs=None):
        dec = self.decoder
        enc = self.encoder
        # Accessing .decoder / .encoder already asserts the correct types;
        # touch both here to surface misconfiguration early.
        _ = dec, enc

        self._phase = "stabilize"
        self._phase_step = 0
        self._total_steps = 0
        self._base_dec_expelbo_temp = self.model.dec_expelbo_temp

        if self.verbose:
            shape = dec.current_output_size()
            print(
                f"\n[ProGrow] Starting progressive training at stage "
                f"{dec.current_stage} — output {shape}"
            )

    def on_train_batch_end(self, batch, logs=None):
        dec = self.decoder
        enc = self.encoder
        if dec.is_fully_grown:
            return

        self._total_steps += 1

        # Pause progressive schedule while equilibrium has encoder side frozen
        if not (self.model.train_encoder_enabled and self.model.train_decoder_enabled):
            return

        self._phase_step += 1

        if self._phase == "stabilize":
            if self._phase_step >= self.stabilize_steps:
                # Stabilization complete — grow to next stage
                dec.grow()
                enc.grow()

                if dec.is_fully_grown:
                    # Restore full expelbo temperature
                    self.model.dec_expelbo_temp = self._base_dec_expelbo_temp
                    if self.verbose:
                        print(
                            f"\n[ProGrow] Step {self._total_steps}: "
                            f"Fully grown — switching to full network path"
                        )
                    return

                # Enter fade-in: dampen expelbo to prevent amplified destabilization
                self._phase = "fade_in"
                self._phase_step = 0
                self.model.dec_expelbo_temp = (
                    self._base_dec_expelbo_temp * self.fade_in_expelbo_factor
                )

                if self.verbose:
                    shape = dec.current_output_size()
                    print(
                        f"\n[ProGrow] Step {self._total_steps}: "
                        f"Growing to stage {dec.current_stage} — "
                        f"fade-in over {self.fade_in_steps} steps — "
                        f"output {shape} — "
                        f"dec_expelbo_temp dampened to {self.model.dec_expelbo_temp:.3f}"
                    )

        elif self._phase == "fade_in":
            dec.alpha = self._phase_step / self.fade_in_steps
            enc.alpha = self._phase_step / self.fade_in_steps

            # Linearly ramp expelbo temperature back up during fade-in
            progress = self._phase_step / self.fade_in_steps
            self.model.dec_expelbo_temp = self._base_dec_expelbo_temp * (
                self.fade_in_expelbo_factor
                + (1.0 - self.fade_in_expelbo_factor) * progress
            )

            if self._phase_step >= self.fade_in_steps:
                dec.alpha = 1.0
                enc.alpha = 1.0
                self._phase = "stabilize"
                self._phase_step = 0

                # Restore full expelbo temperature
                self.model.dec_expelbo_temp = self._base_dec_expelbo_temp

                if self.verbose:
                    print(
                        f"\n[ProGrow] Step {self._total_steps}: "
                        f"Stage {dec.current_stage} fade-in complete — "
                        f"stabilizing for {self.stabilize_steps} steps — "
                        f"dec_expelbo_temp restored to {self.model.dec_expelbo_temp:.3f}"
                    )

    def get_config(self):
        return dict(
            fade_in_steps=self.fade_in_steps,
            stabilize_steps=self.stabilize_steps,
            fade_in_expelbo_factor=self.fade_in_expelbo_factor,
            verbose=self.verbose,
        )
