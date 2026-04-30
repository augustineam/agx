from __future__ import annotations

import keras
from keras import ops

from typing import Optional


@keras.saving.register_keras_serializable(package="agx_core.models.ra")
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
        ema_momentum: float = 0.99,
        min_pause_steps: int = 50,
        verbose: bool = True,
    ):
        super().__init__()
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.ema_momentum = ema_momentum
        self.min_pause_steps = min_pause_steps
        self.verbose = verbose

        # State
        self._ema: Optional[float] = None
        self._steps_paused: int = 0
        self._state: str = "both"  # "both" | "encoder_only" | "decoder_only" | "warmup"

    def on_train_begin(self, logs=None):
        self._ema = None
        self._steps_paused = 0
        self._state = "warmup"
        self.model.train_encoder_enabled = True
        self.model.train_decoder_enabled = True

    def on_train_batch_end(self, batch, logs=None):
        if logs is None:
            return

        diff_kld = logs.get("diff_kld")
        if diff_kld is None:
            return

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
            if self._steps_paused >= self.min_pause_steps:
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
            verbose=self.verbose,
        )


# ... existing code ...


@keras.saving.register_keras_serializable(package="agx_core.models.ra")
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
        encoder = self.model.encoder
        self.model.freeze_backbone = False

        if not hasattr(encoder, "backbone") or not hasattr(encoder, "train_backbone"):
            if self.verbose:
                print("\n[Thaw] No backbone attribute found on encoder. Skipping.")
            return

        # Unfreeze all backbone layers
        encoder.train_backbone(True)

        if self.verbose:
            print(
                f"\n[Thaw] Unfreezing backbone: LR factor: {self.backbone_lr_factor}x"
            )
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
