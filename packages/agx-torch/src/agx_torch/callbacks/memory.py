import torch
import gc

from keras import callbacks


class MemoryDiagnosticCallback(callbacks.Callback):
    """Take memory snapshots at specific epochs to find growth."""

    def __init__(self, snapshot_epochs=(10, 50, 200)):
        super().__init__()
        self.snapshot_epochs = snapshot_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch not in self.snapshot_epochs:
            return

        gc.collect()
        torch.cuda.empty_cache()

        # Print detailed stats
        print(f"\n{'='*60}")
        print(f"[MEM SNAPSHOT] Epoch {epoch}")
        print(torch.cuda.memory_summary(abbreviated=True))
        print(f"{'='*60}\n")

        # Full snapshot with stack traces
        torch.cuda.memory._dump_snapshot(f"mem_epoch_{epoch}.pickle")

    def on_train_begin(self, logs=None):
        # Start recording allocation history with stack traces
        torch.cuda.memory._record_memory_history(
            max_entries=100000,
            stacks="python",  # capture Python stacks (not C++)
        )

    def on_train_end(self, logs=None):
        torch.cuda.memory._record_memory_history(enabled=None)
