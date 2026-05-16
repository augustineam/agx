from __future__ import annotations

import torch
import gc

from pathlib import Path
from typing import Sequence

from keras import callbacks


class GarbageCollectionCallback(callbacks.Callback):
    """Force cyclic GC and optionally clear CUDA cache at regular intervals.

    PyTorch's CUDA allocator caches freed blocks and Python's cyclic GC
    doesn't run frequently enough for GPU-heavy multi-backward training
    steps. This callback forces cleanup at a configurable cadence.

    Args:
        gc_every_n_steps: Run ``gc.collect()`` every N training steps.
            Set to 1 for maximum cleanliness (adds ~1-3ms per step).
        empty_cache_every_n_steps: Run ``torch.cuda.empty_cache()`` every
            N training steps. More expensive than gc alone — releases
            cached blocks back to CUDA, helping fragmentation but adding
            a small sync cost. Set to 0 to disable.
        verbose: Print allocated/reserved MB when cache is emptied.
    """

    def __init__(
        self,
        gc_every_n_steps: int = 1,
        empty_cache_every_n_steps: int = 0,
        verbose: bool = False,
    ):
        super().__init__()
        self.gc_every_n_steps = gc_every_n_steps
        self.empty_cache_every_n_steps = empty_cache_every_n_steps
        self.verbose = verbose
        self._step: int = 0

    def on_train_batch_end(self, batch, logs=None):
        self._step += 1

        if self.gc_every_n_steps > 0 and self._step % self.gc_every_n_steps == 0:
            gc.collect()

        if (
            self.empty_cache_every_n_steps > 0
            and self._step % self.empty_cache_every_n_steps == 0
        ):
            torch.cuda.empty_cache()
            if self.verbose:
                alloc = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                print(
                    f"\n[GC] Step {self._step}: "
                    f"allocated={alloc:.1f}MB, reserved={reserved:.1f}MB"
                )


class MemorySnapshotCallback(callbacks.Callback):
    """Capture CUDA memory snapshots over configurable step windows.

    Unlike point-in-time snapshots, this records allocation history over
    a window of N steps so you can see exactly which allocations accumulate
    and which are freed within that window.

    Usage:
        To capture 10-step windows starting at steps 100 and 500:

            MemorySnapshotCallback(
                snapshot_at_steps=[100, 500],
                capture_duration=10,
                output_dir="./mem_snapshots",
            )

        This will:
        - Start recording at step 100, stop at step 110, dump snapshot.
        - Start recording at step 500, stop at step 510, dump snapshot.

    Args:
        snapshot_at_steps: Global training steps at which to BEGIN
            capturing memory history.
        capture_duration: Number of steps to record before stopping
            and dumping the snapshot.
        output_dir: Directory to write ``.pickle`` snapshot files.
            Created if it doesn't exist.
        max_entries: Maximum allocation entries to record per window.
            Higher = more detail but more CPU overhead during recording.
        print_summary: Print ``torch.cuda.memory_summary`` at each
            snapshot dump.
    """

    def __init__(
        self,
        snapshot_at_steps: Sequence[int] = (100, 500),
        capture_duration: int = 10,
        output_dir: str = "./mem_snapshots",
        max_entries: int = 1_000_000,
        print_summary: bool = True,
    ):
        super().__init__()
        self.snapshot_at_steps = sorted(snapshot_at_steps)
        self.capture_duration = capture_duration
        self.output_dir = Path(output_dir)
        self.max_entries = max_entries
        self.print_summary = print_summary

        self._step: int = 0
        self._recording: bool = False
        self._recording_start_step: int = 0
        self._pending_starts: list[int] = []

    def on_train_begin(self, logs=None):
        self._step = 0
        self._recording = False
        self._pending_starts = list(self.snapshot_at_steps)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_train_batch_end(self, batch, logs=None):
        self._step += 1

        # Check if we should start recording
        if (
            not self._recording
            and self._pending_starts
            and self._step >= self._pending_starts[0]
        ):
            self._start_recording()

        # Check if capture window is complete
        elif self._recording:
            elapsed = self._step - self._recording_start_step
            if elapsed >= self.capture_duration:
                self._stop_and_dump()

    def on_train_end(self, logs=None):
        if self._recording:
            self._stop_and_dump()

    def _start_recording(self):
        """Begin recording CUDA allocation history."""
        self._pending_starts.pop(0)
        self._recording = True
        self._recording_start_step = self._step

        # Clean state before recording
        gc.collect()
        torch.cuda.empty_cache()

        torch.cuda.memory._record_memory_history(
            max_entries=self.max_entries,
            stacks="python",
        )

        print(
            f"\n[MemSnapshot] Recording started at step {self._step} "
            f"(will capture {self.capture_duration} steps)"
        )

    def _stop_and_dump(self):
        """Stop recording and write snapshot to disk."""
        self._recording = False

        start = self._recording_start_step
        end = self._step
        filename = self.output_dir / f"snapshot_steps_{start}_to_{end}.pickle"

        torch.cuda.memory._dump_snapshot(str(filename))
        torch.cuda.memory._record_memory_history(enabled=None)

        print(f"[MemSnapshot] Dumped steps {start}-{end} → {filename}")

        if self.print_summary:
            gc.collect()
            torch.cuda.empty_cache()
            print(torch.cuda.memory_summary(abbreviated=True))


class MemoryLeakDiagnosticCallback(callbacks.Callback):
    """Diagnose GPU memory leaks by comparing allocated memory across steps
    and detecting unreachable CUDA tensors (ghost blocks).

    Prints per-step:
      - allocated / reserved delta from previous step
      - number of live CUDA tensors found via gc
      - total size of live CUDA tensors vs reported allocation (gap = ghosts)
      - after gc.collect() + empty_cache: residual that won't go away

    Usage:
        MemoryLeakDiagnosticCallback(
            start_step=5,        # skip warmup
            every_n_steps=10,    # don't spam every step
            full_gc_every=50,    # expensive full diagnostic less often
        )
    """

    def __init__(
        self,
        start_step: int = 5,
        every_n_steps: int = 10,
        full_gc_every: int = 50,
        verbose_tensors: bool = False,
    ):
        super().__init__()
        self.start_step = start_step
        self.every_n_steps = every_n_steps
        self.full_gc_every = full_gc_every
        self.verbose_tensors = verbose_tensors
        self._step = 0
        self._prev_alloc = 0
        self._prev_reserved = 0

    def on_train_batch_end(self, batch, logs=None):
        self._step += 1

        if self._step < self.start_step:
            return

        if self._step % self.every_n_steps != 0:
            return

        torch.cuda.synchronize()

        alloc = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        delta_alloc = alloc - self._prev_alloc
        delta_reserved = reserved - self._prev_reserved

        # Count live CUDA tensors reachable from Python
        live_tensor_bytes, live_tensor_count = self._count_live_cuda_tensors()

        ghost_bytes = alloc - live_tensor_bytes

        print(
            f"\n[MemDiag] Step {self._step}: "
            f"alloc={alloc / 1024**2:.1f}MB (Δ{delta_alloc / 1024**2:+.1f}MB)  "
            f"reserved={reserved / 1024**2:.1f}MB (Δ{delta_reserved / 1024**2:+.1f}MB)  "
            f"live_tensors={live_tensor_count} ({live_tensor_bytes / 1024**2:.1f}MB)  "
            f"ghosts={ghost_bytes / 1024**2:.1f}MB"
        )

        # Full diagnostic: gc + empty_cache to find true leaks
        if self._step % self.full_gc_every == 0:
            self._full_diagnostic()

        self._prev_alloc = alloc
        self._prev_reserved = reserved

    def _count_live_cuda_tensors(self):
        """Walk all Python objects to find CUDA tensors and sum their storage."""
        seen_data_ptrs = set()
        total_bytes = 0
        count = 0

        for obj in gc.get_objects():
            try:
                if isinstance(obj, torch.Tensor) and obj.is_cuda:
                    # Use data_ptr to avoid double-counting shared storage
                    ptr = obj.data_ptr()
                    if ptr not in seen_data_ptrs and ptr != 0:
                        seen_data_ptrs.add(ptr)
                        total_bytes += obj.element_size() * obj.nelement()
                        count += 1

                        if self.verbose_tensors and obj.nelement() > 1_000_000:
                            print(
                                f"    [large tensor] shape={tuple(obj.shape)} "
                                f"dtype={obj.dtype} "
                                f"size={obj.element_size() * obj.nelement() / 1024**2:.1f}MB "
                                f"grad_fn={obj.grad_fn is not None} "
                                f"requires_grad={obj.requires_grad}"
                            )
            except (ReferenceError, RuntimeError):
                pass

        return total_bytes, count

    def _full_diagnostic(self):
        """Run GC + empty_cache and report what remains."""
        # Before GC
        pre_gc_alloc = torch.cuda.memory_allocated()

        gc.collect()
        torch.cuda.synchronize()

        post_gc_alloc = torch.cuda.memory_allocated()
        gc_freed = pre_gc_alloc - post_gc_alloc

        torch.cuda.empty_cache()
        post_cache_reserved = torch.cuda.memory_reserved()

        # Recount after GC
        live_bytes, live_count = self._count_live_cuda_tensors()

        print(
            f"  [FULL GC] gc.collect() freed {gc_freed / 1024**2:.1f}MB  "
            f"post_gc_alloc={post_gc_alloc / 1024**2:.1f}MB  "
            f"post_empty_cache_reserved={post_cache_reserved / 1024**2:.1f}MB  "
            f"live={live_count} tensors ({live_bytes / 1024**2:.1f}MB)  "
            f"true_ghosts={(post_gc_alloc - live_bytes) / 1024**2:.1f}MB"
        )


__all__ = [
    "GarbageCollectionCallback",
    "MemorySnapshotCallback",
    "MemoryLeakDiagnosticCallback",
]
