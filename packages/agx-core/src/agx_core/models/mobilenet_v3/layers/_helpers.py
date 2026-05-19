def _depth(v: int, divisor: int = 8, min_value: int | None = None) -> int:
    """Round channel count to the nearest multiple of *divisor*."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Ensure rounding doesn't reduce by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
