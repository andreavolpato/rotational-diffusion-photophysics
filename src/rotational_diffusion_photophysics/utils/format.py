"""Small formatting helpers."""


def _human(n: float) -> str:
    """Format a count compactly, e.g. 24932887 -> '24.9M', 1530 -> '1.5k'."""
    for threshold, suffix in ((1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'k')):
        if abs(n) >= threshold:
            return f'{n / threshold:.1f}{suffix}'
    return f'{n:.0f}'
