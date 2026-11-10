"""Windowing, framing and spectral peak estimation built on the custom FFT."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .fft import next_power_of_two, rfft

__all__ = ["dominant_frequency", "frame_starts", "hann", "power_spectrum"]


def hann(n: int) -> list[float]:
    """Periodic Hann window, w[k] = 0.5 - 0.5*cos(2*pi*k/n).

    The periodic form (rather than the symmetric one) is the right choice for
    spectral analysis: its DFT has exactly three non-zero bins and frames
    overlapped by 50% sum to a constant.
    """
    if n <= 0:
        raise ValueError("window length must be positive")
    return [0.5 - 0.5 * math.cos(2.0 * math.pi * k / n) for k in range(n)]


def frame_starts(length: int, frame_size: int, hop: int) -> list[int]:
    """Start indices of every full frame of frame_size that fits in length."""
    if frame_size <= 0 or hop <= 0:
        raise ValueError("frame_size and hop must be positive")
    if length < frame_size:
        return []
    return list(range(0, length - frame_size + 1, hop))


def power_spectrum(x: Sequence[float], *, window: bool = True) -> list[float]:
    """|X[k]|^2 for the non-negative-frequency bins of x."""
    if window:
        w = hann(len(x))
        x = [a * b for a, b in zip(x, w)]
    return [abs(v) ** 2 for v in rfft(x)]


def dominant_frequency(x: Sequence[float], sample_rate: float, *, pad_factor: int = 4) -> float:
    """Frequency of the strongest spectral peak, in Hz.

    The signal is Hann-windowed and zero-padded to pad_factor * next_pow2(N),
    then the peak is refined with Gaussian (log-parabolic) interpolation:

        delta = 0.5 * (ln a - ln c) / (ln a - 2 ln b + ln c)

    where a, b, c are the magnitudes of the bins left of, at and right of the
    maximum. For a Hann main lobe this is accurate to a few thousandths of a bin.
    """
    if len(x) < 4:
        raise ValueError("need at least 4 samples to locate a spectral peak")
    w = hann(len(x))
    nfft = next_power_of_two(len(x)) * max(1, pad_factor)
    padded = [a * b for a, b in zip(x, w)] + [0.0] * (nfft - len(x))
    mags = [abs(v) for v in rfft(padded)]
    # Skip DC and Nyquist so the interpolation always has two neighbours.
    k = max(range(1, len(mags) - 1), key=mags.__getitem__)
    a, b, c = mags[k - 1], mags[k], mags[k + 1]
    delta = 0.0
    if a > 0.0 and b > 0.0 and c > 0.0:
        la, lb, lc = math.log(a), math.log(b), math.log(c)
        denom = la - 2.0 * lb + lc
        if denom != 0.0:
            delta = 0.5 * (la - lc) / denom
    return (k + delta) * sample_rate / nfft
