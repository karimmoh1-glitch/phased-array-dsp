"""From-scratch discrete Fourier transforms (no numpy, no scipy).

Power-of-two lengths use an iterative, in-place, radix-2 decimation-in-time
Cooley-Tukey FFT. Every other length goes through Bluestein's chirp-z
algorithm, which rewrites the DFT as a circular convolution that is itself
evaluated with the radix-2 kernel, so every length costs O(N log N).

Conventions match the usual engineering definition:

    X[k] = sum_{n=0}^{N-1} x[n] * exp(-2j*pi*k*n/N)          (forward, unscaled)
    x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(+2j*pi*k*n/N)  (inverse)
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence
from functools import lru_cache

__all__ = [
    "fft",
    "fftfreq",
    "ifft",
    "irfft",
    "is_power_of_two",
    "next_power_of_two",
    "rfft",
    "rfftfreq",
]


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    """Smallest power of two >= n (1 for n <= 1)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


@lru_cache(maxsize=128)
def _bit_reversal(n: int) -> tuple[int, ...]:
    """Bit-reversed index permutation for a power-of-two length n."""
    bits = n.bit_length() - 1
    rev = [0] * n
    for i in range(1, n):
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bits - 1))
    return tuple(rev)


@lru_cache(maxsize=128)
def _twiddles(n: int, inverse: bool) -> tuple[complex, ...]:
    """W_n^k = exp(-+2j*pi*k/n) for k < n/2.

    Each factor is computed directly rather than by repeated multiplication,
    so twiddle error stays at one ulp instead of growing with k.
    """
    sign = 1.0 if inverse else -1.0
    return tuple(cmath.exp(complex(0.0, sign * 2.0 * math.pi * k / n)) for k in range(n // 2))


def _radix2(x: Sequence[complex], inverse: bool) -> list[complex]:
    """Unscaled iterative radix-2 DIT FFT. len(x) must be a power of two."""
    n = len(x)
    rev = _bit_reversal(n)
    a = [complex(x[r]) for r in rev]
    tw = _twiddles(n, inverse)
    half = 1
    while half < n:
        size = half << 1
        stride = n // size
        # Loop over twiddles on the outside so each W is fetched once per stage.
        for k in range(half):
            w = tw[k * stride]
            for i in range(k, n, size):
                j = i + half
                t = w * a[j]
                u = a[i]
                a[i] = u + t
                a[j] = u - t
        half = size
    return a


@lru_cache(maxsize=32)
def _bluestein_plan(n: int, inverse: bool) -> tuple[tuple[complex, ...], tuple[complex, ...], int]:
    """Chirp w[k] = exp(-+j*pi*k^2/n) and the FFT of its conjugate kernel."""
    sign = 1.0 if inverse else -1.0
    two_n = 2 * n
    # k^2 mod 2n keeps the phase argument small, so exp() stays exact for large k.
    chirp = tuple(cmath.exp(complex(0.0, sign * math.pi * ((k * k) % two_n) / n)) for k in range(n))
    m = next_power_of_two(2 * n - 1)
    kernel = [0j] * m
    kernel[0] = chirp[0].conjugate()
    for k in range(1, n):
        c = chirp[k].conjugate()
        kernel[k] = c
        kernel[m - k] = c
    return chirp, tuple(_radix2(kernel, False)), m


def _bluestein(x: Sequence[complex], inverse: bool) -> list[complex]:
    """Unscaled DFT of arbitrary length via nk = (n^2 + k^2 - (k-n)^2) / 2."""
    n = len(x)
    chirp, kernel_hat, m = _bluestein_plan(n, inverse)
    a = [0j] * m
    for k in range(n):
        a[k] = complex(x[k]) * chirp[k]
    a_hat = _radix2(a, False)
    conv = _radix2([p * q for p, q in zip(a_hat, kernel_hat)], True)
    scale = 1.0 / m
    return [chirp[k] * conv[k] * scale for k in range(n)]


def _transform(x: Sequence[complex], inverse: bool) -> list[complex]:
    if len(x) == 0:
        raise ValueError("cannot transform an empty sequence")
    if is_power_of_two(len(x)):
        return _radix2(x, inverse)
    return _bluestein(x, inverse)


def fft(x: Sequence[complex]) -> list[complex]:
    """Forward DFT of a real or complex sequence of any non-zero length."""
    return _transform(x, False)


def ifft(spectrum: Sequence[complex]) -> list[complex]:
    """Inverse DFT, scaled by 1/N so that ifft(fft(x)) == x."""
    n = len(spectrum)
    scale = 1.0 / n if n else 0.0
    return [v * scale for v in _transform(spectrum, True)]


def rfft(x: Sequence[float]) -> list[complex]:
    """Non-negative-frequency half (N//2 + 1 bins) of the DFT of a real signal.

    For even N the real signal is packed into an N/2-point complex FFT
    (z[k] = x[2k] + j*x[2k+1]) and the two interleaved half-spectra are
    separated afterwards, roughly halving the work of a full complex FFT.
    """
    n = len(x)
    if n == 0:
        raise ValueError("cannot transform an empty sequence")
    if n % 2:
        return fft(x)[: n // 2 + 1]
    m = n // 2
    z = fft([complex(x[2 * k], x[2 * k + 1]) for k in range(m)])
    tw = _twiddles(n, False)
    out = [0j] * (m + 1)
    for k in range(m + 1):
        zk = z[k % m]
        zc = z[(m - k) % m].conjugate()
        even = 0.5 * (zk + zc)
        odd = -0.5j * (zk - zc)
        w = tw[k] if k < m else complex(-1.0, 0.0)
        out[k] = even + w * odd
    return out


def irfft(spectrum: Sequence[complex], n: int | None = None) -> list[float]:
    """Inverse of rfft. Returns n real samples (default 2*(len(spectrum)-1)).

    If n is larger than the original length, the missing high-frequency bins
    are treated as zero, i.e. this performs band-limited (sinc) interpolation.
    """
    if n is None:
        n = 2 * (len(spectrum) - 1)
    if n <= 0:
        raise ValueError("output length must be positive")
    full = [0j] * n
    for k in range(min(len(spectrum), n // 2 + 1)):
        full[k] = complex(spectrum[k])
    for k in range(1, (n + 1) // 2):
        full[n - k] = full[k].conjugate()
    return [v.real for v in ifft(full)]


def fftfreq(n: int, sample_rate: float = 1.0) -> list[float]:
    """Centre frequency of each fft() bin, negative frequencies in the upper half."""
    return [(k if k < (n + 1) // 2 else k - n) * sample_rate / n for k in range(n)]


def rfftfreq(n: int, sample_rate: float = 1.0) -> list[float]:
    """Centre frequency of each rfft() bin."""
    return [k * sample_rate / n for k in range(n // 2 + 1)]
