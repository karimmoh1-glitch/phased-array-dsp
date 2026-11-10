"""FFT-based cross-correlation and GCC-PHAT time-delay estimation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .fft import irfft, next_power_of_two, rfft
from .spectral import hann

__all__ = ["DelayEstimate", "cross_correlate", "gcc_phat"]


@dataclass(frozen=True)
class DelayEstimate:
    """Delay of one signal relative to a reference.

    A positive delay means `signal` arrives *later* than `reference`,
    i.e. signal(t) ~= reference(t - delay).
    """

    delay_samples: float
    delay_s: float
    peak: float  # height of the correlation peak (~1.0 for a perfect PHAT match)


def _zero_pad(x: Sequence[float], n: int) -> list[float]:
    return list(x) + [0.0] * (n - len(x))


def cross_correlate(reference: Sequence[float], signal: Sequence[float]) -> tuple[list[int], list[float]]:
    """Full linear cross-correlation r[l] = sum_n reference[n] * signal[n + l].

    Computed as IFFT(conj(R) * S) with enough zero padding (>= len_a + len_b - 1)
    that the circular correlation equals the linear one. Returns (lags, values)
    for lags -(len(reference)-1) .. len(signal)-1.
    """
    na, nb = len(reference), len(signal)
    if na == 0 or nb == 0:
        raise ValueError("cannot correlate an empty sequence")
    nfft = next_power_of_two(na + nb - 1)
    ref_hat = rfft(_zero_pad(reference, nfft))
    sig_hat = rfft(_zero_pad(signal, nfft))
    circ = irfft([r.conjugate() * s for r, s in zip(ref_hat, sig_hat)], nfft)
    lags = list(range(-(na - 1), nb))
    return lags, [circ[lag % nfft] for lag in lags]


def gcc_phat(
    reference: Sequence[float],
    signal: Sequence[float],
    sample_rate: float,
    *,
    max_delay_s: float | None = None,
    interp: int = 16,
    window: bool = True,
    epsilon: float = 1e-12,
) -> DelayEstimate:
    """Generalised cross-correlation with phase transform (GCC-PHAT).

        G[k]  = conj(X_ref[k]) * X_sig[k]
        R(tau) = IFFT( G[k] / |G[k]| )

    Whitening by |G| throws away magnitude and keeps only phase, so a pure
    delay D becomes an impulse at tau = D regardless of the source spectrum.
    Sub-sample resolution comes from evaluating the IFFT on an `interp`-times
    finer grid (zero-padding the spectrum) followed by a parabolic fit
    through the three samples around the peak.

    With window=True both inputs are Hann-tapered first; this suppresses the
    edge discontinuity of the finite record, which PHAT would otherwise
    amplify in low-energy bins.

    Note: a single pure tone has a periodic correlation, so its delay is only
    known modulo one period. Use the narrowband estimator for tones.
    """
    if len(reference) == 0 or len(signal) == 0:
        raise ValueError("cannot correlate an empty sequence")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if interp < 1:
        raise ValueError("interp must be >= 1")

    if window:
        reference = [v * w for v, w in zip(reference, hann(len(reference)))]
        signal = [v * w for v, w in zip(signal, hann(len(signal)))]
    nfft = next_power_of_two(len(reference) + len(signal))
    ref_hat = rfft(_zero_pad(reference, nfft))
    sig_hat = rfft(_zero_pad(signal, nfft))
    whitened: list[complex] = []
    for r, s in zip(ref_hat, sig_hat):
        g = r.conjugate() * s
        mag = abs(g)
        whitened.append(g / mag if mag > epsilon else 0j)

    n_fine = nfft * interp
    # irfft divides by n_fine; multiplying by interp restores a unit-height peak.
    cc = [v * interp for v in irfft(whitened, n_fine)]

    max_shift = n_fine // 2
    if max_delay_s is not None:
        max_shift = min(max_shift, int(math.ceil(abs(max_delay_s) * sample_rate * interp)) + 1)
    max_shift = max(max_shift, 1)
    # Reorder so index i corresponds to lag (i - max_shift) on the fine grid.
    lags = cc[n_fine - max_shift :] + cc[: max_shift + 1]

    best = max(range(len(lags)), key=lags.__getitem__)
    offset = 0.0
    if 0 < best < len(lags) - 1:
        y0, y1, y2 = lags[best - 1], lags[best], lags[best + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom != 0.0:
            offset = 0.5 * (y0 - y2) / denom
    delay_samples = (best - max_shift + offset) / interp
    return DelayEstimate(delay_samples, delay_samples / sample_rate, lags[best])
