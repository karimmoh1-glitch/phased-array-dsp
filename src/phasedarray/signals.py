"""Plane-wave array simulation with exact fractional delays.

Sources are continuous-time functions s(t), so channel m is sampled as
s(n/fs - tau_m) directly. That gives mathematically exact sub-sample delays
with no interpolation filter error, which is what precision tests need.
"""

from __future__ import annotations

import cmath
import math
import random
from collections.abc import Callable, Sequence

from .aoa import LinearArray
from .fft import irfft

__all__ = ["Source", "multitone", "simulate_array", "simulate_broadband_array", "tone"]

Source = Callable[[float], float]


def multitone(
    frequencies_hz: Sequence[float],
    amplitudes: Sequence[float] | None = None,
    phases_rad: Sequence[float] | None = None,
) -> Source:
    """s(t) = sum_i A_i * sin(2*pi*f_i*t + phi_i)."""
    amps = list(amplitudes) if amplitudes is not None else [1.0] * len(frequencies_hz)
    phs = list(phases_rad) if phases_rad is not None else [0.0] * len(frequencies_hz)
    if not len(amps) == len(phs) == len(frequencies_hz):
        raise ValueError("frequencies, amplitudes and phases must have the same length")
    comps = [(2.0 * math.pi * f, a, p) for f, a, p in zip(frequencies_hz, amps, phs)]

    def source(t: float) -> float:
        return sum(a * math.sin(w * t + p) for w, a, p in comps)

    return source


def tone(frequency_hz: float, sample_rate: float, n_samples: int, *, amplitude: float = 1.0,
         phase_rad: float = 0.0, delay_s: float = 0.0) -> list[float]:
    """Sampled A*sin(2*pi*f*(t - delay) + phase)."""
    w = 2.0 * math.pi * frequency_hz
    return [amplitude * math.sin(w * (n / sample_rate - delay_s) + phase_rad) for n in range(n_samples)]


def simulate_array(
    array: LinearArray,
    angle_deg: float,
    source: Source,
    sample_rate: float,
    n_samples: int,
    *,
    snr_db: float | None = None,
    seed: int | None = None,
) -> list[list[float]]:
    """Channels x_m[n] = s(n/fs - tau_m) + w_m[n] for a far-field source at angle_deg.

    w_m is independent white Gaussian noise scaled so each channel has the
    requested signal-to-noise ratio: sigma^2 = P_signal / 10^(snr_db/10).
    """
    delays = array.arrival_delays(angle_deg)
    channels = [[source(n / sample_rate - tau) for n in range(n_samples)] for tau in delays]
    if snr_db is not None:
        _add_noise(channels, snr_db, random.Random(seed))
    return channels


def simulate_broadband_array(
    array: LinearArray,
    angle_deg: float,
    sample_rate: float,
    n_samples: int,
    *,
    low_hz: float = 100.0,
    high_hz: float | None = None,
    snr_db: float | None = None,
    seed: int | None = None,
) -> list[list[float]]:
    """Band-limited Gaussian noise source arriving as a plane wave at angle_deg.

    The source is drawn directly in the frequency domain: every rfft bin in
    [low_hz, high_hz] gets an independent complex Gaussian coefficient S[k].
    Channel m is then synthesised as

        x_m = IRFFT( S[k] * exp(-2j*pi*f_k*tau_m) )

    which is an exact fractional delay of the (N-periodic) band-limited
    source - no interpolation filter, no sparse spectrum. Dense spectral
    support matters for GCC-PHAT, which weights every bin equally.
    """
    if high_hz is None:
        high_hz = 0.45 * sample_rate
    if not 0.0 <= low_hz < high_hz < sample_rate / 2.0:
        raise ValueError("need 0 <= low_hz < high_hz < Nyquist")
    rng = random.Random(seed)
    spectrum = [0j] * (n_samples // 2 + 1)
    for k in range(1, len(spectrum)):
        f = k * sample_rate / n_samples
        if low_hz <= f <= high_hz:
            spectrum[k] = complex(rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0))
    channels: list[list[float]] = []
    for tau in array.arrival_delays(angle_deg):
        shifted = [
            s * cmath.exp(complex(0.0, -2.0 * math.pi * (k * sample_rate / n_samples) * tau))
            for k, s in enumerate(spectrum)
        ]
        channels.append(irfft(shifted, n_samples))
    rms = math.sqrt(sum(v * v for v in channels[0]) / n_samples)
    channels = [[v / rms for v in ch] for ch in channels]
    if snr_db is not None:
        _add_noise(channels, snr_db, rng)
    return channels


def _add_noise(channels: list[list[float]], snr_db: float, rng: random.Random) -> None:
    for ch in channels:
        power = sum(v * v for v in ch) / len(ch)
        sigma = math.sqrt(power / 10.0 ** (snr_db / 10.0))
        for i in range(len(ch)):
            ch[i] += rng.gauss(0.0, sigma)
