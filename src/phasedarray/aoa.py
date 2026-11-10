"""Array geometry and angle-of-arrival (AoA) estimation.

Geometry: a linear array with elements at positions x_m (metres) along one
axis. The angle theta is measured from broadside (perpendicular to the array),
positive towards +x. A far-field plane wave from theta reaches element m at

    tau_m = -x_m * sin(theta) / c

relative to the origin, so elements further along +x hear a +theta source
earlier. Both estimators reduce to a least-squares fit of

    D_ij = tau_j - tau_i = -(x_j - x_i) * sin(theta) / c

over measured inter-channel delays D_ij.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .correlation import gcc_phat
from .fft import next_power_of_two, rfft
from .spectral import dominant_frequency, frame_starts, hann

__all__ = [
    "SPEED_OF_LIGHT",
    "SPEED_OF_SOUND",
    "AoAEstimate",
    "LinearArray",
    "estimate_aoa",
    "estimate_aoa_gcc_phat",
    "estimate_aoa_narrowband",
]

SPEED_OF_SOUND = 343.0  # m/s, dry air at 20 C
SPEED_OF_LIGHT = 299_792_458.0  # m/s

Method = Literal["narrowband", "gcc_phat"]


@dataclass(frozen=True)
class LinearArray:
    """Sensor positions (metres, strictly increasing) and wave propagation speed."""

    positions_m: tuple[float, ...]
    propagation_speed: float = SPEED_OF_SOUND

    def __post_init__(self) -> None:
        if len(self.positions_m) < 2:
            raise ValueError("an array needs at least two elements")
        if any(b <= a for a, b in zip(self.positions_m, self.positions_m[1:])):
            raise ValueError("element positions must be strictly increasing")
        if self.propagation_speed <= 0:
            raise ValueError("propagation_speed must be positive")

    @classmethod
    def uniform(cls, n_elements: int, spacing_m: float, propagation_speed: float = SPEED_OF_SOUND) -> LinearArray:
        """Uniform linear array (ULA) with element 0 at the origin."""
        return cls(tuple(m * spacing_m for m in range(n_elements)), propagation_speed)

    @property
    def size(self) -> int:
        return len(self.positions_m)

    def arrival_delays(self, angle_deg: float) -> list[float]:
        """tau_m = -x_m sin(theta) / c for a far-field source at angle_deg."""
        s = math.sin(math.radians(angle_deg))
        return [-x * s / self.propagation_speed for x in self.positions_m]

    def max_unambiguous_frequency(self) -> float:
        """Highest frequency whose adjacent-element phase is unambiguous (d <= lambda/2)."""
        widest = max(b - a for a, b in zip(self.positions_m, self.positions_m[1:]))
        return self.propagation_speed / (2.0 * widest)


@dataclass(frozen=True)
class AoAEstimate:
    angle_deg: float
    sin_theta: float  # unclipped least-squares value; |sin_theta| > 1 flags inconsistent data
    delays_s: tuple[float, ...]  # measured delay of each channel relative to channel 0
    residual_rms_s: float  # RMS misfit of the measured delays to the plane-wave model
    method: Method
    frequency_hz: float | None = None


def _validate(channels: Sequence[Sequence[float]], array: LinearArray, sample_rate: float) -> int:
    if len(channels) != array.size:
        raise ValueError(f"got {len(channels)} channels for a {array.size}-element array")
    n = len(channels[0])
    if n == 0 or any(len(ch) != n for ch in channels):
        raise ValueError("all channels must be non-empty and the same length")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return n


def _fit_sin_theta(baselines: Sequence[float], delays: Sequence[float], speed: float) -> tuple[float, float]:
    """Least-squares sin(theta) for D = -baseline * sin(theta) / c, plus RMS residual.

    Minimising sum (D_i + b_i s / c)^2 over s gives s = -c * sum(D_i b_i) / sum(b_i^2).
    """
    num = sum(d * b for d, b in zip(delays, baselines))
    den = sum(b * b for b in baselines)
    s = -speed * num / den
    residual = math.sqrt(sum((d + b * s / speed) ** 2 for d, b in zip(delays, baselines)) / len(delays))
    return s, residual


def _angle(sin_theta: float) -> float:
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_theta))))


def _default_frame_size(n: int) -> int:
    """Largest power of two <= n, capped at 4096 so long buffers are averaged over frames."""
    size = next_power_of_two(n)
    if size > n:
        size //= 2
    return min(size, 4096)


def estimate_aoa_narrowband(
    channels: Sequence[Sequence[float]],
    sample_rate: float,
    array: LinearArray,
    frequency_hz: float | None = None,
    *,
    frame_size: int | None = None,
) -> AoAEstimate:
    """AoA of a narrowband (tone-like) source from cross-spectral phase.

    Each channel is cut into 50%-overlapped Hann frames and transformed with
    rfft. At the bin k nearest frequency_hz, the cross-spectrum between
    adjacent elements is averaged across frames (Welch's method):

        C_m = sum_frames conj(X_{m-1}[k]) * X_m[k]
        psi_m = arg(C_m) = 2*pi*f*(x_m - x_{m-1})*sin(theta)/c

    Adjacent phases are unambiguous when spacing <= lambda/2, so summing them
    unwraps the phase across the whole aperture before the least-squares fit.
    If frequency_hz is None the strongest spectral peak of channel 0 is used.
    """
    n = _validate(channels, array, sample_rate)
    size = frame_size if frame_size is not None else _default_frame_size(n)
    if size < 4 or size > n:
        raise ValueError(f"frame_size must be between 4 and the buffer length ({n})")
    if frequency_hz is None:
        frequency_hz = dominant_frequency(channels[0], sample_rate)
    if not 0.0 < frequency_hz < sample_rate / 2.0:
        raise ValueError("frequency must lie strictly between 0 and Nyquist")
    if frequency_hz > array.max_unambiguous_frequency():
        raise ValueError(
            f"{frequency_hz:.1f} Hz exceeds the spatial-aliasing limit "
            f"{array.max_unambiguous_frequency():.1f} Hz (element spacing > lambda/2)"
        )

    k = min(max(1, round(frequency_hz * size / sample_rate)), size // 2 - 1)
    window = hann(size)
    cross = [0j] * array.size
    for start in frame_starts(n, size, max(1, size // 2)):
        bins = [rfft([v * w for v, w in zip(ch[start : start + size], window)])[k] for ch in channels]
        for m in range(1, array.size):
            cross[m] += bins[m - 1].conjugate() * bins[m]

    x0 = array.positions_m[0]
    unwrapped = 0.0
    baselines: list[float] = []
    delays: list[float] = []
    for m in range(1, array.size):
        unwrapped += math.atan2(cross[m].imag, cross[m].real)
        baselines.append(array.positions_m[m] - x0)
        # arg C = -2*pi*f*D  =>  D = -phase / (2*pi*f)
        delays.append(-unwrapped / (2.0 * math.pi * frequency_hz))

    s, residual = _fit_sin_theta(baselines, delays, array.propagation_speed)
    return AoAEstimate(_angle(s), s, (0.0, *delays), residual, "narrowband", frequency_hz)


def estimate_aoa_gcc_phat(
    channels: Sequence[Sequence[float]],
    sample_rate: float,
    array: LinearArray,
    *,
    interp: int = 16,
) -> AoAEstimate:
    """AoA of a broadband source from GCC-PHAT delays over every element pair.

    For each pair (i, j), the search is bounded by the physical maximum delay
    |x_j - x_i| / c (plus two samples of slack), which rejects spurious peaks
    outside the visible region. All M(M-1)/2 pairwise delays then feed one
    least-squares fit, which averages down per-pair errors.
    """
    _validate(channels, array, sample_rate)
    c = array.propagation_speed
    x = array.positions_m
    baselines: list[float] = []
    delays: list[float] = []
    relative_to_first = [0.0] * array.size
    for i in range(array.size):
        for j in range(i + 1, array.size):
            bound = (x[j] - x[i]) / c + 2.0 / sample_rate
            est = gcc_phat(channels[i], channels[j], sample_rate, max_delay_s=bound, interp=interp)
            baselines.append(x[j] - x[i])
            delays.append(est.delay_s)
            if i == 0:
                relative_to_first[j] = est.delay_s
    s, residual = _fit_sin_theta(baselines, delays, c)
    return AoAEstimate(_angle(s), s, tuple(relative_to_first), residual, "gcc_phat")


def estimate_aoa(
    channels: Sequence[Sequence[float]],
    sample_rate: float,
    array: LinearArray,
    *,
    method: Method = "narrowband",
    frequency_hz: float | None = None,
    frame_size: int | None = None,
    interp: int = 16,
) -> AoAEstimate:
    """Dispatch to the narrowband or GCC-PHAT estimator."""
    if method == "narrowband":
        return estimate_aoa_narrowband(channels, sample_rate, array, frequency_hz, frame_size=frame_size)
    if method == "gcc_phat":
        return estimate_aoa_gcc_phat(channels, sample_rate, array, interp=interp)
    raise ValueError(f"unknown method {method!r}")
