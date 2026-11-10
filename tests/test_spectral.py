from __future__ import annotations

import math

import pytest

from phasedarray.spectral import dominant_frequency, frame_starts, hann, power_spectrum
from phasedarray.signals import tone


def test_hann_window_properties() -> None:
    w = hann(64)
    assert w[0] == 0.0
    assert math.isclose(w[32], 1.0)
    assert math.isclose(sum(w), 32.0, abs_tol=1e-12)  # periodic Hann sums to N/2
    # 50%-overlapped periodic Hann windows sum to exactly 1 (COLA)
    assert all(math.isclose(w[i] + w[i + 32], 1.0) for i in range(32))


def test_hann_rejects_bad_length() -> None:
    with pytest.raises(ValueError):
        hann(0)


@pytest.mark.parametrize("freq", [440.0, 1000.0, 1234.567, 3999.9, 7777.7])
def test_dominant_frequency_on_and_off_bin(freq: float) -> None:
    fs, n = 16_000.0, 4096
    estimate = dominant_frequency(tone(freq, fs, n, phase_rad=0.4), fs)
    bin_width = fs / n
    assert abs(estimate - freq) < 0.01 * bin_width


def test_power_spectrum_peak_location() -> None:
    n, k = 256, 20
    ps = power_spectrum([math.cos(2 * math.pi * k * i / n) for i in range(n)])
    assert max(range(len(ps)), key=ps.__getitem__) == k


def test_frame_starts() -> None:
    assert frame_starts(10, 4, 2) == [0, 2, 4, 6]
    assert frame_starts(3, 4, 2) == []
