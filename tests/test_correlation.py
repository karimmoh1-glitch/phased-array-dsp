from __future__ import annotations

import random

import pytest

from phasedarray.aoa import LinearArray
from phasedarray.correlation import cross_correlate, gcc_phat
from phasedarray.signals import simulate_broadband_array
from reference import random_real

FS = 48_000.0


def test_cross_correlate_matches_direct_sum() -> None:
    a, b = random_real(37, 1), random_real(50, 2)
    lags, values = cross_correlate(a, b)
    assert lags[0] == -36 and lags[-1] == 49
    for lag, v in zip(lags, values):
        direct = sum(a[n] * b[n + lag] for n in range(len(a)) if 0 <= n + lag < len(b))
        assert abs(v - direct) < 1e-12


@pytest.mark.parametrize("delay", [0, 5, -7, 31])
def test_cross_correlate_peak_at_integer_delay(delay: int) -> None:
    x = random_real(600, 3)
    a = x[100:400]
    b = x[100 - delay : 400 - delay]  # b[n] = a[n - delay]
    lags, values = cross_correlate(a, b)
    assert lags[max(range(len(values)), key=values.__getitem__)] == delay


@pytest.mark.parametrize("delay", [0, 1, -4, 23, -60])
def test_gcc_phat_integer_delay(delay: int) -> None:
    x = random_real(5000, 4)
    a = x[1000:3048]
    b = x[1000 - delay : 3048 - delay]
    est = gcc_phat(a, b, FS)
    assert abs(est.delay_samples - delay) < 0.01
    assert est.delay_s == pytest.approx(est.delay_samples / FS)


@pytest.mark.parametrize("delay", [0.25, -0.5, 1.37, -3.81, 12.125])
def test_gcc_phat_fractional_delay(delay: float) -> None:
    # Two elements spaced so an endfire source gives exactly `delay` samples:
    # tau_1 = -x_1 sin(theta) / c, so theta = -90 deg delays element 1 by +x_1/c.
    c = 343.0
    array = LinearArray((0.0, abs(delay) / FS * c), c)
    angle = -90.0 if delay > 0 else 90.0
    a, b = simulate_broadband_array(array, angle, FS, 4096, seed=9)
    est = gcc_phat(a, b, FS)
    # Finite-record estimator variance is a few hundredths of a sample (~1 us at 48 kHz).
    assert abs(est.delay_samples - delay) < 0.05


def test_gcc_phat_identical_signals_have_unit_peak() -> None:
    x = random_real(1024, 5)
    est = gcc_phat(x, x, FS)
    assert abs(est.delay_samples) < 1e-9
    assert est.peak == pytest.approx(1.0, abs=0.05)


def test_gcc_phat_with_noise() -> None:
    rng = random.Random(6)
    x = random_real(6000, 6)
    a = [v + rng.gauss(0, 0.3) for v in x[2000:6000]]
    b = [v + rng.gauss(0, 0.3) for v in x[1990:5990]]  # 10-sample delay, ~10 dB SNR
    assert abs(gcc_phat(a, b, FS).delay_samples - 10) < 0.05


def test_gcc_phat_respects_max_delay() -> None:
    x = random_real(5000, 7)
    a, b = x[1000:3000], x[960:2960]  # true delay 40
    bounded = gcc_phat(a, b, FS, max_delay_s=10 / FS)
    assert abs(bounded.delay_samples) <= 11


def test_gcc_phat_validates_inputs() -> None:
    with pytest.raises(ValueError):
        gcc_phat([], [1.0], FS)
    with pytest.raises(ValueError):
        gcc_phat([1.0], [1.0], 0.0)
    with pytest.raises(ValueError):
        gcc_phat([1.0], [1.0], FS, interp=0)
