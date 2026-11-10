"""Precision tests for the from-scratch FFT against the DFT definition and closed forms."""

from __future__ import annotations

import cmath
import math

import pytest

from phasedarray.fft import (
    fft,
    fftfreq,
    ifft,
    irfft,
    is_power_of_two,
    next_power_of_two,
    rfft,
    rfftfreq,
)
from reference import max_abs_diff, naive_dft, random_complex, random_real

POW2_SIZES = [1, 2, 4, 8, 16, 32, 64, 256, 1024]
OTHER_SIZES = [3, 5, 6, 7, 12, 15, 100, 127, 243, 1000]  # exercises Bluestein (incl. primes)
ALL_SIZES = POW2_SIZES + OTHER_SIZES
TOL = 1e-10


@pytest.mark.parametrize("n", ALL_SIZES)
def test_fft_matches_naive_dft(n: int) -> None:
    x = random_complex(n, seed=n)
    assert max_abs_diff(fft(x), naive_dft(x)) < TOL * max(1, n)


@pytest.mark.parametrize("n", ALL_SIZES)
def test_ifft_matches_naive_inverse_dft(n: int) -> None:
    spectrum = random_complex(n, seed=1000 + n)
    assert max_abs_diff(ifft(spectrum), naive_dft(spectrum, inverse=True)) < TOL


@pytest.mark.parametrize("n", ALL_SIZES + [4096, 4095])
def test_round_trip_is_identity(n: int) -> None:
    x = random_complex(n, seed=2000 + n)
    assert max_abs_diff(ifft(fft(x)), x) < 1e-12


@pytest.mark.parametrize("n", [8, 64, 1024, 1000])
@pytest.mark.parametrize("k", [1, 3])
def test_cosine_on_bin_gives_two_spikes(n: int, k: int) -> None:
    amp = 2.5
    x = [amp * math.cos(2 * math.pi * k * i / n) for i in range(n)]
    spectrum = fft(x)
    expected = [0j] * n
    expected[k] = expected[n - k] = complex(amp * n / 2)
    assert max_abs_diff(spectrum, expected) < 1e-9


@pytest.mark.parametrize("n", [16, 1024, 300])
def test_sine_on_bin_is_purely_imaginary(n: int) -> None:
    k = 5
    x = [math.sin(2 * math.pi * k * i / n) for i in range(n)]
    spectrum = fft(x)
    assert abs(spectrum[k] - complex(0, -n / 2)) < 1e-9
    assert abs(spectrum[n - k] - complex(0, n / 2)) < 1e-9


@pytest.mark.parametrize("n", [1, 8, 100])
def test_unit_impulse_is_flat(n: int) -> None:
    x = [1.0] + [0.0] * (n - 1)
    assert max_abs_diff(fft(x), [1.0] * n) < 1e-12


@pytest.mark.parametrize("n,d", [(64, 3), (100, 17)])
def test_shifted_impulse_is_linear_phase(n: int, d: int) -> None:
    x = [0.0] * n
    x[d] = 1.0
    expected = [cmath.exp(-2j * math.pi * k * d / n) for k in range(n)]
    assert max_abs_diff(fft(x), expected) < 1e-12


@pytest.mark.parametrize("n", [32, 45])
def test_constant_is_dc_only(n: int) -> None:
    expected = [0j] * n
    expected[0] = complex(3.0 * n)
    assert max_abs_diff(fft([3.0] * n), expected) < 1e-11


@pytest.mark.parametrize("n", [64, 1000])
def test_geometric_sequence_closed_form(n: int) -> None:
    # sum_j a^j e^{-2 pi i jk/N} = (1 - a^N) / (1 - a e^{-2 pi i k/N})
    a = 0.97 * cmath.exp(0.3j)
    x = [a**j for j in range(n)]
    expected = [(1 - a**n) / (1 - a * cmath.exp(-2j * math.pi * k / n)) for k in range(n)]
    assert max_abs_diff(fft(x), expected) < 1e-10


@pytest.mark.parametrize("n", [256, 210])
def test_parseval(n: int) -> None:
    x = random_complex(n, seed=7)
    time_energy = sum(abs(v) ** 2 for v in x)
    freq_energy = sum(abs(v) ** 2 for v in fft(x)) / n
    assert math.isclose(time_energy, freq_energy, rel_tol=1e-12)


def test_linearity() -> None:
    n = 128
    x, y = random_complex(n, 1), random_complex(n, 2)
    a, b = 1.5 - 0.5j, -2.0 + 0.25j
    combined = fft([a * p + b * q for p, q in zip(x, y)])
    separate = [a * p + b * q for p, q in zip(fft(x), fft(y))]
    assert max_abs_diff(combined, separate) < 1e-11


@pytest.mark.parametrize("n", [64, 90])
def test_circular_shift_theorem(n: int) -> None:
    x = random_complex(n, 3)
    m = 11
    shifted = [x[(i - m) % n] for i in range(n)]
    expected = [v * cmath.exp(-2j * math.pi * k * m / n) for k, v in enumerate(fft(x))]
    assert max_abs_diff(fft(shifted), expected) < 1e-11


@pytest.mark.parametrize("n", [32, 50])
def test_convolution_theorem(n: int) -> None:
    x, h = random_complex(n, 4), random_complex(n, 5)
    direct = [sum(x[j] * h[(i - j) % n] for j in range(n)) for i in range(n)]
    via_fft = ifft([p * q for p, q in zip(fft(x), fft(h))])
    assert max_abs_diff(via_fft, direct) < 1e-11


@pytest.mark.parametrize("n", [2, 4, 64, 1024, 6, 100, 1, 3, 7, 101])
def test_rfft_equals_first_half_of_fft(n: int) -> None:
    x = random_real(n, seed=n)
    assert max_abs_diff(rfft(x), fft(x)[: n // 2 + 1]) < 1e-11


@pytest.mark.parametrize("n", [2, 64, 1024, 100, 3, 101])
def test_irfft_inverts_rfft(n: int) -> None:
    x = random_real(n, seed=50 + n)
    assert max_abs_diff(irfft(rfft(x), n), x) < 1e-12


def test_irfft_zero_padding_interpolates_band_limited_signal() -> None:
    n, k, up = 32, 3, 4
    x = [math.cos(2 * math.pi * k * i / n) for i in range(n)]
    fine = irfft(rfft(x), n * up)
    expected = [math.cos(2 * math.pi * k * i / (n * up)) / up for i in range(n * up)]
    assert max_abs_diff(fine, expected) < 1e-12


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError):
        fft([])
    with pytest.raises(ValueError):
        rfft([])
    with pytest.raises(ValueError):
        irfft([0j], 0)


def test_frequency_axes() -> None:
    assert fftfreq(4, 8.0) == [0.0, 2.0, -4.0, -2.0]
    assert fftfreq(5, 5.0) == [0.0, 1.0, 2.0, -2.0, -1.0]
    assert rfftfreq(8, 16.0) == [0.0, 2.0, 4.0, 6.0, 8.0]


def test_power_of_two_helpers() -> None:
    assert [is_power_of_two(n) for n in (0, 1, 2, 3, 4, 6, 1024)] == [False, True, True, False, True, False, True]
    assert [next_power_of_two(n) for n in (0, 1, 2, 3, 5, 1024, 1025)] == [1, 1, 2, 4, 8, 1024, 2048]
