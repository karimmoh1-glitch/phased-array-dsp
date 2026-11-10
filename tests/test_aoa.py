"""End-to-end angle-of-arrival accuracy on simulated plane waves."""

from __future__ import annotations

import math

import pytest

from phasedarray.aoa import (
    SPEED_OF_LIGHT,
    LinearArray,
    estimate_aoa,
    estimate_aoa_gcc_phat,
    estimate_aoa_narrowband,
)
from phasedarray.signals import multitone, simulate_array, simulate_broadband_array

FS = 48_000.0
ULA = LinearArray.uniform(4, 0.1)  # lambda/2 limit = 1715 Hz in air
ANGLES = [-80.0, -60.0, -33.3, -10.0, 0.0, 5.0, 27.5, 45.0, 72.0]


def test_geometry_sign_convention() -> None:
    delays = ULA.arrival_delays(90.0)
    # Source on the +x side: the +x elements hear it first (negative delay).
    assert delays[0] == 0.0
    assert delays[1] == pytest.approx(-0.1 / 343.0)
    assert all(b < a for a, b in zip(delays, delays[1:]))
    assert ULA.arrival_delays(0.0) == [0.0, -0.0, -0.0, -0.0]
    assert ULA.max_unambiguous_frequency() == pytest.approx(1715.0)


def test_array_validation() -> None:
    with pytest.raises(ValueError):
        LinearArray((0.0,))
    with pytest.raises(ValueError):
        LinearArray((0.0, 0.2, 0.1))
    with pytest.raises(ValueError):
        LinearArray((0.0, 0.1), propagation_speed=0.0)


@pytest.mark.parametrize("angle", ANGLES)
@pytest.mark.parametrize("freq", [500.0, 1234.5, 1700.0])
def test_narrowband_noiseless_precision(angle: float, freq: float) -> None:
    channels = simulate_array(ULA, angle, multitone([freq], phases_rad=[0.7]), FS, 8192)
    est = estimate_aoa_narrowband(channels, FS, ULA, freq)
    assert est.angle_deg == pytest.approx(angle, abs=1e-3)
    assert est.residual_rms_s < 1e-9
    assert est.frequency_hz == freq
    # Per-channel delays match the analytic plane-wave model.
    truth = ULA.arrival_delays(angle)
    for measured, expected in zip(est.delays_s, truth):
        assert measured == pytest.approx(expected - truth[0], abs=1e-8)


@pytest.mark.parametrize("angle", [-40.0, 15.0, 63.0])
def test_narrowband_auto_detects_frequency(angle: float) -> None:
    channels = simulate_array(ULA, angle, multitone([987.6]), FS, 8192)
    est = estimate_aoa_narrowband(channels, FS, ULA)
    assert est.frequency_hz is not None
    assert est.frequency_hz == pytest.approx(987.6, abs=0.1)
    assert est.angle_deg == pytest.approx(angle, abs=1e-3)


@pytest.mark.parametrize("angle", [-50.0, 0.0, 20.0, 70.0])
def test_narrowband_with_noise(angle: float) -> None:
    channels = simulate_array(ULA, angle, multitone([1100.0]), FS, 16384, snr_db=0.0, seed=11)
    est = estimate_aoa_narrowband(channels, FS, ULA, 1100.0)
    assert est.angle_deg == pytest.approx(angle, abs=1.0)


def test_narrowband_with_interferer_in_another_band() -> None:
    source = multitone([800.0, 3000.0], amplitudes=[1.0, 1.0])
    channels = simulate_array(ULA, 33.0, source, FS, 8192)
    assert estimate_aoa_narrowband(channels, FS, ULA, 800.0).angle_deg == pytest.approx(33.0, abs=0.01)


def test_narrowband_non_uniform_array() -> None:
    array = LinearArray((0.0, 0.04, 0.13, 0.21, 0.3))
    channels = simulate_array(array, -27.0, multitone([1500.0]), FS, 8192)
    assert estimate_aoa_narrowband(channels, FS, array, 1500.0).angle_deg == pytest.approx(-27.0, abs=1e-3)


def test_narrowband_rf_array() -> None:
    # 2.4 GHz carrier sampled at 10 GS/s by a half-wavelength (6.25 cm) 8-element RF array.
    freq, fs = 2.4e9, 10e9
    array = LinearArray.uniform(8, SPEED_OF_LIGHT / freq / 2, SPEED_OF_LIGHT)
    channels = simulate_array(array, 41.0, multitone([freq]), fs, 4096)
    assert estimate_aoa_narrowband(channels, fs, array, freq).angle_deg == pytest.approx(41.0, abs=1e-3)


def test_narrowband_rejects_spatial_aliasing() -> None:
    channels = simulate_array(ULA, 10.0, multitone([2500.0]), FS, 4096)
    with pytest.raises(ValueError, match="spatial-aliasing"):
        estimate_aoa_narrowband(channels, FS, ULA, 2500.0)


def test_narrowband_rejects_bad_frequency_and_frame() -> None:
    channels = simulate_array(ULA, 10.0, multitone([1000.0]), FS, 1024)
    with pytest.raises(ValueError):
        estimate_aoa_narrowband(channels, FS, ULA, 0.0)
    with pytest.raises(ValueError):
        estimate_aoa_narrowband(channels, FS, ULA, 1000.0, frame_size=2048)


@pytest.mark.parametrize("angle", ANGLES)
def test_gcc_phat_broadband_precision(angle: float) -> None:
    channels = simulate_broadband_array(ULA, angle, FS, 8192, seed=1)
    est = estimate_aoa_gcc_phat(channels, FS, ULA)
    assert est.angle_deg == pytest.approx(angle, abs=0.15)
    assert est.method == "gcc_phat"


@pytest.mark.parametrize("angle", [-35.0, 12.0, 55.0])
def test_gcc_phat_broadband_with_noise(angle: float) -> None:
    channels = simulate_broadband_array(ULA, angle, FS, 8192, snr_db=0.0, seed=2)
    assert estimate_aoa_gcc_phat(channels, FS, ULA).angle_deg == pytest.approx(angle, abs=0.5)


def test_gcc_phat_handles_signal_above_aliasing_limit() -> None:
    # Broadband time-delay estimation has no lambda/2 restriction: wide spacing is fine.
    array = LinearArray.uniform(3, 0.5)
    channels = simulate_broadband_array(array, -18.0, FS, 8192, seed=3)
    assert estimate_aoa_gcc_phat(channels, FS, array).angle_deg == pytest.approx(-18.0, abs=0.05)


def test_estimate_aoa_dispatch_and_validation() -> None:
    channels = simulate_array(ULA, 22.0, multitone([1000.0]), FS, 4096)
    assert estimate_aoa(channels, FS, ULA, frequency_hz=1000.0).method == "narrowband"
    with pytest.raises(ValueError):
        estimate_aoa(channels[:3], FS, ULA)
    with pytest.raises(ValueError):
        estimate_aoa([channels[0], channels[1][:-1], channels[2], channels[3]], FS, ULA)
    with pytest.raises(ValueError):
        estimate_aoa(channels, FS, ULA, method="music")  # type: ignore[arg-type]


def test_sin_theta_is_consistent_with_angle() -> None:
    channels = simulate_array(ULA, -48.0, multitone([1000.0]), FS, 4096)
    est = estimate_aoa_narrowband(channels, FS, ULA, 1000.0)
    assert math.degrees(math.asin(est.sin_theta)) == pytest.approx(est.angle_deg)
