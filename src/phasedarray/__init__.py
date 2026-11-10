"""Pure-Python DSP pipeline for SDR / acoustic phased-array angle-of-arrival estimation."""

from .aoa import (
    SPEED_OF_LIGHT,
    SPEED_OF_SOUND,
    AoAEstimate,
    LinearArray,
    estimate_aoa,
    estimate_aoa_gcc_phat,
    estimate_aoa_narrowband,
)
from .audio_io import read_wav, write_wav
from .correlation import DelayEstimate, cross_correlate, gcc_phat
from .fft import fft, fftfreq, ifft, irfft, rfft, rfftfreq
from .signals import multitone, simulate_array, simulate_broadband_array, tone
from .spectral import dominant_frequency, hann, power_spectrum
from .stream import StreamingAoAEstimator

__all__ = [
    "SPEED_OF_LIGHT",
    "SPEED_OF_SOUND",
    "AoAEstimate",
    "DelayEstimate",
    "LinearArray",
    "StreamingAoAEstimator",
    "cross_correlate",
    "dominant_frequency",
    "estimate_aoa",
    "estimate_aoa_gcc_phat",
    "estimate_aoa_narrowband",
    "fft",
    "fftfreq",
    "gcc_phat",
    "hann",
    "ifft",
    "irfft",
    "multitone",
    "power_spectrum",
    "read_wav",
    "rfft",
    "rfftfreq",
    "simulate_array",
    "simulate_broadband_array",
    "tone",
    "write_wav",
]

__version__ = "0.1.0"
