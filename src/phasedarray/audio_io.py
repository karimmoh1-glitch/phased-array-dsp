"""Multichannel PCM WAV read/write using only the standard library."""

from __future__ import annotations

import wave
from collections.abc import Sequence
from os import PathLike

__all__ = ["read_wav", "write_wav"]

StrPath = str | PathLike[str]


def _full_scale(sample_width: int) -> int:
    return 1 << (8 * sample_width - 1)


def read_wav(path: StrPath) -> tuple[int, list[list[float]]]:
    """Return (sample_rate, channels) with samples normalised to [-1, 1)."""
    with wave.open(str(path), "rb") as w:
        n_channels = w.getnchannels()
        width = w.getsampwidth()
        sample_rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    if width not in (1, 2, 3, 4):
        raise ValueError(f"unsupported sample width: {width} bytes")
    scale = 1.0 / _full_scale(width)
    if width == 1:  # 8-bit WAV is unsigned with a 128 offset
        samples = [(b - 128) * scale for b in raw]
    else:
        samples = [
            int.from_bytes(raw[i : i + width], "little", signed=True) * scale
            for i in range(0, len(raw), width)
        ]
    return sample_rate, [samples[c::n_channels] for c in range(n_channels)]


def write_wav(path: StrPath, sample_rate: int, channels: Sequence[Sequence[float]], *, sample_width: int = 2) -> None:
    """Write float channels in [-1, 1] as interleaved little-endian PCM (values are clipped)."""
    if sample_width not in (1, 2, 3, 4):
        raise ValueError(f"unsupported sample width: {sample_width} bytes")
    if not channels or any(len(ch) != len(channels[0]) for ch in channels):
        raise ValueError("need at least one channel, all the same length")
    full = _full_scale(sample_width)
    out = bytearray()
    for frame in zip(*channels):
        for v in frame:
            q = max(-full, min(full - 1, round(v * full)))
            if sample_width == 1:
                out.append(q + 128)
            else:
                out += q.to_bytes(sample_width, "little", signed=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(len(channels))
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        w.writeframes(bytes(out))
