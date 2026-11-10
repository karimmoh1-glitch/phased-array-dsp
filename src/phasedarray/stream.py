"""Block-streaming AoA estimation for live audio / SDR buffers."""

from __future__ import annotations

from collections.abc import Sequence

from .aoa import AoAEstimate, LinearArray, Method, estimate_aoa

__all__ = ["StreamingAoAEstimator"]


class StreamingAoAEstimator:
    """Accumulates multichannel blocks of any size and emits one estimate per hop.

    Each estimate covers the most recent `frame_size` samples; consecutive
    estimates are `hop` samples apart (default: 50% overlap).
    """

    def __init__(
        self,
        array: LinearArray,
        sample_rate: float,
        *,
        frame_size: int = 4096,
        hop: int | None = None,
        method: Method = "narrowband",
        frequency_hz: float | None = None,
        interp: int = 16,
    ) -> None:
        if frame_size < 4:
            raise ValueError("frame_size must be at least 4")
        self.array = array
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop = hop if hop is not None else frame_size // 2
        if not 0 < self.hop <= frame_size:
            raise ValueError("hop must be in (0, frame_size]")
        self.method: Method = method
        self.frequency_hz = frequency_hz
        self.interp = interp
        self._buffers: list[list[float]] = [[] for _ in range(array.size)]

    @property
    def buffered(self) -> int:
        return len(self._buffers[0])

    def push(self, block: Sequence[Sequence[float]]) -> list[AoAEstimate]:
        """Append one block (channels x samples); return estimates for every frame completed."""
        if len(block) != self.array.size:
            raise ValueError(f"block has {len(block)} channels, expected {self.array.size}")
        if any(len(ch) != len(block[0]) for ch in block):
            raise ValueError("all channels in a block must be the same length")
        for buf, ch in zip(self._buffers, block):
            buf.extend(ch)
        results: list[AoAEstimate] = []
        while self.buffered >= self.frame_size:
            frame = [buf[: self.frame_size] for buf in self._buffers]
            results.append(
                estimate_aoa(
                    frame,
                    self.sample_rate,
                    self.array,
                    method=self.method,
                    frequency_hz=self.frequency_hz,
                    frame_size=self.frame_size,
                    interp=self.interp,
                )
            )
            for buf in self._buffers:
                del buf[: self.hop]
        return results

    def reset(self) -> None:
        for buf in self._buffers:
            buf.clear()
