from __future__ import annotations

import re
from pathlib import Path

import pytest

from phasedarray.__main__ import main
from phasedarray.aoa import LinearArray, estimate_aoa_narrowband
from phasedarray.audio_io import read_wav, write_wav
from phasedarray.signals import multitone, simulate_array
from phasedarray.stream import StreamingAoAEstimator

FS = 48_000
ULA = LinearArray.uniform(4, 0.1)


def test_streaming_emits_one_estimate_per_hop() -> None:
    total, frame, block = 20_000, 4096, 1500
    channels = simulate_array(ULA, 25.0, multitone([1200.0]), FS, total)
    est = StreamingAoAEstimator(ULA, FS, frame_size=frame, frequency_hz=1200.0)
    results = []
    for start in range(0, total, block):
        results += est.push([ch[start : start + block] for ch in channels])
    assert len(results) == (total - frame) // (frame // 2) + 1
    assert all(r.angle_deg == pytest.approx(25.0, abs=1e-3) for r in results)
    assert est.buffered < frame
    est.reset()
    assert est.buffered == 0


def test_streaming_validates_blocks() -> None:
    est = StreamingAoAEstimator(ULA, FS)
    with pytest.raises(ValueError):
        est.push([[0.0]] * 3)
    with pytest.raises(ValueError):
        est.push([[0.0], [0.0], [0.0], [0.0, 0.0]])
    with pytest.raises(ValueError):
        StreamingAoAEstimator(ULA, FS, frame_size=64, hop=128)


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_wav_round_trip(tmp_path: Path, width: int) -> None:
    channels = simulate_array(ULA, 0.0, multitone([440.0], amplitudes=[0.8]), FS, 500)
    path = tmp_path / "x.wav"
    write_wav(path, FS, channels, sample_width=width)
    fs, back = read_wav(path)
    assert fs == FS and len(back) == 4 and len(back[0]) == 500
    quantum = 1.0 / (1 << (8 * width - 1))
    for orig, got in zip(channels, back):
        assert max(abs(a - b) for a, b in zip(orig, got)) <= quantum


def test_aoa_from_16bit_wav(tmp_path: Path) -> None:
    channels = simulate_array(ULA, -36.0, multitone([1000.0], amplitudes=[0.5]), FS, 8192, snr_db=30, seed=1)
    path = tmp_path / "array.wav"
    write_wav(path, FS, channels)
    fs, back = read_wav(path)
    assert estimate_aoa_narrowband(back, fs, ULA, 1000.0).angle_deg == pytest.approx(-36.0, abs=0.1)


def test_cli_simulate_and_wav(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["simulate", "--angle", "12", "--freq", "900", "--samples", "4096"]) == 0
    assert "+12.0000 deg" in capsys.readouterr().out
    assert main(["simulate", "--angle", "-30", "--method", "gcc_phat", "--samples", "4096"]) == 0
    assert "angle of arrival" in capsys.readouterr().out
    path = tmp_path / "a.wav"
    write_wav(path, FS, simulate_array(ULA, 20.0, multitone([1000.0], amplitudes=[0.5]), FS, 4096))
    assert main(["wav", str(path), "--spacing", "0.1", "--freq", "1000"]) == 0
    match = re.search(r"angle of arrival: ([+-][\d.]+) deg", capsys.readouterr().out)
    assert match is not None and float(match.group(1)) == pytest.approx(20.0, abs=0.01)
