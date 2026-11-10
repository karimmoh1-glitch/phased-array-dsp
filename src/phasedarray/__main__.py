"""Command-line entry point: `python -m phasedarray simulate|wav ...`."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .aoa import SPEED_OF_SOUND, AoAEstimate, LinearArray, Method, estimate_aoa
from .audio_io import read_wav
from .signals import multitone, simulate_array, simulate_broadband_array


def _report(est: AoAEstimate, truth: float | None = None) -> str:
    lines = [f"method         : {est.method}"]
    if est.frequency_hz is not None:
        lines.append(f"frequency      : {est.frequency_hz:.3f} Hz")
    lines.append(f"angle of arrival: {est.angle_deg:+.4f} deg")
    if truth is not None:
        lines.append(f"true angle     : {truth:+.4f} deg (error {est.angle_deg - truth:+.5f} deg)")
    lines.append("delays vs ch0  : " + ", ".join(f"{d * 1e6:+.3f} us" for d in est.delays_s))
    lines.append(f"fit residual   : {est.residual_rms_s * 1e9:.3f} ns RMS")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phasedarray", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def add_array_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--spacing", type=float, default=0.1, help="ULA element spacing in metres")
        sp.add_argument("--speed", type=float, default=SPEED_OF_SOUND, help="propagation speed in m/s")
        sp.add_argument("--method", choices=["narrowband", "gcc_phat"], default="narrowband")
        sp.add_argument("--freq", type=float, default=None, help="tone frequency in Hz (narrowband)")

    sim = sub.add_parser("simulate", help="simulate a plane wave hitting a ULA and estimate its AoA")
    add_array_args(sim)
    sim.add_argument("--angle", type=float, default=30.0, help="true angle from broadside, degrees")
    sim.add_argument("--mics", type=int, default=4)
    sim.add_argument("--fs", type=float, default=48_000.0)
    sim.add_argument("--samples", type=int, default=8192)
    sim.add_argument("--snr", type=float, default=None, help="per-channel SNR in dB (default: noiseless)")
    sim.add_argument("--seed", type=int, default=0)

    wav = sub.add_parser("wav", help="estimate AoA from a multichannel WAV file (one channel per element)")
    add_array_args(wav)
    wav.add_argument("path")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    method: Method = args.method
    if args.command == "simulate":
        array = LinearArray.uniform(args.mics, args.spacing, args.speed)
        freq: float | None = None
        if method == "narrowband":
            freq = args.freq if args.freq is not None else 1000.0
            channels = simulate_array(
                array, args.angle, multitone([freq]), args.fs, args.samples, snr_db=args.snr, seed=args.seed
            )
        else:
            channels = simulate_broadband_array(
                array, args.angle, args.fs, args.samples, snr_db=args.snr, seed=args.seed
            )
        est = estimate_aoa(channels, args.fs, array, method=method, frequency_hz=freq)
        print(_report(est, args.angle))
    else:
        fs, channels = read_wav(args.path)
        array = LinearArray.uniform(len(channels), args.spacing, args.speed)
        est = estimate_aoa(channels, fs, array, method=method, frequency_hz=args.freq)
        print(_report(est))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
