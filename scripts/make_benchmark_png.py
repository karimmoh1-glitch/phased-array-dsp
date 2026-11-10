"""Generate assets/aoa_benchmark.png: measured delay and AoA accuracy (300 DPI).

Every point is computed live by the phasedarray library; nothing is hard-coded.
Run:  python scripts/make_benchmark_png.py      (needs matplotlib; ~1 minute)
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.projections.polar import PolarAxes  # noqa: E402

from phasedarray import (  # noqa: E402
    LinearArray,
    estimate_aoa_gcc_phat,
    estimate_aoa_narrowband,
    gcc_phat,
    multitone,
    simulate_array,
    simulate_broadband_array,
)

OUT = ROOT / "assets" / "aoa_benchmark.png"

BG = "#0B0C0E"
PANEL = "#121418"
BLUE = "#2F9BFF"
YELLOW = "#E8FF3A"
WHITE = "#F4F6F8"
MUTED = "#8A93A0"
GRID = "#2A2E35"

FS = 48_000.0
C = 343.0
SNRS_DB = [-5.0 + 2.5 * i for i in range(11)]  # -5 … +20 dB
TRIALS = 40
DELAY_BOUND = 0.05  # samples, asserted in tests/test_correlation.py
ANGLES = [-80.0 + 8.0 * i for i in range(20)]  # -80 … +72 deg
TONE_HZ = 1234.5
ULA = LinearArray.uniform(4, 0.1)


def delay_errors_vs_snr(seed: int = 42) -> dict[float, list[float]]:
    """|estimated - true| GCC-PHAT delay (samples) for random fractional delays in ±8 samples."""
    rng = random.Random(seed)
    results: dict[float, list[float]] = {}
    for snr in SNRS_DB:
        errs = []
        for _ in range(TRIALS):
            true = rng.uniform(-8.0, 8.0)
            # Two elements spaced so an endfire source produces exactly `true` samples of delay.
            pair = LinearArray((0.0, abs(true) / FS * C), C)
            a, b = simulate_broadband_array(
                pair, -90.0 if true > 0 else 90.0, FS, 4096, snr_db=snr, seed=rng.randrange(1 << 30)
            )
            errs.append(abs(gcc_phat(a, b, FS).delay_samples - true))
        results[snr] = errs
        print(f"  SNR {snr:+5.1f} dB  RMS {rms(errs):.4f} samples")
    return results


def rms(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values))


def aoa_sweep() -> tuple[list[float], list[float]]:
    narrow, broad = [], []
    for angle in ANGLES:
        tone = simulate_array(ULA, angle, multitone([TONE_HZ], phases_rad=[0.7]), FS, 8192)
        narrow.append(estimate_aoa_narrowband(tone, FS, ULA, TONE_HZ).angle_deg)
        noise = simulate_broadband_array(ULA, angle, FS, 8192, seed=1)
        broad.append(estimate_aoa_gcc_phat(noise, FS, ULA).angle_deg)
    return narrow, broad


def style_axes(ax: Axes) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def plot_delay(ax: Axes, errors: dict[float, list[float]]) -> None:
    style_axes(ax)
    jitter = random.Random(0)
    floor = 1e-4  # log axis: show exact hits at the floor rather than dropping them
    xs, ys = [], []
    for snr, errs in errors.items():
        for e in errs:
            xs.append(snr + jitter.uniform(-0.55, 0.55))
            ys.append(max(e, floor))
    ax.scatter(xs, ys, s=9, color=BLUE, alpha=0.35, linewidths=0, label=f"Single trial ({TRIALS} per SNR)")
    snrs = list(errors)
    rms_vals = [rms(errors[s]) for s in snrs]
    ax.plot(snrs, rms_vals, color=YELLOW, lw=2, marker="o", ms=6, mec=BG, mew=1.5, label="RMS error", zorder=3)
    ax.axhline(DELAY_BOUND, color=WHITE, lw=1.2, ls=(0, (5, 4)), alpha=0.8, zorder=2)
    ax.text(21.2, DELAY_BOUND * 1.12, "0.05-sample bound", color=WHITE, fontsize=8.5, ha="right", va="bottom")
    ax.annotate(
        f"RMS {rms_vals[-1]:.3f} smp ≈ {rms_vals[-1] / FS * 1e9:.0f} ns",
        xy=(snrs[-1], rms_vals[-1]), xytext=(16.0, 0.026),
        color=YELLOW, fontsize=8.5, ha="center",
        arrowprops={"arrowstyle": "-", "color": YELLOW, "lw": 0.8},
    )
    ax.set_yscale("log")
    ax.set_ylim(2e-5, 0.5)  # headroom below the data for the legend
    ax.set_xlim(-6.5, 21.5)
    ax.set_xticks(SNRS_DB[::2])
    ax.set_xlabel("Per-channel SNR (dB)", color=WHITE, fontsize=10)
    ax.set_ylabel("|Delay error| (samples, log scale)", color=WHITE, fontsize=10)
    ax.set_title("GCC-PHAT delay accuracy vs SNR", color=WHITE, fontsize=12, fontweight="bold", loc="left", pad=10)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.grid(True, which="minor", axis="y", color=GRID, lw=0.3, alpha=0.5)
    leg = ax.legend(loc="lower left", fontsize=8.5, frameon=True, facecolor=BG, edgecolor=GRID, labelcolor=WHITE)
    leg.get_frame().set_alpha(0.9)


def plot_polar(ax: PolarAxes, narrow: list[float], broad: list[float]) -> None:
    ax.set_facecolor(PANEL)
    ax.set_theta_zero_location("N")  # broadside straight up
    ax.set_theta_direction(-1)  # +θ towards +x (right), matching the library convention
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([])
    ax.set_xticks([math.radians(a) for a in range(-90, 91, 30)])
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, lw=0.6)
    ax.spines["polar"].set_color(GRID)

    for angle in ANGLES:  # true source directions
        t = math.radians(angle)
        ax.plot([t, t], [0, 1.05], color=WHITE, lw=0.8, alpha=0.35, zorder=1)
    ax.scatter([math.radians(a) for a in narrow], [0.95] * len(narrow), s=42, marker="D",
               color=YELLOW, edgecolors=BG, linewidths=1.0, zorder=3)
    ax.scatter([math.radians(a) for a in broad], [0.72] * len(broad), s=40, marker="o",
               color=BLUE, edgecolors=BG, linewidths=1.0, zorder=3)

    n_err = max(abs(e - t) for e, t in zip(narrow, ANGLES))
    b_err = max(abs(e - t) for e, t in zip(broad, ANGLES))
    handles = [
        Line2D([], [], color=WHITE, alpha=0.5, lw=1, label="True source direction (−80° … +72°)"),
        Line2D([], [], color=YELLOW, marker="D", ls="", ms=7, mec=BG,
               label=f"Single-frequency  max err {n_err:.1e}°  (bound 0.001°)"),
        Line2D([], [], color=BLUE, marker="o", ls="", ms=7, mec=BG,
               label=f"Broadband GCC-PHAT  max err {b_err:.3f}°  (bound 0.15°)"),
    ]
    fig = ax.get_figure()
    assert fig is not None
    # Figure-level title and legend so neither can overlap the half-disc.
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.75, 0.25), fontsize=8.5,
               frameon=True, facecolor=BG, edgecolor=GRID, labelcolor=WHITE)
    fig.text(0.75, 0.885, "Reconstructed angle of arrival vs truth", color=WHITE, fontsize=12,
             fontweight="bold", ha="center")
    fig.text(0.75, 0.85, "4-mic ULA, 10 cm spacing · markers sit on the true rays", color=MUTED,
             fontsize=9, ha="center")
    print(f"  narrowband max error {n_err:.2e} deg, broadband max error {b_err:.4f} deg")


def main() -> None:
    print("delay accuracy sweep …")
    errors = delay_errors_vs_snr()
    print("angle sweep …")
    narrow, broad = aoa_sweep()

    fig = plt.figure(figsize=(14, 6.2), facecolor=BG)
    ax_delay = fig.add_axes((0.06, 0.13, 0.42, 0.72))
    ax_polar = fig.add_axes((0.54, -0.06, 0.42, 1.2), projection="polar")
    plot_delay(ax_delay, errors)
    assert isinstance(ax_polar, PolarAxes)
    plot_polar(ax_polar, narrow, broad)
    fig.suptitle("phasedarray · measured accuracy (pure-Python FFT, no numpy)",
                 color=WHITE, fontsize=14, fontweight="bold", x=0.06, ha="left", y=0.97)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, facecolor=BG)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
