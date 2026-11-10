"""Generate assets/dsp_pipeline.svg: the AoA signal-flow diagram (stdlib only).

Run:  python scripts/make_pipeline_svg.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent.parent / "assets" / "dsp_pipeline.svg"

WIDTH, HEIGHT = 1760, 960
BG = "#0B0C0E"
PANEL = "#121418"
BLUE = "#2F9BFF"
YELLOW = "#E8FF3A"
WHITE = "#F4F6F8"
MUTED = "#8A93A0"

SANS = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', 'SF Mono', Menlo, Consolas, 'DejaVu Sans Mono', monospace"

BLOCK_W, BLOCK_H = 340, 300
GAP = 72
LEFT = (WIDTH - 4 * BLOCK_W - 3 * GAP) // 2
ROW1_Y, ROW2_Y = 150, 560


@dataclass(frozen=True)
class Block:
    number: str
    title: str
    subtitle: str
    lines: tuple[str, ...]  # equation lines; _{..} subscript, ^{..} superscript
    x: int
    y: int

    @property
    def cx(self) -> int:
        return self.x + BLOCK_W // 2

    @property
    def cy(self) -> int:
        return self.y + BLOCK_H // 2


def col(i: int) -> int:
    return LEFT + i * (BLOCK_W + GAP)


BLOCKS = [
    Block("01", "MULTICHANNEL SOURCE", "M-element linear array", (
        "WAV · live stream · simulator",
        "x_{m}[n],  m = 0 … M−1",
        "f_{s} sample rate, N per frame",
        "c = 343 m/s  |  c = 3·10^{8} m/s (RF)",
    ), col(0), ROW1_Y),
    Block("02", "SUB-SAMPLE DELAY MODEL", "far-field plane wave", (
        "τ_{m} = −x_{m} · sin θ / c",
        "x_{m}[n] = s(n/f_{s} − τ_{m}) + w_{m}[n]",
        "exact fractional delays",
        "σ^{2} = P / 10^{SNR/10}",
    ), col(1), ROW1_Y),
    Block("03", "HANN WINDOW TAPER", "leakage suppression", (
        "w[n] = ½ − ½ · cos(2πn / N)",
        "1st sidelobe −31 dB, −18 dB/oct",
        "50 % overlap · COLA",
        "x̃_{m}[n] = w[n] · x_{m}[n]",
    ), col(2), ROW1_Y),
    Block("04", "HAND-ROLLED RFFT CORE", "no numpy · no scipy", (
        "Cooley–Tukey radix-2, N = 2^{p}",
        "X[k] = E[k] + W_{N}^{k} · O[k]",
        "X[k+N/2] = E[k] − W_{N}^{k} · O[k]",
        "Bluestein chirp-z, any N",
        "nk = ½[n^{2} + k^{2} − (k−n)^{2}]",
        "z[n] = x[2n] + j · x[2n+1]",
    ), col(3), ROW1_Y),
    Block("05", "GCC-PHAT CORRELATION", "frequency-domain phase transform", (
        "G_{ij}[k] = X_{i}^{*}[k] · X_{j}[k]",
        "R_{ij}(τ) = IFFT{ G_{ij}[k] / |G_{ij}[k]| }",
        "         = δ(τ − D_{ij})",
        "all M(M−1)/2 element pairs",
    ), col(3), ROW2_Y),
    Block("06", "PEAK SEARCH LOCALIZATION", "sub-sample resolution", (
        "16× sinc interpolation (zero-pad)",
        "δ = ½(y_{−1} − y_{+1}) /",
        "    (y_{−1} − 2y_{0} + y_{+1})",
        "|τ| ≤ |x_{j} − x_{i}| / c + 2/f_{s}",
    ), col(2), ROW2_Y),
    Block("07", "AoA COORDINATE FIT", "linear least squares", (
        "D_{ij} = −(x_{j} − x_{i}) · sin θ / c",
        "ŝ = −c · Σ D_{p} b_{p} / Σ b_{p}^{2}",
        "θ̂ = arcsin(clip(ŝ, −1, 1))",
        "→ θ̂, delays, RMS residual",
    ), col(1), ROW2_Y),
]

# (from, to, label) — arrows follow the snake: row 1 left→right, down, row 2 right→left.
EDGES = [
    (0, 1, "M × N"),
    (1, 2, "x_{m}[n]"),
    (2, 3, "x̃_{m}[n]"),
    (3, 4, "X_{m}[k]"),
    (4, 5, "R_{ij}(τ)"),
    (5, 6, "D_{ij}"),
]

_MARKUP = re.compile(r"([_^])\{([^}]*)\}")


def rich(text: str, size: float) -> str:
    """Escape text and turn _{..} / ^{..} into shifted, smaller tspans.

    Renderers ignore dy on empty tspans, so the baseline reset is carried by
    the next non-empty run of text (or folded into the next script's shift).
    """
    out: list[str] = []
    pos = 0
    offset = 0.0  # current baseline offset from the line's baseline
    small = round(size * 0.68, 1)
    for m in _MARKUP.finditer(text):
        plain = text[pos : m.start()]
        if plain:
            out.append(f'<tspan dy="{-offset:.1f}">{escape(plain)}</tspan>' if offset else escape(plain))
            offset = 0.0
        shift = 0.28 * size if m.group(1) == "_" else -0.42 * size
        out.append(f'<tspan dy="{shift - offset:.1f}" font-size="{small}">{escape(m.group(2))}</tspan>')
        offset = shift
        pos = m.end()
    tail = text[pos:]
    if tail:
        out.append(f'<tspan dy="{-offset:.1f}">{escape(tail)}</tspan>' if offset else escape(tail))
    return "".join(out)


def block_svg(b: Block) -> str:
    parts = [
        f'<g id="block-{b.number}">',
        f'<rect x="{b.x}" y="{b.y}" width="{BLOCK_W}" height="{BLOCK_H}" rx="10" '
        f'fill="{PANEL}" stroke="{YELLOW}" stroke-width="1.5" filter="url(#glow-y)"/>',
        f'<rect x="{b.x}" y="{b.y}" width="{BLOCK_W}" height="4" rx="2" fill="{YELLOW}"/>',
        f'<text x="{b.x + 22}" y="{b.y + 40}" font-family="{MONO}" font-size="15" '
        f'font-weight="700" fill="{YELLOW}">{b.number}</text>',
        f'<text x="{b.x + 56}" y="{b.y + 40}" font-family="{SANS}" font-size="16" '
        f'font-weight="700" letter-spacing="0.6" fill="{YELLOW}">{escape(b.title)}</text>',
        f'<text x="{b.x + 56}" y="{b.y + 62}" font-family="{SANS}" font-size="13" '
        f'fill="{MUTED}">{escape(b.subtitle)}</text>',
        f'<line x1="{b.x + 22}" y1="{b.y + 80}" x2="{b.x + BLOCK_W - 22}" y2="{b.y + 80}" '
        f'stroke="{YELLOW}" stroke-opacity="0.25"/>',
    ]
    line_h = 32 if len(b.lines) > 4 else 40
    for i, line in enumerate(b.lines):
        y = b.y + 116 + i * line_h
        parts.append(
            f'<text x="{b.x + 22}" y="{y}" font-family="{MONO}" font-size="14" '
            f'fill="{WHITE}" xml:space="preserve">{rich(line, 14)}</text>'
        )
    parts.append("</g>")
    return "\n".join(parts)


def edge_svg(a: Block, b: Block, label: str) -> str:
    if a.y == b.y:  # horizontal, either direction
        direction = 1 if b.x > a.x else -1
        x1 = a.x + (BLOCK_W if direction > 0 else 0)
        x2 = b.x + (0 if direction > 0 else BLOCK_W)
        y = a.cy
        path = f"M {x1} {y} L {x2 - direction * 6} {y}"
        lx, ly, anchor = (x1 + x2) / 2, float(y - 14), "middle"
    else:  # vertical drop from row 1 to row 2
        x = a.cx
        path = f"M {x} {a.y + BLOCK_H} L {x} {b.y - 6}"
        lx, ly, anchor = float(x + 14), (a.y + BLOCK_H + b.y) / 2 + 5, "start"
    return (
        f'<path d="{path}" stroke="{BLUE}" stroke-width="2.5" fill="none" '
        f'marker-end="url(#arrow)" filter="url(#glow-b)"/>'
        f'<text x="{lx}" y="{ly}" text-anchor="{anchor}" font-family="{MONO}" font-size="13" '
        f'fill="{BLUE}">{rich(label, 13)}</text>'
    )


def legend_svg() -> str:
    x, y = col(0), ROW2_Y
    rows = [
        ("FFT vs O(N²) DFT", "< 1e-10"),
        ("Narrowband AoA error", "< 0.001°"),
        ("Broadband AoA error", "< 0.15°"),
        ("GCC-PHAT delay error", "< 0.05 smp"),
        ("Unit tests · mypy --strict", "198 · clean"),
    ]
    parts = [
        f'<rect x="{x}" y="{y}" width="{BLOCK_W}" height="{BLOCK_H}" rx="10" fill="none" '
        f'stroke="{MUTED}" stroke-opacity="0.45" stroke-dasharray="4 5"/>',
        f'<text x="{x + 22}" y="{y + 40}" font-family="{SANS}" font-size="16" font-weight="700" '
        f'letter-spacing="0.6" fill="{WHITE}">VALIDATED BOUNDS</text>',
        f'<text x="{x + 22}" y="{y + 62}" font-family="{SANS}" font-size="13" fill="{MUTED}">'
        "asserted by the test suite</text>",
    ]
    for i, (name, value) in enumerate(rows):
        ry = y + 112 + i * 38
        parts.append(
            f'<text x="{x + 22}" y="{ry}" font-family="{SANS}" font-size="14" fill="{MUTED}">{escape(name)}</text>'
            f'<text x="{x + BLOCK_W - 22}" y="{ry}" text-anchor="end" font-family="{MONO}" '
            f'font-size="14" font-weight="700" fill="{WHITE}">{escape(value)}</text>'
        )
    return "\n".join(parts)


def build() -> str:
    defs = f"""<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="{BLUE}"/>
  </marker>
  <!-- userSpaceOnUse: straight arrows have a zero-height bbox, which would clip a bbox-relative filter -->
  <filter id="glow-b" filterUnits="userSpaceOnUse" x="0" y="0" width="{WIDTH}" height="{HEIGHT}">
    <feGaussianBlur stdDeviation="3" result="blur"/>
    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <filter id="glow-y" x="-10%" y="-10%" width="120%" height="120%">
    <feGaussianBlur in="SourceAlpha" stdDeviation="6" result="blur"/>
    <feFlood flood-color="{YELLOW}" flood-opacity="0.18"/>
    <feComposite in2="blur" operator="in" result="glow"/>
    <feMerge><feMergeNode in="glow"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
    <path d="M 40 0 L 0 0 0 40" fill="none" stroke="{WHITE}" stroke-opacity="0.035"/>
  </pattern>
</defs>"""
    header = (
        f'<text x="{LEFT}" y="72" font-family="{SANS}" font-size="30" font-weight="800" fill="{WHITE}">'
        "Phased-Array Angle-of-Arrival Pipeline</text>"
        f'<text x="{LEFT}" y="104" font-family="{SANS}" font-size="16" fill="{MUTED}">'
        "Pure-Python DSP for SDR &amp; acoustic arrays · hand-rolled FFT · GCC-PHAT · least-squares AoA</text>"
        f'<text x="{WIDTH - LEFT}" y="72" text-anchor="end" font-family="{MONO}" font-size="14" '
        f'fill="{BLUE}">phasedarray v0.1</text>'
    )
    footer = (
        f'<text x="{LEFT}" y="{HEIGHT - 32}" font-family="{MONO}" font-size="13" fill="{MUTED}">'
        f'{rich("Narrowband branch: C_{m} = Σ X_{m−1}^{*}[k] · X_{m}[k] at the tone bin → unwrap arg C_{m} → same LS fit (spacing ≤ λ/2)", 13)}'
        "</text>"
    )
    body = [block_svg(b) for b in BLOCKS]
    body += [edge_svg(BLOCKS[a], BLOCKS[b], label) for a, b, label in EDGES]
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        "<title id=\"title\">Phased-array angle-of-arrival DSP pipeline</title>",
        "<desc id=\"desc\">Signal flow from a multichannel source through a sub-sample delay model, "
        "Hann window, radix-2/Bluestein RFFT, GCC-PHAT cross-correlation and peak search to a "
        "least-squares angle-of-arrival fit.</desc>",
        defs,
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BG}"/>',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="url(#grid)"/>',
        header,
        *body,
        legend_svg(),
        footer,
        "</svg>",
    ])


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
