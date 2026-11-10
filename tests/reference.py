"""Slow, obviously-correct reference implementations used as test oracles."""

from __future__ import annotations

import cmath
import math
import random
from collections.abc import Sequence


def naive_dft(x: Sequence[complex], inverse: bool = False) -> list[complex]:
    """O(N^2) DFT straight from the definition. (k*n mod N) keeps exp() arguments small."""
    n = len(x)
    sign = 1.0 if inverse else -1.0
    out = [
        sum(complex(x[j]) * cmath.exp(complex(0.0, sign * 2.0 * math.pi * ((j * k) % n) / n)) for j in range(n))
        for k in range(n)
    ]
    return [v / n for v in out] if inverse else out


def random_complex(n: int, seed: int) -> list[complex]:
    rng = random.Random(seed)
    return [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(n)]


def random_real(n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(n)]


def max_abs_diff(a: Sequence[complex], b: Sequence[complex]) -> float:
    assert len(a) == len(b), f"length mismatch {len(a)} != {len(b)}"
    return max(abs(complex(p) - complex(q)) for p, q in zip(a, b))
