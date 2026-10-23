"""
Organic Perlin-noise camera shake for media layers.

Pure-Python Perlin noise (Ken Perlin's improved algorithm) with optional
fractal Brownian motion (fBm) layering. Produces smooth, organic camera
offsets that never feel mechanical -- eliminating the sterile digital look
from static or sinusoid-driven shakes.

The class generates per-frame (dx, dy, dzoom) offsets that callers apply
to moviepy clips via `fl_image`/`fl` or bake into ffmpeg filter graphs.

No heavy third-party noise libraries required; only numpy.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

import numpy as np

LOG_PREFIX = "[perlin_shake]"


def log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", flush=True)


def _build_permutation_table(seed: int = 1013) -> List[int]:
    """Build the classic Perlin permutation table (256 entries, doubled)."""
    rng = random.Random(seed)
    p = list(range(256))
    rng.shuffle(p)
    return p + p


# Shared default permutation table (seeded, deterministic).
_PERM: List[int] = _build_permutation_table(1013)


def _fade(t: float) -> float:
    """Smoothstep fade curve: 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def _grad(hash_val: int, x: float, y: float, z: float) -> float:
    """Select gradient vector from hash (16 dirs) and dot with (x, y, z)."""
    h = hash_val & 15
    u = x if h < 8 else y
    v = y if h < 4 else (x if h == 12 or h == 14 else z)
    return ((u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v))


def noise3(x: float, y: float, z: float) -> float:
    """3D Perlin noise in range [-1, 1]."""
    floor_x = math.floor(x)
    floor_y = math.floor(y)
    floor_z = math.floor(z)
    xi = floor_x & 255
    yi = floor_y & 255
    zi = floor_z & 255
    xf = x - floor_x
    yf = y - floor_y
    zf = z - floor_z
    u = _fade(xf)
    v = _fade(yf)
    w = _fade(zf)

    aa = _PERM[_PERM[xi] + yi] + zi
    ab = _PERM[_PERM[xi] + yi + 1] + zi
    ba = _PERM[_PERM[xi + 1] + yi] + zi
    bb = _PERM[_PERM[xi + 1] + yi + 1] + zi

    d1 = _grad(_PERM[aa], xf, yf, zf)
    d2 = _grad(_PERM[ab], xf, yf, zf - 1)
    d3 = _grad(_PERM[ba], xf - 1, yf, zf)
    d4 = _grad(_PERM[bb], xf - 1, yf, zf - 1)
    d5 = _grad(_PERM[aa + 1], xf, yf - 1, zf)
    d6 = _grad(_PERM[ab + 1], xf, yf - 1, zf - 1)
    d7 = _grad(_PERM[ba + 1], xf - 1, yf - 1, zf)
    d8 = _grad(_PERM[bb + 1], xf - 1, yf - 1, zf - 1)

    x1 = _lerp(d1, d3, u)
    x2 = _lerp(d5, d7, u)
    y1 = _lerp(d2, d4, u)
    y2 = _lerp(d6, d8, u)
    z1 = _lerp(x1, y1, v)
    z2 = _lerp(x2, y2, v)
    return _lerp(z1, z2, w)


class PerlinShake:
    """Organic camera shake driven by layered Perlin noise.

    Multiple octaves of noise are summed (fractal Brownian motion) so the
    shake contains both micro-tremor and occasional larger jolts -- far more
    organic than a fixed sine wave and never mechanical.

    Parameters
    ----------
    intensity:
        Base shake amplitude in pixels (applied to dx/dy) and as a fraction
        for dzoom. Default 6 px pan/tilt, ~0.6% zoom.
    frequency:
        Base sampling frequency (Hz). Lower = slower drift.
    octaves:
        Number of fBm noise layers. More = richer texture.
    persistence:
        Amplitude decay per octave (0..1). Lower = smoother dominant layer.
    seed:
        Random seed for deterministic output.
    """

    def __init__(
        self,
        intensity: float = 6.0,
        frequency: float = 0.9,
        octaves: int = 3,
        persistence: float = 0.5,
        seed: int = 42,
    ):
        self.intensity = max(0.0, float(intensity))
        self.frequency = max(0.05, float(frequency))
        self.octaves = max(1, int(octaves))
        self.persistence = max(0.0, min(1.0, float(persistence)))
        self.seed = seed

    # ------------------------------------------------------------------
    # Fractal Brownian Motion (fBm)
    # ------------------------------------------------------------------
    def fbm(self, x: float, y: float, z: float) -> float:
        total = 0.0
        amplitude = 1.0
        frequency = self.frequency
        max_value = 0.0
        for octave in range(self.octaves):
            # Offset each octave by a prime-ish stride to decorrelate layers.
            stride = [37, 59, 83][octave % 3]
            ox = (x + octave * stride) * frequency
            oy = (y + octave * stride) * frequency
            oz = (z + octave * stride) * frequency
            total += noise3(ox, oy, oz) * amplitude
            max_value += amplitude
            amplitude *= self.persistence
            frequency *= 2.0
        return total / max(max_value, 1e-6)

    def offset(self, t: float) -> Tuple[float, float, float]:
        """Return (dx_px, dy_px, dzoom_frac) at time t (seconds)."""
        z = t * 0.5
        x = t * self.frequency + self.seed
        y = t * self.frequency * 0.7 + self.seed + 100.0

        n = self.fbm(x, y, z)                 # ~[-1, 1] combined pan/tilt
        pan_n = self.fbm(x + 33.0, y + 33.0, z + 11.0)
        tilt_n = self.fbm(x - 21.0, y + 47.0, z + 23.0)
        zoom_n = self.fbm(x + 50.0, y + 50.0, z + 50.0)

        dx = pan_n * self.intensity
        dy = tilt_n * self.intensity
        # dzoom is a gentle fractional scale factor (e.g. 1.006 at intensity 6).
        dzoom = 1.0 + (zoom_n * self.intensity / 100.0) * 0.008
        return dx, dy, dzoom

    # ------------------------------------------------------------------
    # Curve baking: sample offsets over a duration for batch application.
    # ------------------------------------------------------------------
    def bake_curve(
        self,
        duration: float,
        fps: float = 25.0,
    ) -> List[Tuple[float, float, float]]:
        """Return a list of (dx, dy, dzoom) sampled every 1/fps seconds."""
        n_frames = max(int(round(duration * fps)), 1)
        return [self.offset(i / fps) for i in range(n_frames)]
