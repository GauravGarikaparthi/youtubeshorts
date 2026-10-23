"""
Visual pattern interrupters every 2.5–4 seconds.

Audience retention dies when a shot holds too long. This module breaks up
monotony by inserting a pattern interrupter on a randomized 2.5–4s cadence,
alternating between:

  1. A 10% digital zoom (push-in / pull-out pulse) on the current layer.
  2. A random B-roll overlay composited on top of the main layer.

The result is a constantly-shifting visual rhythm that resets the viewer's
attention -- a proven high-retention technique for Shorts/TikTok.

Implemented as a moviepy post-process: given a base CompositeVideoClip and a
list of B-roll asset paths, it returns a new clip with interrupters baked in
via per-segment `fl_image` transforms and overlay compositing.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np
from moviepy import (
    CompositeVideoClip,
    VideoClip,
    VideoFileClip,
)

LOG_PREFIX = "[pattern_interrupter]"


def log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", flush=True)


# Cadence bounds (seconds) between interrupters.
MIN_INTERVAL = 2.5
MAX_INTERVAL = 4.0

# Digital zoom magnitude (10% push-in).
ZOOM_SCALE = 1.10

# B-roll overlay opacity and size fraction of the canvas.
BROLL_OPACITY = 0.55
BROLL_SIZE_FRACTION = 0.45  # overlay covers ~45% of canvas width


def _random_interval(rng: random.Random) -> float:
    return rng.uniform(MIN_INTERVAL, MAX_INTERVAL)


def _ease_in_out(t: float) -> float:
    """Smoothstep ease for ramping the zoom in and out."""
    return t * t * (3.0 - 2.0 * t)


def _zoom_transform(get_frame, t: float, segment_start: float, segment_dur: float,
                    scale: float = ZOOM_SCALE):
    """
    Return a frame with a 10% digital zoom that eases in and out over the
    segment. Centered on the frame middle so the subject stays roughly put.
    """
    frame = get_frame(t)
    h, w = frame.shape[:2]

    # Position within the interrupter segment (0..1), eased.
    local = (t - segment_start) / max(segment_dur, 1e-6)
    local = max(0.0, min(1.0, local))
    ease = _ease_in_out(local)

    # Zoom factor: 1.0 -> scale -> 1.0 across the segment.
    factor = 1.0 + (scale - 1.0) * ease

    new_w = max(1, int(w / factor))
    new_h = max(1, int(h / factor))

    # Crop a centered window then scale back up to canvas size.
    x0 = max(0, (w - new_w) // 2)
    y0 = max(0, (h - new_h) // 2)
    crop = frame[y0:y0 + new_h, x0:x0 + new_w]

    # Resize crop back to canvas using PIL (moviepy frames are uint8 RGB).
    from PIL import Image
    img = Image.fromarray(crop).resize((w, h), Image.LANCZOS)
    return np.array(img)


def _apply_zoom_segment(clip: VideoClip, start: float, end: float,
                        scale: float = ZOOM_SCALE) -> VideoClip:
    """Apply a 10% digital zoom to `clip` between start and end seconds."""
    def make_fl(c, s, e, sc):
        def fl(get_frame, t):
            if s <= t <= e:
                return _zoom_transform(get_frame, t, s, e - s, sc)
            return get_frame(t)
        return fl

    return clip.fl(make_fl(clip, start, end, scale), apply_to=["mask"])


def _overlay_broll(base_clip: VideoClip, broll_path: str, start: float,
                   end: float, canvas_size: Tuple[int, int]) -> CompositeVideoClip:
    """
    Composite a random B-roll clip over `base_clip` between start and end.

    The B-roll is resized to BROLL_SIZE_FRACTION of canvas width, positioned
    in the upper-right corner (a classic picture-in-picture interrupter), and
    faded in/out over the segment edges.
    """
    broll = VideoFileClip(broll_path)
    # Choose a random sub-clip of the broll matching the segment duration.
    seg_dur = end - start
    max_start = max(0.0, broll.duration - seg_dur - 0.1)
    b_start = random.uniform(0.0, max_start) if max_start > 0 else 0.0
    broll_seg = broll.subclip(b_start, b_start + seg_dur)

    # Resize to fraction of canvas width, preserving aspect.
    target_w = int(canvas_size[0] * BROLL_SIZE_FRACTION)
    broll_seg = broll_seg.resize(width=target_w)

    # Position upper-right with a small margin.
    margin = int(canvas_size[0] * 0.03)
    x_pos = canvas_size[0] - broll_seg.w - margin
    y_pos = margin

    # Fade in/out at segment edges so the overlay appears/disappears smoothly.
    fade = min(0.15, seg_dur * 0.25)
    broll_seg = broll_seg.crossfadein(fade).crossfadeout(fade)
    broll_seg = broll_seg.set_opacity(BROLL_OPACITY)
    broll_seg = broll_seg.set_position((x_pos, y_pos))
    broll_seg = broll_seg.set_start(start).set_end(end)

    return CompositeVideoClip([base_clip, broll_seg], size=canvas_size)


def apply_interrupters(
    base_clip: VideoClip,
    broll_paths: List[str],
    canvas_size: Tuple[int, int],
    seed: int = 42,
    start_with_zoom: bool = True,
) -> VideoClip:
    """
    Apply alternating pattern interrupters across the whole clip.

    Parameters
    ----------
    base_clip:
        The main video clip (already assembled with cuts).
    broll_paths:
        List of file paths to B-roll assets. If empty, only zoom interrupters
        are used.
    canvas_size:
        (width, height) of the output canvas.
    seed:
        RNG seed for deterministic cadence and B-roll selection.
    start_with_zoom:
        Whether the first interrupter is a zoom (True) or B-roll (False).

    Returns
    -------
    A new CompositeVideoClip with interrupters applied.
    """
    rng = random.Random(seed)
    duration = base_clip.duration
    if duration <= 0:
        return base_clip

    # Build interrupter segments covering the full timeline.
    segments: List[Tuple[float, float, str]] = []  # (start, end, type)
    t = 0.0
    use_zoom = start_with_zoom
    while t < duration - 0.5:
        interval = _random_interval(rng)
        end = min(t + interval, duration)
        kind = "zoom" if use_zoom else "broll"
        segments.append((t, end, kind))
        t = end
        use_zoom = not use_zoom  # alternate

    log(f"Pattern interrupters: {len(segments)} segments "
        f"({sum(1 for _,_,k in segments if k=='zoom')} zooms, "
        f"{sum(1 for _,_,k in segments if k=='broll')} B-rolls).")

    # Apply zoom interrupters first via fl_image (per-segment), then layer
    # B-roll overlays on top via CompositeVideoClip.
    clip = base_clip
    for start, end, kind in segments:
        if kind == "zoom":
            clip = _apply_zoom_segment(clip, start, end, ZOOM_SCALE)

    # B-roll overlays are composited; build a stack then composite once.
    if broll_paths:
        layers = [clip]
        for start, end, kind in segments:
            if kind != "broll":
                continue
            broll_path = rng.choice(broll_paths)
            try:
                overlay = _overlay_broll(clip, broll_path, start, end, canvas_size)
                # _overlay_broll already composites with base; we instead want
                # to layer the broll on top of the accumulated `clip`. Rebuild:
                broll = VideoFileClip(broll_path)
                seg_dur = end - start
                max_start = max(0.0, broll.duration - seg_dur - 0.1)
                b_start = rng.uniform(0.0, max_start) if max_start > 0 else 0.0
                bseg = broll.subclip(b_start, b_start + seg_dur)
                target_w = int(canvas_size[0] * BROLL_SIZE_FRACTION)
                bseg = bseg.resize(width=target_w)
                margin = int(canvas_size[0] * 0.03)
                bseg = bseg.set_position((canvas_size[0] - bseg.w - margin, margin))
                fade = min(0.15, seg_dur * 0.25)
                bseg = bseg.crossfadein(fade).crossfadeout(fade)
                bseg = bseg.set_opacity(BROLL_OPACITY)
                bseg = bseg.set_start(start).set_end(end)
                layers.append(bseg)
            except Exception as exc:
                log(f"Skipped B-roll {broll_path}: {exc}")
        if len(layers) > 1:
            clip = CompositeVideoClip(layers, size=canvas_size)

    return clip
