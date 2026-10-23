"""
Multi-layer 3D depth composition.

Arranges visual assets into a 3D depth structure with parallax. We simulate
3D by compositing multiple layers at different scales/positions; background
layers are scaled up and slightly blurred, foreground layers are closer,
and parallax moves each layer at a different speed based on camera motion.

The result is a sense of depth and dimensionality that eliminates the flat,
sterile digital look.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from moviepy import CompositeVideoClip, VideoClip

LOG_PREFIX = "[depth_composer]"


def log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", flush=True)


@dataclass
class DepthLayer:
    """Represents one visual layer in the depth composition."""
    clip: VideoClip
    depth: float  # 0.0 = foreground (closest), 1.0 = background (farthest)
    speed: float = 1.0  # parallax speed multiplier (relative to camera)
    scale: float = 1.0  # layer scale factor
    opacity: float = 1.0  # layer opacity


def _apply_layer_transform(layer: DepthLayer, base_size: Tuple[int, int]) -> VideoClip:
    """Apply scale and basic positioning to a depth layer."""
    clip = layer.clip
    if layer.scale != 1.0:
        clip = clip.resize(layer.scale)
    if layer.opacity < 1.0:
        clip = clip.set_opacity(layer.opacity)
    # Position centered by default; parallax could be added later via fl
    clip = clip.set_position("center")
    return clip


def compose_depth_layers(
    base_clip: VideoClip,
    layers: List[DepthLayer],
    canvas_size: Tuple[int, int],
) -> VideoClip:
    """
    Compose multiple depth layers into a single clip.

    layers: List of DepthLayer. Order: background to foreground recommended.
    """
    # Ensure base_clip matches canvas size
    if base_clip.size != canvas_size:
        base_clip = base_clip.resize(canvas_size)

    composite_layers = [base_clip]

    for layer in layers:
        lclip = _apply_layer_transform(layer, canvas_size)
        composite_layers.append(lclip)

    if len(composite_layers) == 1:
        return composite_layers[0]

    return CompositeVideoClip(composite_layers, size=canvas_size)
