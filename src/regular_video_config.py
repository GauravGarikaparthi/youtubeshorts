"""
Configuration for long-form regular videos.

Specifies 16:9 aspect ratio, 60fps, 4K resolution, high-quality audio/video
codecs, and all thumbnail/encoding defaults that differ from the Shorts pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Long-form video target specs (per user requirements)
LONGFORM_WIDTH = 3840
LONGFORM_HEIGHT = 2160
LONGFORM_ASPECT = "16:9"
LONGFORM_FPS = 60
LONGFORM_DURATION_TARGET = 600.0  # seconds (max practical for ~50 credits)

# Thumbnail specs for long-form (16:9, high-CTR)
THUMBNAIL_WIDTH = 1280
THUMBNAIL_HEIGHT_16_9 = 720

# High-quality codec configuration
VIDEO_CODEC = "h264"
VIDEO_CODEC_HIGH_PROFILE = "high"
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "320k"
AUDIO_SAMPLE_RATE = 48000
AUDIO_CHANNELS = 2
VIDEO_BITRATE = "50M"  # 4K high bitrate

# Google Flow free tier
GOOGLE_FLOW_FREE_CREDITS = 50
GOOGLE_FLOW_PROJECT_DIR = os.environ.get("GOOGLE_FLOW_PROJECT_DIR", "flow_projects")


@dataclass(frozen=True)
class LongformVideoConfig:
    """All encoding/thumbnail/render settings for regular long-form videos."""

    width: int = LONGFORM_WIDTH
    height: int = LONGFORM_HEIGHT
    aspect_ratio: str = LONGFORM_ASPECT
    fps: int = LONGFORM_FPS
    duration_target: float = LONGFORM_DURATION_TARGET
    thumbnail_width: int = THUMBNAIL_WIDTH
    thumbnail_height: int = THUMBNAIL_HEIGHT_16_9
    video_codec: str = VIDEO_CODEC
    video_codec_profile: str = VIDEO_CODEC_HIGH_PROFILE
    video_bitrate: str = VIDEO_BITRATE
    audio_codec: str = AUDIO_CODEC
    audio_bitrate: str = AUDIO_BITRATE
    audio_sample_rate: int = AUDIO_SAMPLE_RATE
    audio_channels: int = AUDIO_CHANNELS
    pixel_format: str = "yuv420p"
    crf: int = 18  # High quality (lower = better, 18-28 range)
    preset: str = "slow"
    project_dir: str = GOOGLE_FLOW_PROJECT_DIR

    def to_dict(self) -> dict:
        return asdict(self)
