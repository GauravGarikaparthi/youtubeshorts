"""
High-retention moviepy+ffmpeg assembler.

Combines all the cinematic effects into a single, optimized pipeline:
- J-cuts and L-cuts (0.3s audio overlaps)
- Visual pattern interrupters every 2.5-4s (alternating 10% zoom + B-roll)
- Millisecond-accurate kinetic typography with yellow word highlight + emojis
- Organic Perlin noise camera shake
- Multi-layer 3D depth composition
- Seamless transitions and high-retention pacing

Uses moviepy for composition logic, ffmpeg for final encoding.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    VideoClip,
    VideoFileClip,
    concatenate_videoclips,
)

from perlin_shake import PerlinShake
from jcut_lcut import plan_cuts, apply_jl_cuts_to_audio, apply_to_video
from pattern_interrupter import apply_interrupters
from kinetic_typography import build_kinetic_captions
from depth_composer import compose_depth_layers, DepthLayer

LOG_PREFIX = "[high_retention_assembler]"


def log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", flush=True)


@dataclass
class AssemblyConfig:
    """Configuration for high-retention assembly."""
    width: int = 1080
    height: int = 1920
    fps: float = 25.0
    overlap: float = 0.3
    enable_shake: bool = True
    enable_pattern_interrupters: bool = True
    enable_depth: bool = True
    enable_kinetic_captions: bool = True
    shake_intensity: float = 6.0
    music_volume: float = 0.12
    voice_volume: float = 1.0


def _apply_perlin_shake(
    clip: VideoClip,
    duration: float,
    config: AssemblyConfig,
) -> VideoClip:
    """Apply organic Perlin noise camera shake."""
    if not config.enable_shake:
        return clip

    shake = PerlinShake(intensity=config.shake_intensity, frequency=0.9, octaves=3)
    curve = shake.bake_curve(duration, fps=config.fps)

    def shake_frame(get_frame, t):
        frame = get_frame(t)
        # Find nearest frame in curve
        idx = int(t * config.fps) % len(curve) if curve else 0
        dx, dy, dzoom = curve[idx] if idx < len(curve) else (0, 0, 1.0)
        h, w = frame.shape[:2]
        # Apply zoom and pan
        if abs(dzoom - 1.0) > 0.001:
            new_w = int(w / dzoom)
            new_h = int(h / dzoom)
            x0 = max(0, (w - new_w) // 2 + int(dx))
            y0 = max(0, (h - new_h) // 2 + int(dy))
            from PIL import Image
            crop = frame[y0:y0 + new_h, x0:x0 + new_w]
            img = Image.fromarray(crop).resize((w, h), Image.LANCZOS)
            return np.array(img)
        return frame

    return clip.fl(shake_frame)


def assemble_high_retention(
    clip_paths: List[str],
    voiceover_path: str,
    narration: str,
    output_path: str,
    music_path: Optional[str] = None,
    broll_paths: Optional[List[str]] = None,
    config: Optional[AssemblyConfig] = None,
    work_dir: str = "work",
    vertical: bool = True,
) -> str:
    """
    Assemble a high-retention video with all cinematic effects.

    Args:
        clip_paths: List of stock video clip paths
        voiceover_path: Path to voiceover audio
        narration: Narration text
        output_path: Output video path
        music_path: Optional background music
        broll_paths: Optional B-roll clip paths for overlays
        config: Assembly configuration
        work_dir: Working directory
        vertical: True for 9:16, False for 16:9

    Returns:
        Path to output video
    """
    os.makedirs(work_dir, exist_ok=True)
    cfg = config or AssemblyConfig()
    if not vertical:
        cfg.width, cfg.height = 1920, 1080

    canvas_size = (cfg.width, cfg.height)

    # Load clips and audio
    clips = []
    for p in clip_paths:
        try:
            c = VideoFileClip(p)
            # Center crop to canvas
            c = c.resize(height=canvas_size[1])
            if c.w < canvas_size[0]:
                c = c.resize(width=canvas_size[0])
            # Crop center
            x = (c.w - canvas_size[0]) // 2
            y = (c.h - canvas_size[1]) // 2
            c = c.crop(x=x, y=y, width=canvas_size[0], height=canvas_size[1])
            clips.append(c)
        except Exception as exc:
            log(f"Failed to load clip {p}: {exc}")

    if not clips:
        raise ValueError("No valid clips loaded")

    # Load voiceover
    voiceover = AudioFileClip(voiceover_path)

    # Plan J/L cuts
    clip_durs = [min(c.duration, 6.0) for c in clips]
    plan = plan_cuts(clip_durs, voiceover.duration, overlap=cfg.overlap)

    # Apply to video
    video = apply_to_video(clips, plan)

    # Apply Perlin shake
    video = _apply_perlin_shake(video, plan.total_duration, cfg)

    # Apply pattern interrupters
    if cfg.enable_pattern_interrupters and broll_paths:
        video = apply_interrupters(video, broll_paths, canvas_size, seed=42)

    # Apply depth composition (optional)
    if cfg.enable_depth:
        video = compose_depth_layers(video, [], canvas_size)

    # Apply kinetic captions
    if cfg.enable_kinetic_captions and narration:
        video = build_kinetic_captions(
            video,
            narration,
            plan.total_duration,
            canvas_size,
            font_size=64,
        )

    # Apply J/L cuts to audio
    audio = apply_jl_cuts_to_audio(voiceover, plan)

    # Mix with music
    if music_path:
        try:
            music = AudioFileClip(music_path).volumex(cfg.music_volume)
            # Loop music to match duration
            if music.duration < plan.total_duration:
                music = music.fx(lambda x: x.audio_loop(duration=plan.total_duration))
            else:
                music = music.subclip(0, plan.total_duration)
            audio = CompositeAudioClip([audio.volumex(cfg.voice_volume), music])
        except Exception as exc:
            log(f"Failed to mix music: {exc}")
            audio = audio.volumex(cfg.voice_volume)
    else:
        audio = audio.volumex(cfg.voice_volume)

    # Composite video + audio
    final = video.set_audio(audio)
    final = final.set_duration(plan.total_duration)
    final = final.set_fps(cfg.fps)

    # Write using ffmpeg
    log(f"Writing final video to {output_path}")
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=cfg.fps,
        threads=4,
        preset="veryfast",
        bitrate="8M",
    )

    # Close clips
    for c in clips:
        try:
            c.close()
        except Exception:
            pass
    try:
        voiceover.close()
    except Exception:
        pass

    return output_path
