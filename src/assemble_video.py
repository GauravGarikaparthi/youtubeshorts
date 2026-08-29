"""
Assembles the final Shorts-ready video from stock clips + voiceover
(+ optional background music) using ffmpeg.

Requires ffmpeg and ffprobe on PATH. No extra Python video libraries.

High-retention rendering pipeline (unchanged core):
- Center-crops every clip to 1080x1920 (9:16) when vertical, else 1920x1080
- Cuts every 1.5-2.5s with a 1.1x zoom/pan reset so long shots stay lively
- Burns 1-3 word KARAOKE captions (word-by-word highlight) centered on screen
- Wraps the last frames into the first for a seamless Shorts auto-replay loop
- Mixes narration over a quiet music bed

Fixes layered on top of that core:
1. AUDIO IS GUARANTEED -- the voiceover is validated BEFORE rendering (must
   exist, probe cleanly, contain a real audio stream, positive duration) and
   the FINISHED file is probed after muxing: a silent output RAISES instead
   of shipping a mute Short.
2. EXACTLY CENTERED CAPTIONS -- karaoke blocks are pinned at the true frame
   center (\\an5 at x=50%, y=50%), inside the Shorts UI safe zone.
3. TEMPLATE-DRIVEN LOOK -- transitions, pacing, caption style, music level,
   voice gain and color grade all come from a TemplateConfig (see
   viral_templates.py); passing no config preserves the previous tuned look.
4. PERFORMANCE -- hardware H.264 encoders (VideoToolbox/NVENC) are
   auto-detected and used everywhere they exist; ffmpeg runs through a
   hardened wrapper (suppressed noise, stderr tail on failure, retries).
"""

from __future__ import annotations

import math
import os
import re

from performance_optimizer import (
    has_audio_stream,
    log,
    media_duration,
    run_ffmpeg,
    video_encode_args,
)
from template_utils import find_music_track
from viral_captions import ass_font_size, ass_style_line
from viral_templates import DEFAULT_TEMPLATE, TemplateConfig
from viral_transitions import cycle_transitions

XFADE_MIN = 0.12
XFADE_MAX = 0.60
LOOP_XFADE = 0.40
TARGET_FPS = 25
MAX_SHORTS_SECONDS = 55.0

# Hard 9:16 Shorts canvas. Landscape sources are center-cropped, never stretched.
SHORTS_WIDTH = 1080
SHORTS_HEIGHT = 1920

FONT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts", "Montserrat-Bold.ttf"
)
FONTS_DIR = os.path.dirname(FONT_PATH)

# ASS colours are &HAABBGGRR (see viral_captions.hex_to_ass). Kept for the
# title drawtext (which is plain white-on-box, not style-driven).
MAX_CAPTION_WORDS = 3
TITLE_FONT_SIZE = 52

# ~10% linear (-20 dB) default music bed when no template overrides it.
DEFAULT_MUSIC_VOLUME = 0.10
VOICE_FADE_MS = 0.04


def _escape_ffmpeg_path(path: str) -> str:
    return path.replace("\\", "/").replace(":", "\\:").replace("'", r"\'")


def _write_caption_file(text: str, path: str) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text.replace("\n", " ").strip())
    return path


def _chunk_narration_for_captions(narration: str, max_words: int = MAX_CAPTION_WORDS) -> list[dict]:
    """1-3 word kinetic chunks, timed later from the real voiceover duration."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", narration.strip()) if s.strip()]
    chunks: list[dict] = []
    for sentence in sentences:
        words = [w for w in sentence.split() if w]
        for i in range(0, len(words), max_words):
            word_slice = words[i : i + max_words]
            chunk = " ".join(word_slice).strip(".,!?;:")
            if chunk:
                chunks.append({"text": chunk.upper(), "word_count": len(word_slice)})
    return chunks


def _ass_timestamp(seconds: float) -> str:
    seconds = max(seconds, 0)
    total_cs = int(round(seconds * 100))
    h, rem = divmod(total_cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", r"\\\\").replace("{", r"\{").replace("}", r"\}")


def _build_karaoke_ass(
    chunks: list[dict],
    caption_start: float,
    seconds_per_word: float,
    width: int,
    height: int,
    path: str,
    style_key: str,
) -> str:
    """
    Kinetic karaoke captions CENTERED on the frame (\\an5 at x=50%, y=50%),
    with margins keeping text clear of the right-side Shorts UI stack.
    The style line (colors/outline/shadow) comes from the active template's
    caption_style via viral_captions.
    """
    font_size = ass_font_size(height)
    pos_x = width // 2          # <-- exact horizontal center (the fix)
    pos_y = int(height * 0.50)  # vertical center of the middle third

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"{ass_style_line(style_key, font_size)}\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    lines = [header]
    cursor = caption_start
    for chunk in chunks:
        words = chunk["text"].split()
        if not words:
            continue
        start = cursor
        end = start + len(words) * seconds_per_word
        cursor = end
        kf = max(int(round(seconds_per_word * 100)), 8)
        karaoke = "".join(f"{{\\kf{kf}}}{_ass_escape(w)} " for w in words).strip()
        # \an5 + \pos pins the CENTER of the text block at (pos_x, pos_y),
        # so multi-line wraps stay centered as well.
        text = f"{{\\an5\\pos({pos_x},{pos_y})\\fsp2}}{karaoke}"
        lines.append(
            f"Dialogue: 0,{_ass_timestamp(start)},{_ass_timestamp(end)},Kinetic,,0,0,0,,{text}\n"
        )

    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(lines)
    return path


def _motion_filter(width: int, height: int, frames: int, fps: int, slot_index: int) -> str:
    """1.1x Ken-Burns zoom/pan so a long Pexels shot still 'resets' the eye."""
    frames = max(frames, 1)
    presets = [
        ("min(zoom+0.0009,1.10)", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),
        ("if(eq(on,0),1.10,max(zoom-0.0009,1.0))", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),
        ("1.10", f"iw/2-(iw/zoom/2)+min((iw-iw/zoom)/2,(on/{frames})*(iw*0.08))", "ih/2-(ih/zoom/2)"),
        ("1.10", f"iw/2-(iw/zoom/2)-min((iw-iw/zoom)/2,(on/{frames})*(iw*0.08))", "ih/2-(ih/zoom/2)"),
        ("1.08", "iw/2-(iw/zoom/2)", f"ih/2-(ih/zoom/2)+min((ih-ih/zoom)/2,(on/{frames})*(ih*0.06))"),
        ("1.08", "iw/2-(iw/zoom/2)", f"ih/2-(ih/zoom/2)-min((ih-ih/zoom)/2,(on/{frames})*(ih*0.06))"),
    ]
    zoom_expr, x_expr, y_expr = presets[slot_index % len(presets)]
    return (
        f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}'"
        f":d={frames}:s={width}x{height}:fps={fps}"
    )


def _center_crop_chain(width: int, height: int) -> str:
    """
    Scale with aspect preserved until the frame *covers* the canvas, then
    center-crop. Horizontal 16:9 stock becomes a vertical punch-in, never a
    stretch.
    """
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=increase:force_divisible_by=2,"
        f"crop={width}:{height}"
    )


def _render_segment(job: tuple) -> tuple:
    slot_index, clip, seg_path, per_clip_seconds, width, height = job
    frames = max(int(round(per_clip_seconds * TARGET_FPS)), 1)
    motion = _motion_filter(width, height, frames, TARGET_FPS, slot_index)
    cover = _center_crop_chain(width * 2, height * 2)

    try:
        source_dur = media_duration(clip)
    except (RuntimeError, FileNotFoundError):
        source_dur = 0.0

    input_args: list[str]
    if source_dur >= per_clip_seconds + 0.05:
        # Start each reuse of a clip at a DIFFERENT offset (golden-ratio
        # stride) so cycled clips don't repeat the same moment.
        spare = max(source_dur - per_clip_seconds, 0)
        start = (slot_index * 1.6180339887) % spare if spare > 0 else 0.0
        input_args = ["-ss", f"{start:.3f}", "-t", f"{per_clip_seconds:.3f}", "-i", clip]
    else:
        input_args = ["-stream_loop", "-1", "-t", f"{per_clip_seconds:.3f}", "-i", clip]

    run_ffmpeg(
        [
            *input_args,
            "-vf", f"{cover},{motion}",
            "-r", str(TARGET_FPS),
            "-an",
            *video_encode_args(),
            "-threads", "0",
            seg_path,
        ],
        desc=f"scene {slot_index} render",
    )
    return slot_index, seg_path


def _slot_durations(config: TemplateConfig) -> list[float]:
    """
    Scene-length pool: the template's explicit clip_durations when provided,
    otherwise a varied pool derived around clip_seconds. Every entry is
    clamped above the xfade duration so scenes never fully overlap.
    """
    if config.clip_durations:
        pool = list(config.clip_durations)
    else:
        base = max(config.clip_seconds, 0.8)
        pool = [round(base * f, 2) for f in (0.85, 1.1, 0.95, 1.25, 0.9, 1.15)]
    floor = config.transition_duration + 0.2
    return [max(d, floor) for d in pool] or [max(1.5, floor)]


def _plan_slots(voice_duration: float, config: TemplateConfig) -> list[float]:
    """Enough varied-length shots to cover the voiceover after xfade shrinkage."""
    pool = _slot_durations(config)
    td = config.transition_duration
    durations: list[float] = []
    covered = 0.0
    slot = 0
    while covered < voice_duration + td:
        dur = pool[slot % len(pool)]
        durations.append(dur)
        covered = dur if slot == 0 else covered + dur - td
        slot += 1
        if slot > 80:
            break
    return durations


def _concat_with_xfade(
    segment_paths: list[str],
    durations: list[float],
    out_path: str,
    trim: float,
    config: TemplateConfig,
) -> None:
    n_slots = len(segment_paths)
    if n_slots == 1:
        run_ffmpeg(
            [
                "-i", segment_paths[0], "-t", f"{trim:.3f}",
                *video_encode_args(), "-an", "-threads", "0",
                out_path,
            ],
            desc="single-scene passthrough",
        )
        return

    td = config.transition_duration
    transitions = cycle_transitions(config.transitions, n_slots - 1)

    input_args: list[str] = []
    for seg_path in segment_paths:
        input_args += ["-i", seg_path]

    filters: list[str] = []
    prev_label = "0:v"
    cumulative = durations[0]
    for slot in range(1, n_slots):
        offset = max(cumulative - td, 0)
        out_label = f"xf{slot}"
        filters.append(
            f"[{prev_label}][{slot}:v]xfade=transition={transitions[slot - 1]}:"
            f"duration={td:.3f}:offset={offset:.3f}[{out_label}]"
        )
        prev_label = out_label
        cumulative += durations[slot] - td

    run_ffmpeg(
        [
            *input_args,
            "-filter_complex", ";".join(filters),
            "-map", f"[{prev_label}]",
            *video_encode_args(), "-an", "-threads", "0",
            "-t", f"{trim:.3f}",
            out_path,
        ],
        desc="transition merge",
    )


def _make_seamless_loop(src: str, dst: str, duration: float) -> None:
    """Crossfade the tail into the head so Shorts auto-replay feels continuous."""
    fade = min(LOOP_XFADE, max(duration * 0.08, 0.15))
    offset = max(duration - fade, 0)
    run_ffmpeg(
        [
            "-i", src,
            "-filter_complex",
            (
                f"[0:v]split=2[main][head];"
                f"[head]trim=0:{fade:.3f},setpts=PTS-STARTPTS[headc];"
                f"[main][headc]xfade=transition=fade:duration={fade:.3f}:offset={offset:.3f}[vout]"
            ),
            "-map", "[vout]",
            "-t", f"{duration:.3f}",
            *video_encode_args(), "-an", "-threads", "0",
            dst,
        ],
        desc="seamless loop",
    )


def _linear_to_db(volume: float) -> float:
    """Linear gain -> dB for ffmpeg's volume filter (volume=0 -> mute)."""
    if volume <= 0:
        return -60.0
    return 20.0 * math.log10(volume)


def _validate_inputs(clip_paths: list[str], voiceover_path: str) -> float:
    """
    Fail-fast validation. THE silent-upload root cause lived here: nothing
    previously verified the voiceover before building a video around it.
    Returns the (capped) voiceover duration.
    """
    if not clip_paths:
        raise ValueError("No clips provided to assemble_video")
    missing = [p for p in clip_paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} clip file(s) missing on disk: {missing[:3]}"
            f"{'...' if len(missing) > 3 else ''}"
        )

    if not os.path.isfile(voiceover_path):
        raise FileNotFoundError(
            f"Voiceover file not found: '{voiceover_path}'. "
            "Refusing to assemble a Short without narration audio."
        )
    voice_duration = min(media_duration(voiceover_path), MAX_SHORTS_SECONDS)
    if not has_audio_stream(voiceover_path):
        raise RuntimeError(
            f"Voiceover '{voiceover_path}' contains NO audio stream. "
            "Regenerate it (generate_voiceover.py) before assembling."
        )
    if voice_duration < 0.5:
        raise RuntimeError(
            f"Voiceover duration ({voice_duration:.2f}s) is too short to build a video."
        )
    return voice_duration


def assemble_video(
    clip_paths: list[str],
    voiceover_path: str,
    title_text: str,
    out_path: str,
    work_dir: str = "work",
    vertical: bool = False,
    narration: str | None = None,
    music_path: str | None = None,
    template_config: TemplateConfig | None = None,
):
    """
    High-retention Shorts assembler driven by a TemplateConfig:

    - Center-crops every clip to 1080x1920 (9:16) when vertical, else 1920x1080
    - Cuts on the template's varied pacing with a 1.1x zoom/pan reset
    - Joins scenes with the template's cycled transition palette
    - Burns 1-3 word karaoke captions in the template's caption style,
      CENTERED on screen
    - Wraps the last frames into the first for a seamless Shorts loop
    - Mixes narration (template voice_gain) over music (template music_volume)

    template_config=None falls back to DEFAULT_TEMPLATE, which mirrors the
    previously-tuned neutral look -- existing callers change nothing.
    """
    config = template_config or DEFAULT_TEMPLATE
    os.makedirs(work_dir, exist_ok=True)

    # ---- 1. Validate everything BEFORE spending render time ----------------
    voice_duration = _validate_inputs(clip_paths, voiceover_path)
    log(f"Voiceover OK: {voice_duration:.2f}s, audio stream present.")

    if music_path and not os.path.isfile(music_path):
        log(f"Music path does not exist ({music_path}) -- continuing without music.")
        music_path = None
    if music_path is None:
        music_path = find_music_track()
    if config.music_volume <= 0:
        if music_path:
            log(f"Template '{config.name}' disables music -- ignoring {music_path}.")
        music_path = None

    width, height = (SHORTS_WIDTH, SHORTS_HEIGHT) if vertical else (1920, 1080)

    # ---- 2. Scene plan + PARALLEL renders -----------------------------------
    slot_durations = _plan_slots(voice_duration, config)
    n_slots = len(slot_durations)
    log(f"Plan: {n_slots} scenes, template='{config.name}', "
        f"transitions={config.transitions[:3]}{'...' if len(config.transitions) > 3 else ''}, "
        f"captions={config.caption_style}.")

    segment_jobs = [
        (
            slot,
            clip_paths[slot % len(clip_paths)],
            os.path.join(work_dir, f"seg_{slot:02d}.mp4"),
            slot_durations[slot],
            width,
            height,
        )
        for slot in range(n_slots)
    ]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    segment_paths: list[str | None] = [None] * n_slots
    workers = min(4, os.cpu_count() or 2, n_slots)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_render_segment, job) for job in segment_jobs]
        for future in as_completed(futures):
            slot_index, seg_path = future.result()
            segment_paths[slot_index] = seg_path

    # ---- 3. Merge with the template's varied transitions --------------------
    concat_video_path = os.path.join(work_dir, "concat_video.mp4")
    _concat_with_xfade(
        [p for p in segment_paths if p], slot_durations, concat_video_path,
        voice_duration, config,
    )

    # ---- 4. Seamless loop (best effort) --------------------------------------
    looped_video_path = os.path.join(work_dir, "looped_video.mp4")
    try:
        _make_seamless_loop(concat_video_path, looped_video_path, voice_duration)
        picture_path = looped_video_path
    except RuntimeError as exc:
        log(f"Seamless loop pass failed ({exc}) -- using linear cut.")
        picture_path = concat_video_path

    # ---- 5. Overlays: template color grade + centered title + karaoke captions
    vf_parts: list[str] = []
    if config.eq:
        vf_parts.append(config.eq)  # grade the PICTURE; captions burn on after

    if os.path.isfile(FONT_PATH):
        title_file = _write_caption_file(title_text, os.path.join(work_dir, "title.txt"))
        vf_parts.append(
            f"drawtext=fontfile='{_escape_ffmpeg_path(FONT_PATH)}':"
            f"textfile='{_escape_ffmpeg_path(title_file)}':"
            "fontcolor=white:fontsize={}:"
            "borderw=6:bordercolor=black@0.95:"
            "x=(w-text_w)/2:y=h*0.12:"
            "enable='between(t,0,3)'".format(TITLE_FONT_SIZE)
        )

    if narration:
        chunks = _chunk_narration_for_captions(narration)
        total_words = sum(c["word_count"] for c in chunks) or 1
        if chunks:
            seconds_per_word = voice_duration / total_words
            ass_path = os.path.join(work_dir, "captions.ass")
            _build_karaoke_ass(
                chunks, 0.0, seconds_per_word, width, height, ass_path,
                style_key=config.caption_style,
            )
            fonts_arg = f":fontsdir={_escape_ffmpeg_path(FONTS_DIR)}" if os.path.isdir(FONTS_DIR) else ""
            vf_parts.append(f"subtitles={_escape_ffmpeg_path(ass_path)}{fonts_arg}")

    # ---- 6. Audio graph: voice (gain + fades) over music (template level) ----
    fade = min(LOOP_XFADE, max(voice_duration * 0.08, 0.15))
    voice_gain = max(0.2, min(3.0, config.voice_gain))
    voice_af = (
        f"volume={voice_gain:.2f},"
        f"afade=t=in:d={VOICE_FADE_MS},afade=t=out:st={max(voice_duration - fade, 0):.3f}:d={fade:.3f}"
    )

    cmd: list[str] = ["-i", picture_path, "-i", voiceover_path]
    if music_path:
        cmd += ["-stream_loop", "-1", "-i", music_path]
        filter_complex = (
            f"[1:a]{voice_af},aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[voice];"
            f"[2:a]volume={_linear_to_db(config.music_volume):.1f}dB,"
            f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[music];"
            f"[voice][music]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
        )
    else:
        filter_complex = (
            f"[1:a]{voice_af},"
            f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[aout]"
        )

    # ---- 7. POST-MUX VERIFICATION (never ship a mute/broken file) ------------
    if not has_audio_stream(out_path):
        raise RuntimeError(
            f"Assembled output '{out_path}' has NO audio stream after muxing. "
            "This would upload as a silent Short -- failing the run instead. "
            "Check the voiceover file and the ffmpeg audio filters above."
        )
    out_duration = media_duration(out_path)
    if abs(out_duration - voice_duration) > 1.5:
        log(f"WARNING: output duration {out_duration:.2f}s differs from voiceover "
            f"{voice_duration:.2f}s by more than 1.5s.")

    mix_note = f"voice+music (vol {config.music_volume})" if music_path else "voice only"
    log(f"Assembly complete: {out_path} ({out_duration:.2f}s, audio: {mix_note}, "
        f"template '{config.name}', captions centered).")
    return out_path


if __name__ == "__main__":
    print("Run via main.py with real clip/voiceover paths.")
