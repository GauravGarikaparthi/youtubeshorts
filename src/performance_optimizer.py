"""
Shared performance + reliability layer for every ffmpeg-touching module.

What lives here (and why):
- run_ffmpeg(): one hardened subprocess wrapper -- captures stderr, prints the
  tail on failure (bare CalledProcessError hid the real reason for every
  failure historically), supports retries with backoff for flaky transients.
- Hardware-accelerated H.264 encoder detection (VideoToolbox on macOS,
  NVENC on NVIDIA boxes), detected ONCE per process and cached.
- ffprobe helpers: duration + audio-stream presence. has_audio_stream() is
  the backbone of the "Shorts posted without audio" fix -- we now PROBE the
  finished file and refuse to hand back a mute video.
- parallel_map(): bounded thread pool for fan-out ffmpeg jobs. Threads are
  the right tool here because the heavy work happens inside subprocesses
  (the GIL is never held during the encode itself).
"""

from __future__ import annotations

import ctypes.util
import re
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

LOG_PREFIX = "[perf]"


def log(message: str) -> None:
    """Timestamp-free prefixed print -- flush immediately so GitHub Actions
    shows progress lines live instead of buffering them until exit."""
    print(f"{LOG_PREFIX} {message}", flush=True)


def ffmpeg_bin() -> str:
    return os.environ.get("FFMPEG_BIN", "ffmpeg")


def ffprobe_bin() -> str:
    return os.environ.get("FFPROBE_BIN", "ffprobe")


def run_ffmpeg(args: list[str], desc: str = "ffmpeg step", retries: int = 1) -> None:
    """
    Runs ffmpeg with console noise suppressed (-loglevel error) but NEVER
    silently: on failure the last 4000 chars of stderr are logged so the
    actual cause is visible in CI logs on the first try.

    retries: extra attempts for transient failures (disk hiccup, OOM killer
    on a shared runner). Backoff grows per attempt.
    """
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", *args]
    last_error = ""
    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            return
        last_error = result.stderr.decode(errors="replace")[-4000:]
        log(f"'{desc}' failed (attempt {attempt}/{attempts}, exit {result.returncode}). "
            f"stderr tail:\n{last_error}")
        if attempt < attempts:
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"ffmpeg failed during '{desc}' after {attempts} attempt(s): {last_error}")


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def probe(path: str) -> dict:
    """Full ffprobe (format + streams) as a dict. Raises with a readable
    message if the file is missing or unreadable."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Media file not found: {path}")
    result = subprocess.run(
        [ffprobe_bin(), "-v", "error", "-show_format", "-show_streams", "-of", "json", path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for '{path}' (exit {result.returncode}): "
            f"{result.stderr[-2000:]}"
        )
    return json.loads(result.stdout)


def media_duration(path: str) -> float:
    """
    Duration in seconds. Falls back to the longest stream duration when the
    container-level duration is missing (some raw WAV/MPEG writes omit it).
    Raises if neither exists or the value is not positive -- callers rely on
    this being a REAL, usable duration.
    """
    info = probe(path)
    duration = float(info.get("format", {}).get("duration", 0) or 0)
    if duration <= 0:
        stream_durations = [
            float(s.get("duration", 0) or 0)
            for s in info.get("streams", [])
        ]
        duration = max(stream_durations, default=0.0)
    if duration <= 0:
        raise RuntimeError(f"Could not determine a positive duration for '{path}'. "
                           "The file is likely corrupt or empty.")
    return duration


def has_audio_stream(path: str) -> bool:
    """True iff the file contains at least one audio stream."""
    info = probe(path)
    return any(s.get("codec_type") == "audio" for s in info.get("streams", []))

def audio_levels(path: str) -> tuple[float, float]:
    """
    Returns (mean_volume_db, max_volume_db) via ffmpeg's volumedetect filter.
    Unlike has_audio_stream(), this confirms the audio is actually AUDIBLE,
    not just present -- a filter-graph bug can produce a technically-valid,
    all-silent stream that ffprobe alone would never catch.

    Runs ffmpeg directly (not through run_ffmpeg()) because volumedetect's
    stats print at "info" loglevel, which run_ffmpeg's "-loglevel error"
    would otherwise suppress.
    """
    result = subprocess.run(
        [ffmpeg_bin(), "-hide_banner", "-nostats", "-i", path, "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    mean_match = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", result.stderr)
    max_match = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", result.stderr)
    if not mean_match or not max_match:
        raise RuntimeError(f"Could not parse volumedetect output for '{path}'.")
    return float(mean_match.group(1)), float(max_match.group(1))

# ---------------------------------------------------------------------------
# Encoder selection (cached once per process)
# ---------------------------------------------------------------------------

_ENCODER_LIST_CACHE: str | None = None
_ENCODE_ARGS_CACHE: list[str] | None = None


def _nvenc_available() -> bool:
    """True iff the CUDA driver library needed by h264_nvenc is present."""
    if platform.system() != "Linux":
        return False
    return ctypes.util.find_library("cuda") is not None


def _available_encoders() -> str:
    global _ENCODER_LIST_CACHE
    if _ENCODER_LIST_CACHE is None:
        try:
            result = subprocess.run(
                [ffmpeg_bin(), "-hide_banner", "-encoders"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr[-2000:])
            _ENCODER_LIST_CACHE = result.stdout
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg binary not found on PATH. Install ffmpeg "
                "(brew install ffmpeg / apt-get install ffmpeg) and retry."
            )
    return _ENCODER_LIST_CACHE


def video_encode_args() -> list[str]:
    """
    Best available H.264 encode args, detected once and cached:
      - macOS + VideoToolbox  -> hardware encode (~2-4x faster than libx264)
      - NVIDIA + NVENC        -> hardware encode
      - otherwise             -> libx264 veryfast (best speed/size tradeoff)
    Every intermediate render in the pipeline goes through this so hardware
    acceleration is used uniformly wherever it exists.
    """
    global _ENCODE_ARGS_CACHE
    if _ENCODE_ARGS_CACHE is not None:
        return list(_ENCODE_ARGS_CACHE)

    encoders = _available_encoders()
    if platform.system() == "Darwin" and "h264_videotoolbox" in encoders:
        # Bitrate-controlled: VideoToolbox ignores x264-style CRF/preset knobs.
        chosen = ["-c:v", "h264_videotoolbox", "-b:v", "8M"]
        log("Using hardware encoder: h264_videotoolbox")
    elif "h264_nvenc" in encoders and _nvenc_available():
        chosen = ["-c:v", "h264_nvenc", "-b:v", "8M"]
        log("Using hardware encoder: h264_nvenc")
    else:
        chosen = ["-c:v", "libx264", "-preset", "veryfast"]
        log("Using software encoder: libx264 (veryfast)")

    chosen += ["-pix_fmt", "yuv420p"]  # maximum player/phone compatibility
    _ENCODE_ARGS_CACHE = chosen
    return list(chosen)


# ---------------------------------------------------------------------------
# Parallel fan-out
# ---------------------------------------------------------------------------

def _default_workers() -> int:
    """
    Conservative default: each worker owns a full ffmpeg process that itself
    spawns multiple encode threads, so half the cores (max 4) keeps total
    load sane on 2-core CI runners and 10-core dev machines alike.
    Override with FFMPEG_PARALLEL_WORKERS.
    """
    override = os.environ.get("FFMPEG_PARALLEL_WORKERS", "").strip()
    if override.isdigit() and int(override) > 0:
        return int(override)
    cpus = os.cpu_count() or 2
    return max(1, min(4, cpus // 2))


def parallel_map(
    fn,
    items: list,
    desc: str = "task",
    max_workers: int | None = None,
    on_error: str = "raise",
) -> list:
    """
    Maps fn over items on a thread pool, PRESERVING input order in the result.

    on_error:
      - "raise": first exception propagates (after in-flight jobs finish) --
        correct for pipeline stages where every item is required downstream.
      - "skip":  log the failure, put None in that slot, keep going -- correct
        for best-effort stages like per-keyword asset downloads.
    """
    if not items:
        return []

    workers = max(1, min(len(items), max_workers or _default_workers()))
    results: list = [None] * len(items)
    failures: list[tuple[int, Exception]] = []

    def _work(index: int, item):
        try:
            return index, fn(item), None
        except Exception as exc:  # noqa: BLE001 - boundary must not leak types
            return index, None, exc

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_work, i, item) for i, item in enumerate(items)]
        for future in as_completed(futures):
            index, value, error = future.result()
            if error is not None:
                failures.append((index, error))
                if on_error == "raise":
                    raise error
                log(f"{desc}[{index}] failed, skipping: {error}")
            else:
                results[index] = value

    elapsed = time.monotonic() - started
    ok_count = sum(1 for r in results if r is not None)
    log(f"{desc}: {ok_count}/{len(items)} succeeded in {elapsed:.1f}s "
        f"(workers={workers})")
    if failures and on_error != "raise":
        log(f"{desc}: {len(failures)} item(s) skipped due to errors")
    return results


if __name__ == "__main__":
    # Self-check: encoder detection + probe round-trip on whatever exists.
    print("encode args:", video_encode_args())
    print(f"platform={platform.system()} python={sys.version.split()[0]}")
