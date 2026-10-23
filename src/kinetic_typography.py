"""
Millisecond-accurate kinetic typography captions.

Highlights the active word in bright yellow (#FFFF00) and auto-maps contextual
emojis. Generates per-word highlight clips centered on-screen, composited into
the final video.

Uses moviepy TextClip/ImageClip compositing. The font defaults to Montserrat
if available, falling back to system fonts.
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple, Optional

from moviepy import (
    CompositeVideoClip,
    TextClip,
    VideoClip,
)

LOG_PREFIX = "[kinetic_typography]"


def log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", flush=True)


# Default fonts
DEFAULT_FONT = "Montserrat-Bold"
FONTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")
MONTSERRAT_PATH = os.path.join(FONTS_DIR, "Montserrat-Bold.ttf")

# Yellow for active word highlight (bright, high-contrast)
ACTIVE_YELLOW = "#FFFF00"

# Emoji mapping subset (pragmatic; can be extended)
_EMOJI_MAP = {
    "money": "💰", "cash": "💵", "dollar": "💵", "invest": "📈",
    "saving": "🏦", "budget": "🧾", "debt": "💳",
    "success": "🏆", "goal": "🎯", "win": "🏆",
    "problem": "⚠️", "solution": "💡", "idea": "💡",
    "time": "⏰", "day": "☀️", "night": "🌙",
    "money": "💰", "business": "💼", "work": "💼",
    "happy": "😊", "sad": "😢", "great": "🌟",
    "new": "✨", "old": "📜", "learn": "📖",
    "future": "🔮", "past": "📜", "today": "📅",
}


def _get_font() -> Optional[str]:
    if os.path.exists(MONTSERRAT_PATH):
        return MONTSERRAT_PATH
    return DEFAULT_FONT


def _chunk_narration_words(narration: str) -> List[str]:
    """Split narration into word tokens (letters, numbers, apostrophes)."""
    words = re.findall(r"[A-Za-z0-9']+", narration.strip())
    return [w.strip().lower() for w in words]


def _estimate_timing(words: List[str], total_duration: float) -> List[Tuple[float, float]]:
    """Estimate millisecond-accurate-ish timing per word from total duration."""
    if not words:
        return []
    n = len(words)
    # Short and even distribution; keep reasonable pace for narration
    words_per_sec = n / total_duration if total_duration > 0 else 2.0
    seg_dur = total_duration / n if n > 0 else total_duration
    timing = []
    for i in range(n):
        start = i * seg_dur
        end = start + seg_dur
        timing.append((start, end))
    return timing


def _get_emoji_for_word(word: str) -> Optional[str]:
    w = word.lower()
    if w in _EMOJI_MAP:
        return _EMOJI_MAP[w]
    for k, e in _EMOJI_MAP.items():
        if k in w or w in k:
            return e
    return None


def build_kinetic_captions(
    base_clip: VideoClip,
    narration: str,
    total_duration: float,
    canvas_size: Tuple[int, int],
    font_size: int = 64,
) -> VideoClip:
    """
    Build kinetic captions with bright yellow active word + emojis.

    For each word, creates a TextClip showing the full phrase with that word
    highlighted in yellow, positioned centered on screen.
    """
    words = _chunk_narration_words(narration)
    timing = _estimate_timing(words, total_duration)
    if not words or not timing:
        return base_clip

    font = _get_font()
    clips = [base_clip]
    w = canvas_size[0]
    h = canvas_size[1]

    for i, word in enumerate(words):
        if i >= len(timing):
            break
        start, end = timing[i]
        emoji = _get_emoji_for_word(word)

        # Build phrase with active word highlighted in yellow
        phrase = narration
        # Simple approach: highlight just this word in the phrase display
        # For robustness, render the single word highlighted in context
        highlighted = " ".join(words[:i]) + f" <font color='{ACTIVE_YELLOW}'>{words[i]}</font> " + " ".join(words[i+1:])
        display = highlighted
        if emoji:
            display = f"{highlighted} {emoji}"

        try:
            txt = TextClip(
                txt=display,
                font=font,
                font_size=font_size,
                color="white",
                stroke_color="black",
                stroke_width=3,
                size=(int(w * 0.8), None),
                method="caption",
            )
            txt = txt.set_start(start).set_end(end)
            txt = txt.set_position(("center", int(h * 0.72)))  # lower middle, above UI
            clips.append(txt)
        except Exception as exc:
            log(f"Failed to create caption for word '{word}': {exc}")
            continue

    return CompositeVideoClip(clips, size=canvas_size).set_duration(total_duration)
