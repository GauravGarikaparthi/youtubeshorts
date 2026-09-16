"""
Optimized prompt builder for regular long-form videos.

Builds a comprehensive prompt that incorporates:
  - The "WTF" formula for intros (What, Tease, Forge)
  - First 10-second retention mastery (frequent visual/camera changes)
  - Jump cuts (remove dead air and pauses)
  - Fast-paced, high-retention storytelling ("raw, human perspective")
  - Psychology of Information Packaging:
      * Curiosity Gap (shocking contrast / untold secret)
      * De-Oudience Intro (3-second shocking fact, no credentials)
      * Visual Subtraction (minimalist, dynamic, vibrant)
  - High-Retention Scripting & Visual Architecture:
      * Show, Don't Tell (visual layer change every ~5s)
      * Humanize the Framework (raw, over-the-shoulder, screen-share feel)
      * Micro-Hooks (cliffhanger every ~45s)
      * Remove Audio Dead Air (aggressive cut of pauses/fillers)
  - Engineering Behavioral Triggers:
      * The "Correction" Loop (myth-busting for comment engagement)
      * Spoken Keyword Sync (primary keywords in first 10s)
      * End-Screen Transition ("If you want to see the real magic,
        stay tuned for what's coming next...")

The prompt is designed to drive Google Flow to produce an animated
illustration-type explainer with storytelling that visually maps out and
explains the core concepts.
"""

from __future__ import annotations

from string import Template
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Master system prompt — the full brain of the regular-video script engine
# ---------------------------------------------------------------------------

REGULAR_VIDEO_SYSTEM_PROMPT = """\
You are an elite YouTube long-form video strategist and storyteller. Your job is
to take a single topic and produce a complete, production-ready video brief —
a JSON object — that a video-generation engine (Google Flow) will turn into an
animated illustration explainer. The video must feel like a raw, human
perspective: high-energy, fast-paced, and engineered for maximum retention and
virality.

Below is every rule you must apply. Read them all before producing output.

================================================================================
1. THE "WTF" FORMULA (Intro — first 10 seconds)
================================================================================
  W — What the video is about (1 short declarative sentence, 0-3s)
  T — Tease the core value (the ONE thing the viewer will learn, 3-7s)
  F — Forge on: dive straight into content (7-10s, no more throat-clearing)

  The intro must NOT lead with credentials, a slow setup, or "let me tell you
  about…". It must create immediate tension and hook the viewer before they can
  scroll away.

================================================================================
2. FIRST 10 SECONDS MASTERY
================================================================================
  Every 1-2 seconds in the first 10s, the visual layer must change:
  different camera angle, different on-screen element, different kinetic
  typography burst, or a different data-graphic reveal. The goal is to stop
  early drop-offs by never letting the eye settle.

================================================================================
3. JUMP CUTS & DEAD-AIR REMOVAL
================================================================================
  Structure the script as a series of punchy, rapid-fire beats. Each beat is
  5-12 seconds. Between beats, the editor uses jump cuts to remove ALL pauses,
  ums, ahs, and dead air. The result is an unrelenting, energetic pace with
  zero breathing room for attention to wandering.

================================================================================
4. FAST-PACED, RAW HUMAN PERSPECTIVE
================================================================================
  Strip away the academic-lecture format. Write and visualize as if the
  creator is speaking directly to the camera with raw, unfiltered energy.
  Use slight imperfections: over-the-shoulder screen-share glimpses, hand-
  drawn diagrams, whiteboard scribbles, slightly unpolished graphics that feel
  like a real person's actual screen. Every visual should feel human-made,
  not corporate-stock.

================================================================================
5. THE PSYCHOLOGY OF INFORMATION PACKAGING
================================================================================

5a. The Curiosity Gap
  Frame the title around a shocking contrast or an untold secret, e.g.
  "The AI Lie Everyone Believes" instead of "How AI Works". The title must
  make the viewer feel they are about to learn something they weren't
  supposed to know.

5b. De-Oudience Intro
  Do NOT open with "I'm an expert with X years of experience". Open with a
  3-second statement of the most shocking fact or claim in the entire video.
  Lock in attention before the viewer can decide they don't care.

5c. Visual Subtraction
  Every visual element must fight for its place on screen. Remove everything
  that doesn't serve the single core message of that beat. Thumbnails (and
  on-screen text) must be minimalist: one clear focal point on the left, a
  maximum of three bold words that trigger curiosity. Vibrant colors, high
  contrast, no clutter.

================================================================================
6. HIGH-RETENTION SCRIPTING & VISUAL ARCHITECTURE
================================================================================

6a. Show, Don't Tell (every 5 seconds)
  For every 5 seconds of spoken information, the visual layer MUST change —
  via dynamic B-roll, kinetic typography, or a relevant data graphic.
  Never let an image linger while narration changes topic.

6b. Humanize the Framework
  Replace generic stock footage with raw, over-the-shoulder software B-roll
  or slightly unpolished hand-drawn graphics. The video should look like a
  real person's screen share, not a corporate explainer.

6c. Micro-Hooks (every 45 seconds)
  Seed a mini-cliffhanger at the 45-second mark, the 90-second mark, the
  135-second mark, etc. Each micro-hook is a single question or bold
  statement that primes the viewer to keep watching for what comes next.

6d. Remove Audio Dead Air
  The voiceover script must have NO pauses, fillers, or breathers. Every
  second is packed with information or a hook. Sentence lengths vary but
  the average cadence is 150-180 wpm of high-density content.

================================================================================
7. ENGINEERING BEHAVIORAL TRIGGERS (THE ALGORITHM RULES)
================================================================================

7a. The "Correction" Loop
  Early in the video, state a widely-accepted myth or misconception. When
  viewers rush to the comments to debate or agree, this boosts initial
  engagement velocity (comments + rewatch-to-prove-the-point behaviour).

7b. Spoken Keyword Sync
  The primary concept keywords MUST be spoken clearly within the first 10
  seconds so that YouTube's automatic captioning categorizes the video into
  the correct niche immediately. Keywords should appear naturally in the
  opening sentence.

7c. End-Screen Transition
  Instead of "Thanks for watching", transition seamlessly into the next
  concept: "If you want to see the real magic, stay tuned for what's coming
  next..." — this extends watch time by conditioning the viewer to not leave.

================================================================================
8. STRUCTURE & OUTPUT SCHEMA
================================================================================
  Total video length: target 5-10 minutes (the longest possible that uses all
  50 free daily credits in Google Flow free tier).

  Narrative arc (mandatory):
    - Hook (0-3s): shock fact or counter-intuitive claim (rule 5b)
    - WTF intro (3-10s): What → Tease → Forge (rule 1)
    - Myth-busting setup (10-20s): state the common misconception (rule 7a)
    - Concept 1 (20s-~2min): explain core concept A with visuals every 5s
      * Micro-hook at 45s mark
    - Concept 2 (~2-4min): explain core concept B with show-don't-tell
      * Micro-hook at 45s mark
    - Concept 3 (~4-6min): advanced angle / deeper insight
      * Micro-hook at 45s mark
    - Payoff + callback (~6-8min): resolve the opening hook, callback to
      the shocking fact from the intro
    - End-screen transition (~8-10min): "If you want to see the real magic, stay tuned for what's coming next..." (rule 7c)

  For each visual beat, provide:
    - A precise visual directive (B-roll, data graphic, kinetic text, or
      screen-share frame)
    - The exact spoken narration for that beat
    - The on-screen text/kinetic typography to display
    - A transition note (jump cut, quick zoom, wipe, etc.)

  Metadata (separate from the script):
    - A 16:9, high-contrast, vibrant thumbnail description with maximum 3
      bold words on the left side (rule 5c)
    - A 60-80 char SEO/GEO-optimized title with the primary keyword first
      and a curiosity-gap hook (rule 5a)
    - 15-20 trending tags mixing broad and long-tail keywords (rule for SEO)
    - A 200-400 word description with keyword-rich first 2 lines, chapter
      timestamps, and an explicit subscribe CTA (rule for SEO/GEO/AEO)
    - An engaging 1-2 sentence caption for the video comment section (AEO)

  Return ONLY valid JSON, no markdown fences, no commentary.
"""


@dataclass
class VisualBeat:
    """A single visual-speech beat in the video script."""

    beat_num: int
    start_second: float
    end_second: float
    narration: str
    visual_directive: str
    onscreen_text: str
    transition: str
    hook_marker: bool = False  # True if this beat contains a micro-hook
    myth_marker: bool = False  # True if this beat states a myth to bust
    correction_marker: bool = False  # True if this beat corrects the myth


@dataclass
class LongformScriptPackage:
    """Complete output for a regular long-form video."""

    topic: str
    title: str
    description: str
    tags: List[str]
    thumbnail_prompt: str
    caption: str  # for the comment section (AEO)
    estimated_duration_seconds: float
    intro_wtf: dict  # {what, tease, forge}
    beats: List[VisualBeat]
    raw_narration: str  # concatenated narration for TTS / Google Flow
    spoken_keywords: List[str]

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "title": self.title,
            "description": self.description,
            "tags": self.tags,
            "thumbnail_prompt": self.thumbnail_prompt,
            "caption": self.caption,
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "intro_wtf": self.intro_wtf,
            "beats": [asdict(b) for b in self.beats],
            "raw_narration": self.raw_narration,
            "spoken_keywords": self.spoken_keywords,
        }


# Template for the user prompt that feeds the LLM (or Google Flow prompt)
REGULAR_VIDEO_USER_PROMPT_TEMPLATE = Template("""\
Topic: $topic

You are scripting an animated illustration explainer on "$topic". This is a
long-form video (target 5-10 minutes) that must use every retention and
virality technique in your arsenal.

Apply ALL of the following rules without exception:

1. WTF FORMULA INTRO: W (What it is about) → T (Tease the core value) → F (Forge into content)
2. FIRST 10 SECONDS: change the visual layer every 1-2 seconds
3. JUMP CUTS: remove ALL dead air, pauses, fillers — zero breathing room
4. RAW HUMAN PERSPECTIVE: over-the-shoulder screen-share feel, hand-drawn graphics,
   unpolished imperfections — never corporate stock
5. CURIOUSITY GAP TITLE: shocking contrast or untold secret (rule 5a)
6. DE-OUDIENCE INTRO: open with the most shocking 3-second fact, NOT credentials (rule 5b)
7. VISUAL SUBTRACTION: minimalist, one focal point, max 3 bold words (rule 5c)
8. SHOW DON'T TELL: visual layer changes every 5 seconds of narration (rule 6a)
9. HUMANIZE: raw B-roll, slightly unpolished graphics (rule 6b)
10. MICRO-HOOKS: cliffhanger every 45 seconds (rule 6c)
11. NO DEAD AIR: zero pauses/fillers in voiceover (rule 6d)
12. CORRECTION LOOP: state a widely-believed myth early (rule 7a)
13. SPOKEN KEYWORD SYNC: primary keywords spoken in first 10 seconds (rule 7b)
14. END-SCREEN TRANSITION: "If you want to see the real magic, stay tuned for what's coming next..." (rule 7c)

Output a JSON object with these exact keys:
{
  "title": "string — 60-80 chars, keyword first, curiosity-gap hook (rule 5a)",
  "description": "string — 200-400 words, keyword-rich first 2 lines, chapter timestamps, subscribe CTA",
  "tags": ["15-20 trending tags, mix of broad and long-tail"],
  "thumbnail_prompt": "string — 16:9, high-contrast, vibrant, one focal point on left, max 3 bold words",
  "caption": "string — 1-2 sentences for the comment section (AEO engagement bait)",
  "intro_wtf": {
    "what": "string — what the video is about (0-3s, ~10 words)",
    "tease": "string — the core value teased (3-7s, ~15 words)",
    "forge": "string — dive straight into content (7-10s, ~20 words)"
  },
  "estimated_duration_seconds": 420.0,
  "spoken_keywords": ["primary keyword 1", "secondary keyword 2", "keyword 3"],
  "beats": [
    {
      "beat_num": 1,
      "start_second": 0.0,
      "end_second": 8.0,
      "narration": "string — spoken text for this beat",
      "visual_directive": "string — precise visual: B-roll / data graphic / kinetic text / screen-share frame",
      "onscreen_text": "string — kinetic typography to display",
      "transition": "string — jump cut / quick zoom / wipe / flash / etc.",
      "hook_marker": false,
      "myth_marker": true,
      "correction_marker": false
    }
  ]
}

The intro must open with the most shocking fact about $topic, then immediately
apply the WTF formula. The entire voiceover must be jump-cut tight with zero
dead air. Every beat must change the visual layer. Micro-hooks every 45
seconds. End with the stay-tuned transition.

Produce the complete script now.\
""")


def build_regular_video_prompt(topic: str, **kwargs) -> str:
    """Build the full user prompt for regular video script generation."""
    return REGULAR_VIDEO_USER_PROMPT_TEMPLATE.substitute(
        topic=topic,
        **{k: v for k, v in kwargs.items() if v is not None}
    )


def build_optimized_flow_prompt(topic: str, keywords: list[str] | None = None) -> str:
    """
    Build a single, self-contained, optimized prompt string for Google Flow.

    This combines the system-prompt rules, the topic, the SEO keyword context,
    and the structural directives into one prompt that Google Flow can ingest
    to generate the longest possible animated illustration explainer.
    """
    keyword_ctx = ""
    if keywords:
        keyword_list = "\n".join(f"  - {kw}" for kw in keywords[:10])
        keyword_ctx = f"""

SEO KEYWORD CONTEXT (use these for spoken keyword sync + tag inspiration):
{keyword_list}"""

    # The 50-credit maximization directive
    credit_directive = (
        "\n\nCRITICAL: This video must be as long as possible — target 8-10 minutes "
        "— to consume all 50 free daily credits in the Google Flow free tier. "
        "Maximize scene diversity and visual complexity to justify maximum credit usage."
    )

    # Strip markdown from system prompt for the combined Flow prompt
    combined = (
        f"[SYSTEM]\n{REGULAR_VIDEO_SYSTEM_PROMPT.strip()}"
        f"\n\n[USER]\n{build_regular_video_prompt(topic)}"
        f"{keyword_ctx}"
        f"{credit_directive}"
    )
    return combined
