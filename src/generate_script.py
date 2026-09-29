"""
Generates a short narration script + metadata (title, description, tags)
for a faceless YouTube Short, with hard length and SEO guardrails.
"""

from __future__ import annotations

import json
import os
import re

from groq import Groq

from _sanitize import sanitize_credential
from languages import DEFAULT_LANGUAGE, name_for_language
from title_utils import SHORTS_TAG, format_youtube_title

MODEL = "openai/gpt-oss-120b"

# ~150 wpm Piper/Kokoro speech. 50–55s => ~125–138 words. Hard-cap below 140.
MIN_NARRATION_WORDS = 120
MAX_NARRATION_WORDS = 135
MAX_SPOKEN_SECONDS = 55


def _language_instruction(language: str) -> str:
    if language == DEFAULT_LANGUAGE:
        return ""
    return (
        f"\n\nWrite your ENTIRE response -- title, description, narration, and tags -- "
        f"in {name_for_language(language)}. Only the JSON keys themselves stay in English."
    )


SYSTEM_PROMPT = """You write scripts for faceless YouTube Shorts and you are also an SEO \
copywriter for YouTube.

FORMAT: YouTube video package. Follow the requested duration and aspect ratio. \
Spoken length is 50-55 seconds for Shorts — never longer. \
That is a hard cap, not a target to pad toward.

VOICE: Write like a sharp, friendly expert explaining this to a smart friend over coffee -- \
conversational, a little witty, confident but never condescending. Anticipate the question \
the viewer is silently asking ("wait, why does that even work?", "ok but how do I actually \
do this?") and answer it directly in the next line, as if you read their mind. Avoid \
corporate jargon and buzzwords entirely (no "leverage", "synergy", "unlock your potential", \
"in today's fast-paced world") -- talk like a person, not a LinkedIn post.

HOOK (first 3 seconds, non-negotiable): the FIRST SENTENCE is both the verbal AND visual \
hook. It must be 6-10 words, speakable in under 3 seconds, and create high tension \
(curiosity, stakes, or a pattern interrupt) before any context. Open with ONE of:
- Bold/counterintuitive claim ("Most people get this backwards.")
- Direct question that implies a knowledge gap ("Why do the top 1% do this every morning?")
- Cold open fact/number ("93% of goals fail for one specific reason.")
- Pattern interrupt / myth-bust ("Everything you've heard about X is wrong.")
- Stakes-first ("If you don't fix this by 30, it gets 10x harder.")
Never open with a slow throat-clear like "Have you ever wondered..." or "Let's talk about."

STRUCTURE: 3-second hook -> 2-4 tight value/story beats that each earn the next line \
(cut anything that isn't essential) -> a clean ending that lands and ties back to the \
hook (a twist, a payoff, or a direct callback), never trailing off.

SENTENCE LENGTH: keep sentences SHORT (roughly 4-9 words each), one idea per sentence. \
Narration is auto-split into 1-3 word on-screen caption chunks, so short sentences \
caption cleanly.

LENGTH CAPS (hard limits, do not exceed):
- narration: 120-135 words total (50-55 seconds spoken at ~150 wpm). Count the words. \
  If you are over 135, cut beats until you are under.
- title: 60 to 70 characters, never over 70, INCLUDING " #shorts" for Shorts. Put the exact \
  target search term at the very beginning, capitalize the first letter, add a truthful \
  strong modifier or number, state the factual value or answer, and never use false \
  clickbait. The thumbnail should ask the visual question while the title gives the answer.
- description: first line is exactly "#shorts". Then 3-5 SEO sentences. Soft CTA to subscribe.
- tags: 8-12 items, include "shorts".

Clear simple sentences, no fluff, no stage directions, no headers - just spoken narration \
text. Return ONLY valid JSON, no markdown fences, no preamble."""


def _bullet_block(title: str, items: list[str] | None) -> str:
    """Renders a titled bullet list, or '' when there is nothing to render."""
    lines = [str(item).strip() for item in (items or []) if str(item).strip()]
    if not lines:
        return ""
    body = "\n".join(f"- {line}" for line in lines)
    return f"\n\n{title}\n{body}"


def _research_block(
    pain_points: list[str] | None,
    headlines: list[str] | None,
    strategy: list[str] | None,
    compliance: list[str] | None,
) -> str:
    """
    Turns this run's live internet research into prompt instructions: the real
    questions to answer (AEO targets), fresh headlines for phrasing context
    (never copied), the SEO/AEO content strategy, and finance guardrails.
    """
    return "".join([
        _bullet_block(
            "REAL QUESTIONS people are asking about this topic right now (answer the "
            "single best-fitting one outright in your first lines -- this is what AI "
            "answer engines and featured snippets quote):",
            pain_points,
        ),
        _bullet_block(
            "Fresh headlines touching this topic today. Use them ONLY to borrow common "
            "everyday phrasing. Do NOT copy, quote, or paraphrase any headline:",
            headlines,
        ),
        _bullet_block("CONTENT STRATEGY (SEO + AEO) you must follow:", strategy),
        _bullet_block(
            "ACCURACY AND SAFETY RULES that override everything else:", compliance
        ),
    ])


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text.strip()))


def _trim_narration(narration: str) -> str:
    """Hard-stop spoken length so Piper output stays inside 50–55s."""
    words = re.findall(r"\S+", narration.strip())
    if len(words) <= MAX_NARRATION_WORDS:
        return narration.strip()
    trimmed = words[:MAX_NARRATION_WORDS]
    text = " ".join(trimmed)
    if not re.search(r"[.!?]$", text):
        text = re.sub(r"[,:;]+$", "", text) + "."
    return text


def _primary_keyword(input_topic: str, seo_keywords: list[str] | None) -> str:
    topic = (input_topic or "").strip()
    if topic:
        return topic
    for candidate in seo_keywords or []:
        candidate = candidate.strip()
        if candidate:
            return candidate
    return ""


def _format_description(description: str) -> str:
    body = (description or "").strip()
    body = re.sub(r"^\s*#shorts\s*", "", body, flags=re.IGNORECASE)
    return f"{SHORTS_TAG}\n{body}".strip()


def _normalize_package(
    package: dict,
    base_topic: str,
    seo_keywords: list[str] | None,
    is_shorts: bool = True,
) -> dict:
    if not isinstance(package, dict):
        raise RuntimeError("Groq returned JSON that is not an object.")

    keyword = _primary_keyword(base_topic, seo_keywords)
    narration = _trim_narration(str(package.get("narration") or ""))
    if _word_count(narration) < 8:
        raise RuntimeError("Generated narration is empty or too short to use.")

    tags = package.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    tags = [str(t).strip() for t in tags if str(t).strip()]
    if is_shorts and "shorts" not in {t.lower() for t in tags}:
        tags = ["shorts", *tags]
    visual = package.get("visual_keywords") or []
    if isinstance(visual, str):
        visual = [visual]

    title = format_youtube_title(
        str(package.get("title") or base_topic),
        keyword,
        suffix=f" {SHORTS_TAG}" if is_shorts else "",
        default_modifier="Easy Guide",
    )
    return {
        "title": title,
        "description": _format_description(str(package.get("description") or "")),
        "tags": tags[:12],
        "narration": narration,
        "visual_keywords": [str(v).strip() for v in visual if str(v).strip()][:8],
        "thumbnail_hook": " ".join(title.split()[:3]),
    }


def generate_script(
    base_topic: str,
    seo_keywords: list[str] | None = None,
    api_key: str | None = None,
    language: str = DEFAULT_LANGUAGE,
    is_shorts: bool = True,
    pain_points: list[str] | None = None,
    headlines: list[str] | None = None,
    strategy: list[str] | None = None,
    compliance: list[str] | None = None,
):
    """
    Returns a dict:
    {
        "title": str,          # <=70 chars, keyword-first, factual
        "description": str,    # first line is #shorts for Shorts
        "tags": [str, ...],
        "narration": str,      # full voiceover script, <=135 words
        "visual_keywords": [str, ...]
    }

    pain_points / headlines / strategy / compliance come from this run's live
    internet research (see finance_research.py). They are all optional -- the
    generator works without them, just with less SEO/AEO context.
    """
    client = Groq(api_key=sanitize_credential(api_key or os.environ["GROQ_API_KEY"]))

    keyword_block = ""
    if seo_keywords:
        keyword_list = "\n".join(f"- {kw}" for kw in seo_keywords[:15])
        keyword_block = f"""

Here are REAL phrases people are currently searching for around this topic \
(from YouTube autocomplete and Google Trends related queries) -- use them ONLY as
phrasing hints, never as a replacement for the topic itself:
{keyword_list}

SEO requirements (all secondary to staying on-topic):
- The exact target topic MUST be the first words of the title. Keep the title at most
  70 characters including " #shorts", capitalize its first letter, and add a truthful
  number, bracket, or power modifier such as "easy," "best," or "ultimate."
- Make the title a factual answer or specific value; avoid "you won't believe,"
  "shocking," "forbidden secret," and other false clickbait.
- Use SEO phrases only where they are a natural, accurate fit for the narration.
- Do NOT keyword-stuff; the title and thumbnail must complement each other."""

    research_block = _research_block(pain_points, headlines, strategy, compliance)

    user_prompt = f"""Topic (the video MUST be specifically about this -- do not drift to a \
related but different subject, even if the SEO keywords below point elsewhere): {base_topic}{keyword_block}{research_block}

Create a faceless YouTube video package on this exact topic. Spoken narration MUST fit \
in 50-55 seconds for Shorts (120-135 words). The first sentence is the 3-second hook.

Respond with ONLY this JSON structure:

{{
  "title": "EXACT TOPIC first, factual value, strong modifier, max 70 chars including #shorts",
  "description": "#shorts as line 1, then 3-5 SEO sentences, subscribe CTA last",
  "tags": ["tag1", "tag2", "..."],
  "narration": "120-135 word spoken script. Sentence 1 is a 6-10 word high-tension hook.",
  "visual_keywords": ["keyword1", "keyword2", "keyword3", "..."]
}}{_language_instruction(language)}

visual_keywords are the SEARCH TERMS for a royalty-free STOCK VIDEO library and are never shown to a viewer, \
so always write them in English regardless of the response language above. Give 5-8 short, concrete, \
filmable visual subjects that MOVING footage can actually show -- this library is searched for video, not \
still images, so describe an action or a shot in motion ("piggy bank savings growing", "person budgeting \
on a phone", "stock market charts on screen", "coins stacking on a desk", "calculator and paperwork close \
up"), never an abstract noun like "money" or "success". Match the narration beat for beat. Because this \
library holds only non-copyright stock footage, NEVER use a real person's name, a company or brand name, \
a show/movie/game title, a team, or any copyrighted character as a keyword. Describe the generic scene \
instead (e.g. for a footballer use "soccer player scoring goal" or "stadium crowd cheering", never the \
player's or the club's name)."""

    response = None
    last_error = None
    for attempt in range(1, 4):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=4096,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            break
        except Exception as exc:
            last_error = exc
            is_json_error = (
                "json_validate_failed" in str(getattr(exc, "args", [""]))
                or "Failed to validate JSON" in str(exc)
            )
            if not is_json_error or attempt == 3:
                raise
            print(f"[generate_script] Groq JSON validation failed (attempt {attempt}/3), retrying with simplified prompt...")
            user_prompt = (
                f"Write a JSON object with keys title, description, tags, narration, visual_keywords "
                f"about: {base_topic}. Narration must be 120-135 words. "
                f"Answer one of the listed real questions directly, and obey the accuracy "
                f"and safety rules above."
                f"{research_block}"
                f"Return ONLY valid JSON, no markdown."
            )

    if response is None:
        raise RuntimeError(f"Groq script generation failed after 3 attempts: {last_error}")

    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError(
            "Groq returned an empty response (likely truncated before completing valid "
            "JSON). Try increasing max_tokens further if this recurs."
        )
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        package = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Groq returned invalid JSON for the script package.") from exc

    return _normalize_package(package, base_topic, seo_keywords, is_shorts=is_shorts)


if __name__ == "__main__":
    import sys

    topic = sys.argv[1] if len(sys.argv) > 1 else "space discoveries"
    result = generate_script(topic)
    print(json.dumps(result, indent=2))
