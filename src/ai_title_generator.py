"""
Backend hook: funnel video transcript into AI endpoint.

Takes the video transcript/narration, sends it to an AI endpoint (Groq) to
auto-generate:
- Three (3) high-CTR titles
- Structured description chapters

The AI is instructed to maximize YouTube CTR and retention with factual,
non-clickbait titles, proper chapterization, and SEO-friendly output.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

from groq import Groq

from _sanitize import sanitize_credential

LOG_PREFIX = "[ai_title_generator]"


def log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", flush=True)


SYSTEM_PROMPT = """You are an expert YouTube Shorts SEO, CTR optimization, and audience retention specialist. You write titles and descriptions optimized for high CTR and watch time.

Your task: given a video transcript/narration, generate exactly 3 high-CTR titles and structured description chapters.

RULES:
1. Generate EXACTLY 3 distinct title suggestions. Each title must:
   - Be under 60 characters (leave room for #Shorts)
   - Have the main topic/keyword at the front when possible
   - Be factual and specific (avoid false clickbait like "you won't believe", "shocking", "insane")
   - Focus on curiosity + value (answer a question, promise a clear takeaway)
   - Use strong, truthful modifiers (e.g., "Easy", "Fast", "Proven", "Simple", "Best")
   - Be distinct from each other (different angle/phrasing)

2. Generate structured description chapters with timestamps. For Shorts (50-60s):
   - Create 4-6 logical chapters (e.g., Hook, Problem, Explanation, Example, Key Takeaway, Conclusion)
   - Use MM:SS or 00:00 format
   - Chapters should be short, descriptive phrases

3. Also provide a full description (200-300 words) optimized for SEO with keywords naturally woven in.

4. Return ONLY valid JSON with this exact schema:
{
  "titles": ["title 1", "title 2", "title 3"],
  "chapters": [
    {"timestamp": "00:00", "title": "Hook"},
    ...
  ],
  "description": "Full description text...",
  "hook": "The hook sentence from the narration",
  "primary_keyword": "main keyword/topic"
}

Be precise, factual, and optimized for maximum CTR without misleading viewers."""

MODEL = "openai/gpt-oss-120b"


def generate_titles_and_chapters(
    transcript: str,
    topic: str = "",
    api_key: str | None = None,
) -> Dict:
    """
    Funnel transcript into AI endpoint to generate 3 high-CTR titles and structured chapters.
    """
    groq_key = sanitize_credential(api_key or os.environ.get("GROQ_API_KEY", ""))
    if not groq_key:
        log("No GROQ_API_KEY found; returning fallback metadata")
        return {
            "titles": [
                f"{topic} Explained Simply" if topic else "This Will Change Your Mind",
                f"{topic} Made Easy" if topic else "Here's What You Need to Know",
                f"Why {topic} Matters" if topic else "The Truth About This",
            ],
            "chapters": [
                {"timestamp": "00:00", "title": "Hook"},
                {"timestamp": "00:10", "title": "Problem"},
                {"timestamp": "00:25", "title": "Explanation"},
                {"timestamp": "00:40", "title": "Example"},
                {"timestamp": "00:50", "title": "Key Takeaway"},
            ],
            "description": transcript[:250] if transcript else "",
            "hook": transcript.split(".")[0] if transcript else "",
            "primary_keyword": topic,
        }

    client = Groq(api_key=groq_key)
    user_prompt = f"""Topic: {topic}

Transcript/Narration:
{transcript}

Generate 3 high-CTR titles and structured description chapters for this video. Return ONLY valid JSON with the required schema."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=2048,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("Empty response from AI")
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        log(f"Generated {len(data.get('titles', []))} titles and {len(data.get('chapters', []))} chapters")
        return data
    except Exception as exc:
        log(f"AI title generation failed: {exc}")
        return {
            "titles": [f"{topic} Explained Fast" if topic else "Quick Explanation"],
            "chapters": [{"timestamp": "00:00", "title": "Intro"}],
            "description": transcript[:200] if transcript else "",
            "hook": "",
            "primary_keyword": topic,
        }
