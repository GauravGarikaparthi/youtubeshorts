"""
SEO / GEO / AEO optimization engine for regular long-form videos.

Provides:
  - High-CTR title generation (primary keyword first, factual value, truthful modifiers)
  - Trending tag generation (broad + long-tail mix)
  - SEO-optimized description (keyword-rich first 2 lines, chapter timestamps,
    subscribe CTA)
  - GEO (Google/E-E-A-T) signals: authoritativeness, expertise, helpfulness
  - AEO (Answer Engine Optimization): FAQ-style structured data, concise
    answer boxes, featured-snippet targeting
  - Engaging caption (comment-section AEO hook)
  - AEO-optimized thumbnail text (max 3 bold words)

The optimizer takes a topic, a keyword list, and an optional performance
context (from the recursive learning engine) and produces fully-optimized
metadata ready for YouTube upload.
"""

from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from groq import Groq
from _sanitize import sanitize_credential
from title_utils import format_youtube_title


LOG_PREFIX = "[seo_optimizer]"


def log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", flush=True)


# ---------------------------------------------------------------------------
# SEO/GEO/AEO power-word banks
# ---------------------------------------------------------------------------

VALUE_POWER_WORDS = [
    "best", "easy", "ultimate", "simple", "proven", "essential",
    "complete", "practical", "clear", "key",
]

# Primary keywords for spoken keyword sync (rule 7b)
DEFAULT_KEYWORD_BUCKETS = {
    "how-to": ["how to", "tutorial", "guide", "step by step"],
    "what-is": ["what is", "explain", "meaning", "definition"],
    "why": ["why", "reason", "because", "causes"],
    "best": ["best", "top", "ranking", "vs"],
    "mistake": ["mistake", "error", "wrong", "fail"],
}


@dataclass
class SEOMetadata:
    """Optimized metadata for a long-form YouTube video."""

    title: str
    description: str
    tags: List[str]
    caption: str
    thumbnail_text: str
    keywords_spoken: List[str]
    faq_schema: List[dict] = field(default_factory=list)
    structured_data: dict = field(default_factory=dict)


class SEOOptimizer:
    """
    Generates SEO/GEO/AEO-optimized metadata for long-form videos.

    Uses rule-based title formatting for factual, keyword-first metadata, with
    optional Groq refinement for tags and descriptions.
    """

    def __init__(self, groq_api_key: Optional[str] = None):
        self.groq_api_key = sanitize_credential(
            groq_api_key or os.environ.get("GROQ_API_KEY", "")
        )
        self.client = Groq(api_key=self.groq_api_key) if self.groq_api_key else None
        self._cache_dir = Path("work/seo_cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Title generation
    # ------------------------------------------------------------------

    def _regular_title_modifier(self, topic: str) -> str:
        lowered = (topic or "").lower()
        if re.search(r"\b(how to|tutorial|guide|learn|fix|make)\b", lowered):
            return "Easy Guide"
        if re.search(r"\b(best|top|review|ranking|vs)\b", lowered):
            return "Best Guide"
        if re.search(r"\b(history|story|facts|explained|science)\b", lowered):
            return "Key Facts"
        return "Ultimate Guide"

    def _title_value_phrase(self, topic: str) -> str:
        lowered = (topic or "").lower()
        if re.search(r"\b(how to|tutorial|guide|learn|fix|make)\b", lowered):
            return "Easy Step-by-Step Guide"
        if re.search(r"\b(best|top|review|ranking|vs)\b", lowered):
            return "Best Options Explained"
        if re.search(r"\d+", lowered):
            return "Key Facts and Examples"
        return "Key Facts You Can Use"

    def generate_ctr_title(self, topic: str, primary_keyword: str, performance_context: dict | None = None) -> str:
        """
        Generate a factual title with the exact topic first, a strong modifier,
        and a maximum length of 70 characters.
        """
        keyword = (primary_keyword or topic).strip()
        value_phrase = self._title_value_phrase(topic)
        raw = f"{topic or keyword} — {value_phrase}"
        if len(raw) < 60:
            raw = f"{raw} — What You Need to Know"
        return format_youtube_title(
            raw,
            keyword,
            default_modifier=self._regular_title_modifier(topic),
        )

    # ------------------------------------------------------------------
    # Tag generation
    # ------------------------------------------------------------------

    def generate_trending_tags(self, topic: str, keywords: List[str], performance_context: dict | None = None) -> List[str]:
        """
        Generate 15-20 trending tags mixing broad and long-tail keywords.
        Tags are ordered by search volume / specificity.
        """
        tags = set()

        # Primary keyword + variations
        primary = keywords[0] if keywords else topic
        tags.add(primary)
        tags.add(topic)

        # Add SEO keywords
        for kw in keywords[:15]:
            tags.add(kw.strip().lower())

        # Add power-word-enhanced tags
        for word in VALUE_POWER_WORDS[:5]:
            tags.add(f"{topic} {word}")

        # Category tags
        for bucket_name, bucket_kws in DEFAULT_KEYWORD_BUCKETS.items():
            for kw in bucket_kws:
                if kw in topic.lower() or kw in primary.lower():
                    tags.add(kw)

        # Add historical high-performing tags from performance context
        if performance_context:
            for tag in performance_context.get("high_performing_tags", []):
                tags.add(tag)

        # De-dupe, preserve order, cap at 20
        seen = set()
        ordered = []
        for tag in sorted(tags, key=lambda t: (-len(t), t)):
            if tag.lower() not in seen and len(tag) >= 2:
                seen.add(tag.lower())
                ordered.append(tag)
            if len(ordered) >= 20:
                break

        return ordered

    # ------------------------------------------------------------------
    # Description generation
    # ------------------------------------------------------------------

    def generate_seo_description(
        self,
        topic: str,
        keywords: List[str],
        estimated_duration: float,
        chapters: List[tuple[float, str]] | None = None,
        performance_context: dict | None = None,
    ) -> str:
        """
        Generate a 200-400 word SEO-optimized description:
          - Keyword-rich first 2 lines (most searchable)
          - Compelling summary
          - Chapter timestamps
          - Explicit subscribe CTA
        """
        primary_kw = keywords[0] if keywords else topic

        # Keyword-rich opening
        opening = (
            f"{topic} explained simply. "
            f"This video covers {primary_kw}, "
            f"{keywords[1] if len(keywords) > 1 else 'key insights'}, "
            f"and {keywords[2] if len(keywords) > 2 else 'practical applications'}. "
            f"Stay tuned for a clear summary and practical takeaways."
        )

        # Summary body
        summary_lines = [
            f"In this video, we explain {topic} in a clear, practical way.",
            f"We explore {primary_kw} and how it applies in real situations.",
            f"You'll learn key details, common misconceptions, and useful takeaways.",
            f"We also cover practical examples so you can understand {topic} with confidence.",
            f"By the end of this {int(estimated_duration / 60)}-minute video, you'll have a clearer understanding of {topic}.",
        ]

        # Chapter timestamps
        chapter_lines = ["\n⏱️ Chapter Timestamps:"]
        if chapters:
            for ts, label in chapters:
                mins = int(ts // 60)
                secs = int(ts % 60)
                chapter_lines.append(f"{mins:02d}:{secs:02d} — {label}")
        else:
            chapter_lines.append("00:00 — Introduction and Key Question")
            chapter_lines.append("00:10 — What This Video Covers")
            chapter_lines.append("01:00 — Core Concept")
            chapter_lines.append("02:30 — Practical Example")
            chapter_lines.append("04:30 — Common Misconceptions")
            chapter_lines.append("06:30 — Deeper Explanation")
            chapter_lines.append("08:00 — Key Takeaways")

        # CTA
        cta_lines = [
            "\n👍 If this video helped you, smash the LIKE button!",
            "🔔 Subscribe for more deep-dive explainers every single day!",
            "💬 Drop a comment with your biggest insight — I read every single one!",
            f"🔗 Learn more about {primary_kw}: [resource link placeholder]",
            f"📚 Related videos: [playlist link placeholder]",
        ]

        # Historical optimization note
        perf_note = ""
        if performance_context:
            improvement = performance_context.get("avg_ctr_lift", 0)
            if improvement:
                perf_note = f"\n\n🚀 Optimized based on historical data: +{improvement:.1f}% CTR observed."

        description = (
            opening
            + "\n\n"
            + "\n".join(summary_lines)
            + "\n"
            + "\n".join(chapter_lines)
            + "\n"
            + "\n".join(cta_lines)
            + perf_note
        )

        # Truncate to ~400 words if needed
        words = description.split()
        if len(words) > 400:
            description = " ".join(words[:400])

        return description

    # ------------------------------------------------------------------
    # AEO: FAQ schema + answer-box content
    # ------------------------------------------------------------------

    def generate_aeo_faq(self, topic: str, keywords: List[str]) -> List[dict]:
        """
        Generate FAQ-style structured data for Answer Engine Optimization.
        Each entry targets a featured-snippet-style question.
        """
        faqs = [
            {
                "@type": "Question",
                "name": f"What is {topic} and why does it matter?",
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": f"{topic} is a critical concept that impacts {keywords[0] if keywords else 'this field'}. Understanding it unlocks deeper insights into how {topic} works in practice.",
                },
            },
            {
                "@type": "Question",
                "name": f"Why does everyone get {topic} wrong?",
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": f"Most people confuse {topic} with its surface-level symptoms. The real {topic} operates on deeper principles that are rarely taught.",
                },
            },
            {
                "@type": "Question",
                "name": f"How can I master {topic} quickly?",
                "acceptedAnswer": {
                    "@type": "Answer",
                    f"text": f"The fastest path to mastering {topic} is to focus on {keywords[1] if len(keywords) > 1 else 'the fundamentals'} first, then build complexity layer by layer.",
                },
            },
        ]
        return faqs

    def generate_aeo_structured_data(self, topic: str, title: str, description: str) -> dict:
        """Generate structured data for featured snippet / rich result targeting."""
        return {
            "@context": "https://schema.org",
            "@type": "VideoObject",
            "name": title,
            "description": description[:200],
            "thumbnailUrl": "[thumbnail URL placeholder]",
            "uploadDate": datetime.utcnow().strftime("%Y-%m-%d"),
            "duration": "PT10M",
            "key": topic,
            "mainEntityOfPage": {
                "@type": "WebPage",
                "@id": "[video URL placeholder]",
            },
            "publisher": {
                "@type": "Organization",
                "name": "Auto-Generated Content",
                "logo": {
                    "@type": "ImageObject",
                    "url": "[logo URL placeholder]",
                },
            },
        }

    # ------------------------------------------------------------------
    # Thumbnail text (AEO/CTR)
    # ------------------------------------------------------------------

    def generate_thumbnail_text(
        self,
        title: str,
        performance_context: dict | None = None,
        topic: str | None = None,
    ) -> str:
        """
        Generate a short visual question for a regular-video thumbnail.
        """
        lowered = (topic or title).lower()
        if re.search(r"\b(how to|tutorial|guide|learn|fix|make)\b", lowered):
            question = "How It Works?"
        elif re.search(r"\b(why|reason|cause)\b", lowered):
            question = "Why It Matters?"
        elif re.search(r"\b(best|top|review|ranking|vs)\b", lowered):
            question = "Which One Wins?"
        elif re.search(r"\b(history|story|events)\b", lowered):
            question = "What Happened?"
        else:
            question = "What Matters?"

        words = [word for word in question.replace("?", "").split() if word]
        return " ".join(words[:3]).upper() + "?"

    # ------------------------------------------------------------------
    # Caption (comment-section AEO engagement hook)
    # ------------------------------------------------------------------

    def generate_engaging_caption(self, topic: str, keywords: List[str]) -> str:
        """
        Generate a 1-2 sentence engaging caption for the video comment
        section. Designed to maximize engagement (AEO).
        """
        primary_kw = keywords[0] if keywords else topic
        captions = [
            f"What is the most useful takeaway from {topic}? Share your answer below.",
            f"Which part of {primary_kw} was most surprising to you?",
            f"What question about {topic} should we explain next?",
            f"Save this video if the practical examples helped you understand {topic}.",
        ]
        idx = len(topic) % len(captions)
        return captions[idx]

    # ------------------------------------------------------------------
    # Spoken keyword sync (rule 7b)
    # ------------------------------------------------------------------

    def extract_spoken_keywords(self, topic: str, seo_keywords: List[str]) -> List[str]:
        """
        Extract 3-5 primary concept keywords that must be spoken in the
        first 10 seconds of the video for YouTube's automatic captioning to
        categorize the niche correctly.
        """
        keywords = [topic] + seo_keywords[:4]
        seen = set()
        result = []
        for kw in keywords:
            kw_clean = kw.strip().lower()
            if kw_clean and kw_clean not in seen:
                seen.add(kw_clean)
                result.append(kw.strip())
        return result[:5]

    # ------------------------------------------------------------------
    # Full optimization
    # ------------------------------------------------------------------

    def optimize(
        self,
        topic: str,
        keywords: List[str],
        estimated_duration: float = 600.0,
        chapters: List[tuple[float, str]] | None = None,
        performance_context: dict | None = None,
    ) -> SEOMetadata:
        """
        Run the full SEO/GEO/AEO optimization pipeline and return a
        SEOMetadata object with all fields populated.
        """
        primary_keyword = (topic or (keywords[0] if keywords else "")).strip()

        title = self.generate_ctr_title(topic, primary_keyword, performance_context)
        tags = self.generate_trending_tags(topic, keywords, performance_context)
        description = self.generate_seo_description(
            topic, keywords, estimated_duration, chapters, performance_context
        )
        caption = self.generate_engaging_caption(topic, keywords)
        thumbnail_text = self.generate_thumbnail_text(title, performance_context, topic=topic)
        spoken_keywords = self.extract_spoken_keywords(topic, keywords)
        faq_schema = self.generate_aeo_faq(topic, keywords)
        structured_data = self.generate_aeo_structured_data(topic, title, description)

        log(f"SEO optimization complete: title={title[:60]}... "
            f"tags={len(tags)} spoken_keywords={len(spoken_keywords)}")

        return SEOMetadata(
            title=title,
            description=description,
            tags=tags,
            caption=caption,
            thumbnail_text=thumbnail_text,
            keywords_spoken=spoken_keywords,
            faq_schema=faq_schema,
            structured_data=structured_data,
        )

    # ------------------------------------------------------------------
    # LLM refinement (uses Groq if available)
    # ------------------------------------------------------------------

    def refine_with_llm(self, title: str, description: str, tags: List[str], topic: str) -> dict:
        """
        Use the LLM to further refine title/description/tags for CTR and SEO.
        Falls back to rule-based results if Groq API is unavailable.
        """
        if not self.client:
            log("Groq API key not available — skipping LLM refinement.")
            return {"title": title, "description": description, "tags": tags}

        prompt = f"""\
You are a YouTube SEO and CTR optimization expert. Refine the following
metadata for maximum click-through rate and search discoverability.

Topic: {topic}
Current title: {title}
Current description: {description[:200]}...

Produce refined_title, refined_description, and refined_tags as JSON.
- Title: 60-70 characters maximum, exact topic first, factual value, truthful
  strong modifier, no false clickbait, first letter capitalized
- Description: 200-400 words, keyword-rich first 2 lines, chapter timestamps,
  subscribe CTA
- Tags: 15-20 items mixing broad and long-tail keywords

Return ONLY valid JSON with keys: refined_title, refined_description, refined_tags."""

        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-oss-120b",
                max_tokens=2048,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a YouTube SEO expert."},
                    {"role": "user", "content": prompt},
                ],
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(text)
            refined_title = data.get("refined_title", title)
            return {
                "title": format_youtube_title(
                    refined_title,
                    topic,
                    default_modifier=self._regular_title_modifier(topic),
                ),
                "description": data.get("refined_description", description),
                "tags": data.get("refined_tags", tags),
            }
        except Exception as exc:
            log(f"LLM refinement failed ({exc}) — using rule-based results.")
            return {"title": title, "description": description, "tags": tags}
