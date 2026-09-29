from __future__ import annotations

import re


MAX_TITLE_CHARS = 70
MIN_TITLE_CHARS = 60
SHORTS_TAG = "#shorts"

TITLE_MODIFIERS = (
    "best",
    "easy",
    "ultimate",
    "simple",
    "fast",
    "complete",
    "proven",
    "real",
)

FALSE_CLICKBAIT_PATTERNS = (
    (r"you won'?t believe", ""),
    (r"shocking", ""),
    (r"insane", ""),
    (r"unbelievable", ""),
    (r"click here", ""),
    (r"watch before it'?s too late", ""),
    (r"they don'?t want you to know", ""),
    (r"forbidden secret", "key facts"),
)

VALUE_PHRASES = (
    "Key Facts You Can Use",
    "Clear, Practical Guide",
    "What You Need to Know",
    "Simple Steps and Examples",
    "Essential Details Explained",
    "How It Works in Practice",
    "Key Takeaways",
    "Clear Guide",
    "Practical Tips",
    "Explained Simply",
    "Useful Takeaways",
    "Key Facts",
)


def _clean_keyword(keyword: str) -> str:
    cleaned = re.sub(r"#shorts\b", "", keyword or "", flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip(" \t\r\n-–—|,;:")


def clean_title_text(text: str) -> str:
    cleaned = re.sub(r"#shorts\b", "", text or "", flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\r\n-–—|,;:")
    for pattern, replacement in FALSE_CLICKBAIT_PATTERNS:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+([,.;:!?)])", r"\1", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip(" \t\r\n-–—|,;:")


def capitalize_first_letter(text: str) -> str:
    for index, char in enumerate(text):
        if char.isalpha():
            return text[:index] + char.upper() + text[index + 1 :]
    return text


def has_strong_modifier(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:best|easy|ultimate|simple|fast|complete|proven|real|top|guide|tips|facts|explained|how to|\d+)\b|\[|\(",
            text,
            flags=re.IGNORECASE,
        )
    )


def trim_title(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > int(max_chars * 0.6):
        cut = cut[:last_space]
    return cut.rstrip(" -–—|,;:")


def _fit_title_to_budget(
    text: str,
    keyword: str,
    budget: int,
    minimum: int,
) -> str:
    if not keyword or len(keyword) > budget:
        return trim_title(text or keyword, budget)

    remainder = (
        text[len(keyword):].strip(" \t\r\n-–—|,;:")
        if text.lower().startswith(keyword.lower())
        else text
    )
    remainder_budget = budget - len(keyword) - 3
    if not remainder or remainder_budget < 1:
        fitted = keyword
    else:
        trimmed_remainder = trim_title(remainder, remainder_budget)
        fitted = f"{keyword} — {trimmed_remainder}".strip(" \t\r\n-–—|,;:")

    used_phrases = {phrase.lower() for phrase in VALUE_PHRASES if phrase.lower() in fitted.lower()}
    while len(fitted) < minimum:
        candidates = [
            phrase
            for phrase in VALUE_PHRASES
            if phrase.lower() not in used_phrases
            and len(f"{fitted} — {phrase}") <= budget
        ]
        if not candidates:
            break
        reaching = [
            phrase
            for phrase in candidates
            if len(fitted) + 3 + len(phrase) >= minimum
        ]
        candidate = (
            min(reaching, key=lambda phrase: len(fitted) + 3 + len(phrase))
            if reaching
            else max(candidates, key=len)
        )
        fitted = f"{fitted} — {candidate}".strip()
        used_phrases.add(candidate.lower())

    return trim_title(fitted, budget)


def format_youtube_title(
    raw: str,
    keyword: str,
    *,
    suffix: str = "",
    max_chars: int = MAX_TITLE_CHARS,
    min_chars: int = MIN_TITLE_CHARS,
    default_modifier: str = "Easy Guide",
) -> str:
    cleaned_keyword = _clean_keyword(keyword)
    cleaned_raw = clean_title_text(raw)

    if cleaned_keyword and cleaned_raw.lower().startswith(cleaned_keyword.lower()):
        cleaned_raw = cleaned_keyword + cleaned_raw[len(cleaned_keyword) :]
    elif cleaned_keyword:
        cleaned_raw = f"{cleaned_keyword} {cleaned_raw}".strip()
    else:
        cleaned_raw = cleaned_raw or cleaned_keyword

    if cleaned_raw and default_modifier and not has_strong_modifier(cleaned_raw):
        cleaned_raw = f"{cleaned_raw} — {default_modifier}".strip()

    if cleaned_keyword and not cleaned_raw.lower().startswith(cleaned_keyword.lower()):
        cleaned_raw = f"{cleaned_keyword} {cleaned_raw}".strip()

    budget = max(1, max_chars - len(suffix))
    minimum = min(budget, max(1, min_chars - len(suffix)))
    cleaned_raw = _fit_title_to_budget(cleaned_raw, cleaned_keyword, budget, minimum)
    return capitalize_first_letter(f"{cleaned_raw}{suffix}")[:max_chars]
