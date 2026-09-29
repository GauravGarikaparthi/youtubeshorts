"""
Per-run internet research for the Finance niche.

This module runs BEFORE the script is written on every scheduled upload, so
each Short is built from what people are actually asking about money right now
rather than from a static, stale topic list.

What it scrapes (all keyless, public, no paid API):

  1. Google News RSS   -- news.google.com/rss/search?q=... over several finance
     queries. Gives the freshest *language* people are using about money.
  2. Reddit hot posts  -- r/personalfinance, r/finance, r/fintech, r/investing,
     r/StockMarket. Where people ask the blunt, unpolished questions that make
     good answer-engine (AEO) targets.
  3. YouTube autocomplete -- what the search box actually suggests (reused from
     seo_research, so there is only one implementation).
  4. Google Trends related queries -- top + rising queries for "personal
     finance" / "investing".

Those signals are then used two ways:

  * TOPIC SELECTION -- every evergreen finance topic in trend_fetch's finance
    pool is scored against the live corpus (weighted toward fresh headlines),
    and the winner becomes today's topic. Recent winners are deprioritized via
    work/finance_history.json so six runs a day do not all land on the same
    subject.
  * STRATEGY / AEO CONTEXT -- the live questions and headlines are handed to
    the script generator so the narration answers a question that exists,
    leads with the answer (answer-engine optimisation), and speaks the target
    search phrase in the first seconds.

DESIGN RULE: this module must never fail a run. Every network call is
best-effort and time-boxed; if the internet is unreachable the static pools in
trend_fetch still produce a valid topic, and the caller gets
source="fallback" so it can log that it is shipping un-enriched content.

Nothing here touches copyrighted material: only headlines/titles and search
phrases are read, and only for topic selection and phrasing. No article or
post text is copied into the script.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import requests

from seo_research import get_related_queries, get_youtube_autocomplete
from trend_fetch import CATEGORY_TOPICS

LOG_PREFIX = "[finance_research]"

# Keyless public endpoints only. Any change here must stay free and unauthenticated.
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"

# Reddit's .json endpoints are unavailable here: www.reddit.com returns 403
# "Blocked" to datacenter IPs (every GitHub Actions runner), and old.reddit.com
# answers with HTML instead of JSON. The public Atom feed on www.reddit.com is
# served to a normal browser UA, so that is the reliable route.
REDDIT_FEED_URL = "https://www.reddit.com/r/{subreddit}/.rss"
ATOM_NS = "http://www.w3.org/2005/Atom"
RSS_NS = "http://purl.org/rss/1.0/"

# Stack Exchange's public API is keyless and returns real, upvoted money
# questions -- an excellent source of answer-engine questions even when Reddit
# is unreachable.
STACKEXCHANGE_SEARCH = "https://api.stackexchange.com/2.3/search/advanced"
STACKEXCHANGE_SITES = ["money", "quant"]
STACKEXCHANGE_PAGE_SIZE = 15

# Reddit is frequently blocked from cloud IPs without a real-looking UA.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

HTTP_TIMEOUT = 12

# Reddit throttles bursts; pause between feed requests and retry a 429 once.
REDDIT_RATE_LIMIT_PAUSE = 3.0
REDDIT_FEED_GAP = 1.5

# Finance queries used to build the live corpus. Deliberately mixes evergreen
# search-intent queries ("how to save money") with news-shaped queries
# ("stock market today") so both short-form explainers and timely takes surface.
NEWS_QUERIES = [
    "personal finance when you are:25",
    "how to save money",
    "index fund investing",
    "credit score improvement",
    "stock market today",
    "retirement savings",
    "debt payoff strategy",
    "side hustle income",
    "tax season money tips",
    "emergency fund",
]

# Kept deliberately short: Reddit throttles hard from datacenter IPs, so each
# extra feed buys little and costs a rate-limit wait. Stack Exchange below is
# the dependable question source; Reddit is a bonus when it answers.
REDDIT_SUBREDDITS = [
    "personalfinance",
    "investing",
]

TREND_SEEDS = ["personal finance", "investing for beginners"]

# Question shapes that make strong answer-engine (AEO) targets. Autocomplete
# returns the raw phrase; these prefixes tell us it is a question worth
# answering verbatim on screen.
QUESTION_MARKERS = (
    "how to", "how do i", "how do you", "how can i", "why", "what is", "what are",
    "when should", "where to", "which", "can i", "do i", "is it", "should i",
)

# Finance is a YMYL-adjacent niche. These guardrails go into the prompt so the
# script never gives personalised advice, promises returns, or invents numbers.
FINANCE_COMPLIANCE_RULES = [
    "Never give personalised investment, tax, or legal advice -- speak in general "
    "educational terms only (\"most people\", \"in general\").",
    "Never promise or imply a guaranteed return, a specific outcome, or a price "
    "target. No \"you WILL make\", \"this always works\", or \"risk-free\" claims.",
    "Never invent a specific statistic, interest rate, fee, or tax rule. If a number "
    "matters, describe it qualitatively instead of making one up.",
    "Frame risky topics (crypto, options, day trading, leverage) honestly and briefly "
    "flag the risk rather than hyping them.",
    "No sell signals, no \"buy this now\", no ticker calls, no shilling of any product "
    "or platform.",
]

# Content/SEO/AEO strategy rules handed to the script generator with the live
# research. Written as instructions, not prose, so they land cleanly in the prompt.
FINANCE_STRATEGY_RULES = [
    "Answer-first (AEO): the first sentence after the hook must be the direct answer "
    "or the concrete number. Google, YouTube and AI answer engines quote the clearest "
    "short answer they can find -- bury nothing.",
    "Speak the exact target search phrase naturally within the first 10 seconds, in "
    "one grammatical sentence. YouTube auto-captions are what classify the video into "
    "the Finance niche, so the phrase must be audible, not just in the title.",
    "Use one concrete, universally-true finance number or rule of thumb (for example "
    "an emergency-fund target in months of expenses, or the 50/30/20 shape of a "
    "budget) rather than vague generalities.",
    "Keep the title a truthful question or a specific value so it can win a featured "
    "snippet. Never use \"you won't believe\", \"shocking\", or \"secret\".",
    "Description: first line exactly \"#shorts\", then 2-3 sentences that restate the "
    "answer in plain language (this is what gets quoted as an answer), then a short "
    "FAQ-style line, then the subscribe CTA.",
    "Tags must include the broad \"personal finance\" / \"investing\" terms plus the "
    "long-tail phrase the video actually answers.",
]


def log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", flush=True)


def _get_text(url: str, params: dict | None = None, retries: int = 0) -> str:
    """
    GET with a browser UA. Returns "" on any failure -- never raises.

    retries only helps for 429s, which Reddit returns aggressively when feeds
    are pulled back to back; the pause is long enough to clear its window.
    """
    for attempt in range(retries + 1):
        try:
            resp = requests.get(
                url,
                params=params,
                timeout=HTTP_TIMEOUT,
                headers={"User-Agent": BROWSER_UA, "Accept-Language": "en-US,en;q=0.9"},
            )
            if resp.status_code == 429 and attempt < retries:
                time.sleep(REDDIT_RATE_LIMIT_PAUSE * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.text
        except Exception as exc:  # noqa: BLE001 - research is strictly best-effort
            log(f"GET {url.split('?')[0]} failed ({exc})")
            return ""
    return ""


def _dedupe(items: list[str], limit: int = 25) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        cleaned = " ".join(str(item).split())
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Scrapers
# ---------------------------------------------------------------------------

def fetch_news_headlines(queries: list[str] | None = None, limit: int = 40) -> list[str]:
    """Headline strings from Google News RSS across the finance query ladder."""
    headlines: list[str] = []
    for query in queries or NEWS_QUERIES:
        xml_text = _get_text(
            GOOGLE_NEWS_RSS,
            params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
        )
        if not xml_text:
            continue
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            log(f"Could not parse Google News RSS for '{query}' ({exc})")
            continue
        for item in root.iterfind(".//item"):
            title = (item.findtext("title") or "").strip()
            # Google News appends " - Publisher" to every title.
            if " - " in title:
                title = title.rsplit(" - ", 1)[0]
            if title:
                headlines.append(title)
        if len(headlines) >= limit * 2:
            break
    return _dedupe(headlines, limit=limit)


def fetch_reddit_titles(subreddits: list[str] | None = None) -> list[str]:
    """Post titles from finance subreddits' public Atom feeds (no key needed)."""
    titles: list[str] = []
    feeds = list(subreddits or REDDIT_SUBREDDITS)
    for index, subreddit in enumerate(feeds):
        if index:
            # Spread the requests out; Reddit returns 429 for back-to-back pulls.
            time.sleep(REDDIT_FEED_GAP)
        xml_text = _get_text(
            REDDIT_FEED_URL.format(subreddit=subreddit), retries=1
        )
        if not xml_text:
            continue
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            log(f"Could not parse r/{subreddit} feed ({exc})")
            continue

        # Explicit brace notation, not a prefixed path: a default-namespaced
        # Atom feed does not reliably match "entry" with a prefix->uri map.
        # Reddit's feed is Atom; the RSS branch is cheap insurance.
        entries = root.findall(f"{{{ATOM_NS}}}entry")
        if entries:
            for entry in entries:
                title = (entry.findtext(f"{{{ATOM_NS}}}title") or "").strip()
                if " - Page " in title:
                    title = title.rsplit(" - Page ", 1)[0]
                if title:
                    titles.append(title)
        else:
            for entry in root.findall(f"{{{RSS_NS}}}item"):
                title = (entry.findtext(f"{{{RSS_NS}}}title") or "").strip()
                if title:
                    titles.append(title)
    return _dedupe(titles, limit=40)


def fetch_stackexchange_questions(sites: list[str] | None = None) -> list[str]:
    """
    Real, community-voted money questions from Stack Exchange. Keyless public
    API, and a dependable stand-in for Reddit when Reddit blocks the runner.
    """
    questions: list[str] = []
    for site in sites or STACKEXCHANGE_SITES:
        payload = _get_text(
            STACKEXCHANGE_SEARCH,
            params={
                "site": site,
                "pagesize": STACKEXCHANGE_PAGE_SIZE,
                "order": "desc",
                "sort": "votes",
                "filter": "default",
            },
        )
        if not payload:
            continue
        try:
            items = json.loads(payload).get("items", [])
        except (json.JSONDecodeError, AttributeError) as exc:
            log(f"Could not parse Stack Exchange results for '{site}' ({exc})")
            continue
        for item in items:
            title = _strip_html(str(item.get("title") or ""))
            if title:
                questions.append(title)
    return _dedupe(questions, limit=30)


def _strip_html(text: str) -> str:
    """Stack Exchange escapes HTML entities and can inline tags in titles."""
    unescaped = (
        text.replace("&quot;", '"').replace("&#39;", "'")
        .replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    )
    return " ".join(re.sub(r"<[^>]+>", "", unescaped).split())


# YouTube autocomplete is excellent for search intent but also returns a lot of
# noise for finance queries: translations ("... in hindi", "... in tamil"),
# personal-channel fan queries ("... by <creator>"), and one-off proper nouns.
# None of those describe what a viewer searching this topic actually wants, and
# feeding them to the script writer dilutes the SEO signal, so they are dropped.
NOISE_PHRASES = {
    "in hindi", "in tamil", "in telugu", "in bengali", "in marathi", "in kannada",
    "in malayalam", "in gujarati", "in punjabi", "in urdu", "in arabic", "in turkish",
    "in spanish", "in french", "in german", "in chinese", "in japanese", "in korean",
    "in indonesian", "in russian", "in portuguese", "in italian", "in nepali",
    "in odia", "in assamese", "in sindhi", "in pashto", "in persian", "in bangla",
    "meaning in hindi", "full form", "quiz", "wikipedia", "ppt", "pdf", "notes",
}

# Autocomplete often hangs a bare language or country name off the end of an
# otherwise good phrase ("investment basics for beginners tamil"), with no "in"
# to match NOISE_PHRASES against. A trailing token from this set means the
# phrase is really a translation/region variant, not the search intent we want.
NOISE_TRAILING_TOKENS = {
    "hindi", "tamil", "telugu", "bengali", "marathi", "kannada", "malayalam",
    "gujarati", "punjabi", "urdu", "arabic", "turkish", "spanish", "french",
    "german", "chinese", "japanese", "korean", "indonesian", "russian",
    "portuguese", "italian", "nepali", "odia", "assamese", "sindhi", "pashto",
    "persian", "bangla", "india", "indian", "canada", "canadian", "usa",
    "america", "american", "uk", "britain", "pakistan", "australia", "nigeria",
    "egypt", "saudi", "dubai", "philippines", "vietnam", "thailand",
}
_PERSON_QUERY = re.compile(r"\sby\s[a-z]", re.IGNORECASE)
_MAX_PHRASE_WORDS = 9


def _is_useful_phrase(phrase: str) -> bool:
    lowered = phrase.lower().strip()
    if not lowered or len(lowered) < 3:
        return False
    if any(noise in lowered for noise in NOISE_PHRASES):
        return False
    if _PERSON_QUERY.search(lowered):
        return False
    if len(lowered.split()) > _MAX_PHRASE_WORDS:
        return False
    words = re.findall(r"[a-z0-9]+", lowered)
    if words and words[-1] in NOISE_TRAILING_TOKENS:
        return False
    # Reject phrases that are almost entirely digits.
    letters = [ch for ch in lowered if ch.isalpha()]
    return len(letters) >= max(3, len(lowered) // 2)


def collect_search_phrases(topic: str | None = None) -> list[str]:
    """
    YouTube autocomplete + Google Trends related queries. When no topic is given
    the trend seeds are used so we still harvest finance search language even
    before a topic is chosen.
    """
    seeds = [topic] if topic else list(TREND_SEEDS)
    phrases: list[str] = []
    for seed in seeds:
        if not seed:
            continue
        phrases.extend(get_youtube_autocomplete(seed))
        phrases.extend(get_related_queries(seed))
    return _dedupe([p for p in phrases if _is_useful_phrase(p)], limit=30)


def extract_pain_points(phrases: list[str], titles: list[str], limit: int = 12) -> list[str]:
    """
    Pull out the questions people are actually asking, from both autocomplete
    phrases and Reddit titles. These become the AEO targets the script must
    answer outright.
    """
    candidates: list[str] = []
    for phrase in list(phrases) + list(titles):
        lowered = phrase.lower().strip()
        is_question = lowered.endswith("?") or lowered.startswith(QUESTION_MARKERS)
        if is_question and 8 <= len(lowered) <= 90:
            candidates.append(phrase.strip())
    return _dedupe(candidates, limit=limit)


# ---------------------------------------------------------------------------
# Topic scoring
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "of", "for", "to", "and", "or", "in", "on", "with", "how",
    "what", "why", "when", "where", "which", "who", "your", "you", "my", "is",
    "are", "do", "does", "it", "that", "this", "at", "by", "from", "i", "me",
    "can", "should", "be", "as", "if", "but", "so", "about", "into", "vs",
}


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _load_history(path: str) -> list[dict]:
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as exc:
        log(f"Could not read finance history ({exc}) -- starting fresh.")
    return []


def _save_history(path: str, entry: dict, keep: int = 40) -> None:
    history = [h for h in _load_history(path) if isinstance(h, dict)]
    history.append(entry)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(history[-keep:], handle, indent=2)
    except OSError as exc:
        log(f"Could not write finance history ({exc}) -- ignoring.")


def _decay_penalty(history: list[dict], topic: str, now: datetime) -> float:
    """
    Penalty in [0, 1) for how recently this topic was already used. A topic used
    in the last 12 hours is heavily penalized, and the penalty fades over ~3 days,
    so six runs a day spread across the pool instead of repeating.
    """
    topic_tokens = _tokens(topic)
    penalty = 0.0
    for record in history:
        used_at = str(record.get("used_at") or "")
        try:
            when = datetime.fromisoformat(used_at)
        except ValueError:
            continue
        if when.tzinfo is not None:
            when = when.replace(tzinfo=None)
        age_hours = max((now - when).total_seconds() / 3600.0, 0.0)
        if age_hours > 96:
            continue
        previous = str(record.get("topic") or "")
        overlap = len(topic_tokens & _tokens(previous)) / max(len(topic_tokens), 1)
        if overlap < 0.5:
            continue
        recency = max(0.0, 1.0 - age_hours / 96.0)
        penalty = max(penalty, recency * overlap)
    return penalty


def score_topics(
    candidates: list[str],
    headlines: list[str],
    phrases: list[str],
    history: list[dict],
    now: datetime,
) -> list[tuple[str, float]]:
    """
    Score every candidate topic against the live corpus and return them ranked.

    Live headlines carry the most weight (freshest demand signal), search
    phrases carry less, and the recency penalty is subtracted. A small random
    jitter breaks ties so repeated runs with identical (or fully blocked)
    research still vary.
    """
    headline_tokens = [_tokens(h) for h in headlines]
    phrase_tokens = [_tokens(p) for p in phrases]
    scored: list[tuple[str, float]] = []

    for topic in candidates:
        topic_tokens = _tokens(topic)
        if not topic_tokens:
            continue
        score = 0.0
        for index, tokens in enumerate(headline_tokens):
            overlap = len(topic_tokens & tokens) / len(topic_tokens)
            if overlap > 0:
                # Earlier headlines are the day's biggest stories.
                score += overlap * max(1.0, 3.0 - index * 0.08)
        for tokens in phrase_tokens:
            overlap = len(topic_tokens & tokens) / len(topic_tokens)
            if overlap > 0:
                score += overlap * 1.5

        # Proportional jitter reacts to how much live signal a topic has; the
        # additive term breaks ties, which is what keeps the offline fallback
        # (every score 0.0) from returning the same first topic all day.
        score = score * random.uniform(0.9, 1.1) + random.uniform(0.0, 0.5)
        score -= _decay_penalty(history, topic, now) * 25.0
        scored.append((topic, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def research_finance(
    category: str = "finance",
    history_path: str = os.path.join("work", "finance_history.json"),
    persist: bool = True,
) -> dict:
    """
    Scrape live finance signals and pick the best topic + strategy for this run.

    Returns a dict shaped for generate_script():
      {
        "topic":        str,   # the chosen topic (already de-duplicated recently)
        "keywords":     list,  # live search phrases, best first
        "headlines":    list,  # fresh news context (phrasing only)
        "pain_points":  list,  # real questions the script must answer
        "strategy":     list,  # content/SEO/AEO instructions
        "compliance":   list,  # finance-accuracy guardrails
        "source":       str,   # "live" | "partial" | "fallback"
      }

    Never raises. On a total network failure it returns the first finance topic
    from trend_fetch with source="fallback".
    """
    now = datetime.utcnow()
    candidates = list(CATEGORY_TOPICS.get(category) or CATEGORY_TOPICS["finance"])

    headlines = fetch_news_headlines()
    titles = fetch_reddit_titles()
    questions = fetch_stackexchange_questions()
    phrases = collect_search_phrases()

    live_signals = headlines + titles + questions + phrases
    if not live_signals:
        log("All live finance sources were unreachable -- using the static pool.")
    else:
        log(
            f"Live finance signals: {len(headlines)} headlines, "
            f"{len(titles)} reddit titles, {len(questions)} stack exchange questions, "
            f"{len(phrases)} search phrases."
        )

    history = _load_history(history_path)
    ranked = score_topics(candidates, headlines, phrases, history, now)
    topic = ranked[0][0] if ranked else candidates[0]

    # Re-derive search phrases scoped to the winning topic so the script gets
    # phrasing for what it is actually about, not just the generic finance seed.
    topic_phrases = collect_search_phrases(topic) if topic else []
    keywords = _dedupe(topic_phrases + phrases, limit=25)

    if not headlines and not titles and not phrases:
        source = "fallback"
    elif keywords:
        source = "live"
    else:
        source = "partial"

    pain_points = extract_pain_points(keywords or phrases, titles + questions)
    top_scores = ", ".join(f"{t} ({s:.1f})" for t, s in ranked[:3])

    if persist:
        _save_history(history_path, {
            "topic": topic,
            "used_at": now.isoformat(),
            "keywords": keywords[:8],
            "pain_points": pain_points[:5],
            "source": source,
        })

    log(f"Chosen finance topic: '{topic}' (source={source}). Ranking: {top_scores}")
    if pain_points:
        log(f"AEO pain points: {pain_points[:3]}")

    return {
        "topic": topic,
        "keywords": keywords,
        "headlines": headlines[:12],
        "pain_points": pain_points,
        "strategy": list(FINANCE_STRATEGY_RULES),
        "compliance": list(FINANCE_COMPLIANCE_RULES),
        "source": source,
    }


def is_finance_category(category: str) -> bool:
    return (category or "").strip().lower() == "finance"


__all__ = [
    "research_finance",
    "is_finance_category",
    "FINANCE_STRATEGY_RULES",
    "FINANCE_COMPLIANCE_RULES",
    "NEWS_QUERIES",
    "REDDIT_SUBREDDITS",
]

if __name__ == "__main__":
    result = research_finance(persist=False)
    print(json.dumps(result, indent=2))
