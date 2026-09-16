"""
YouTube virality metric evaluation engine.

Evaluates whether a generated Short or regular video satisfies YouTube
virality metrics by computing a composite ViralityScore based on:

  - Click-Through Rate (CTR)
  - Average View Duration (AVD) / View Duration
  - Watch Time (absolute and relative)
  - Audience Retention (especially the all-important first 30 seconds)
  - Engagement rate (likes, comments, shares)
  - Subscriber growth from the video
  - Traffic source distribution

The scorer compares each video against channel baselines and historical
data, returning a pass/fail recommendation plus specific corrective actions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import List, Optional, Dict, Any


LOG_PREFIX = "[virality]"


def log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", flush=True)


# ---------------------------------------------------------------------------
# Metric thresholds (industry benchmarks for YouTube)
# ---------------------------------------------------------------------------

# These are the thresholds a video must meet to be considered "viral-ready"
VIRALITY_THRESHOLDS = {
    "ctr_percent": 5.0,           # CTR >= 5% is good, < 2% is poor
    "avg_view_duration_ratio": 0.50,  # AVD / video_length >= 50%
    "watch_time_minutes": 30.0,   # absolute watch time (longer for shorts)
    "engagement_rate_percent": 5.0,  # (likes + comments + shares) / views * 100
    "first_15s_retention_percent": 60.0,  # retention at 15s mark
    "first_30s_retention_percent": 50.0,  # retention at 30s mark
    "relative_watch_time_percent": 70.0,  # watch time / upload duration
}

# Shorts-specific thresholds
SHORTS_THRESHOLDS = {
    "ctr_percent": 3.0,
    "avg_view_duration_ratio": 0.60,
    "watch_time_minutes": 5.0,
    "engagement_rate_percent": 8.0,
    "first_15s_retention_percent": 70.0,
    "first_30s_retention_percent": 60.0,
    "relative_watch_time_percent": 75.0,
}


@dataclass
class VideoMetrics:
    """Raw metrics for a single video (from YouTube Analytics API or data import)."""

    video_id: str
    title: str
    is_short: bool
    duration_seconds: float

    # Core metrics
    views: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    subscribers_gained: int = 0
    estimated_minutes_watched: float = 0.0
    average_view_duration_seconds: float = 0.0
    click_through_rate_percent: float = 0.0
    traffic_source_organic_percent: float = 0.0

    # Retention curve (sampled at key points)
    # retention_points: dict mapping second -> retention percentage
    retention_points: Dict[float, float] = field(default_factory=dict)

    # Metadata that was used
    tags_used: List[str] = field(default_factory=list)
    title_used: str = ""
    description_used: str = ""
    thumbnail_text: str = ""
    spoken_keywords: List[str] = field(default_factory=list)

    # When this video was published
    published_at: str = ""

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "title": self.title,
            "is_short": self.is_short,
            "duration_seconds": self.duration_seconds,
            "views": self.views,
            "likes": self.likes,
            "comments": self.comments,
            "shares": self.shares,
            "subscribers_gained": self.subscribers_gained,
            "estimated_minutes_watched": self.estimated_minutes_watched,
            "average_view_duration_seconds": self.average_view_duration_seconds,
            "click_through_rate_percent": self.click_through_rate_percent,
            "traffic_source_organic_percent": self.traffic_source_organic_percent,
            "retention_points": self.retention_points,
            "tags_used": self.tags_used,
            "title_used": self.title_used,
            "description_used": self.description_used,
            "thumbnail_text": self.thumbnail_text,
            "spoken_keywords": self.spoken_keywords,
            "published_at": self.published_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VideoMetrics":
        return cls(**data)


@dataclass
class ViralityScore:
    """Composite virality evaluation for a single video."""

    video_id: str
    overall_score: float  # 0-100
    component_scores: Dict[str, float]  # metric -> 0-100 score
    passed: bool
    corrective_actions: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "overall_score": round(self.overall_score, 1),
            "component_scores": {k: round(v, 1) for k, v in self.component_scores.items()},
            "passed": self.passed,
            "corrective_actions": self.corrective_actions,
            "summary": self.summary,
        }


class ViralityEvaluator:
    """
    Evaluates video metrics against virality thresholds and generates
    corrective actions for underperforming content.
    """

    def __init__(self, historical_data_path: str | Path | None = None):
        self.thresholds = VIRALITY_THRESHOLDS
        self.shorts_thresholds = SHORTS_THRESHOLDS
        self.historical_data_path = Path(historical_data_path or "work/virality_history.json")
        self.historical_data_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_history()

    def _load_history(self) -> None:
        """Load historical performance data for baseline comparison."""
        if not self.historical_data_path.exists():
            self.history: List[dict] = []
            return
        try:
            with open(self.historical_data_path, "r") as f:
                self.history = json.load(f)
        except (json.JSONDecodeError, OSError):
            self.history = []

    def _save_history(self) -> None:
        """Persist historical performance data."""
        try:
            with open(self.historical_data_path, "w") as f:
                json.dump(self.history, f, indent=2, default=str)
        except OSError as exc:
            log(f"Could not save history ({exc}).")

    def _channel_baseline(self, is_short: bool) -> Dict[str, float]:
        """Compute channel baseline metrics from historical data."""
        if not self.history:
            return {
                "ctr": 4.0,
                "avd_ratio": 0.55,
                "engagement": 5.0,
                "first_15s": 65.0,
                "first_30s": 55.0,
            }

        relevant = [
            h for h in self.history
            if h.get("is_short") == is_short and h.get("views", 0) > 100
        ]
        if not relevant:
            return {
                "ctr": 4.0,
                "avd_ratio": 0.55,
                "engagement": 5.0,
                "first_15s": 65.0,
                "first_30s": 55.0,
            }

        def _avg(key: str) -> float:
            vals = [h.get(key, 0) for h in relevant]
            return sum(vals) / len(vals) if vals else 0.0

        return {
            "ctr": _avg("ctr"),
            "avd_ratio": _avg("avd_ratio"),
            "engagement": _avg("engagement"),
            "first_15s": _avg("first_15s_retention"),
            "first_30s": _avg("first_30s_retention"),
        }

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def evaluate(self, metrics: VideoMetrics) -> ViralityScore:
        """
        Evaluate a single video's metrics against virality thresholds.
        Returns a ViralityScore with component scores, pass/fail, and
        corrective actions.
        """
        thresholds = self.shorts_thresholds if metrics.is_short else self.thresholds
        baseline = self._channel_baseline(metrics.is_short)

        component_scores: Dict[str, float] = {}
        corrective_actions: List[str] = []

        # --- CTR ---
        ctr = metrics.click_through_rate_percent
        ctr_relative = ctr / max(baseline["ctr"], 0.1)
        ctr_score = self._score_ratio(ctr, thresholds["ctr_percent"], baseline["ctr"])
        component_scores["ctr"] = ctr_score
        if ctr < thresholds["ctr_percent"]:
            corrective_actions.append(
                f"CTR is {ctr:.1f}% (threshold {thresholds['ctr_percent']}%). "
                "Rewrite title with stronger curiosity-gap hook and refresh thumbnail with higher contrast."
            )

        # --- Average View Duration ---
        if metrics.duration_seconds > 0:
            avd_ratio = metrics.average_view_duration_seconds / metrics.duration_seconds
        else:
            avd_ratio = 0.0
        avd_score = self._score_ratio(avd_ratio, thresholds["avg_view_duration_ratio"], baseline["avd_ratio"])
        component_scores["avg_view_duration"] = avd_score
        if avd_ratio < thresholds["avg_view_duration_ratio"]:
            corrective_actions.append(
                f"AVD ratio is {avd_ratio:.1%} (threshold {thresholds['avg_view_duration_ratio']:.0%}). "
                "Add micro-hooks every 45s, use jump cuts, and make the first 10s more visually dynamic."
            )

        # --- First 15s retention (most critical drop-off point) ---
        first_15s = self._retention_at(metrics, 15)
        first_15s_score = self._score_ratio(first_15s, thresholds["first_15s_retention_percent"], baseline["first_15s"])
        component_scores["first_15s_retention"] = first_15s_score
        if first_15s < thresholds["first_15s_retention_percent"]:
            corrective_actions.append(
                f"15s retention is {first_15s:.1f}% (threshold {thresholds['first_15s_retention_percent']}%). "
                "Apply WTF formula, shocking fact opener, and change visual every 1-2s in the intro."
            )

        # --- First 30s retention ---
        first_30s = self._retention_at(metrics, 30)
        first_30s_score = self._score_ratio(first_30s, thresholds["first_30s_retention_percent"], baseline["first_30s"])
        component_scores["first_30s_retention"] = first_30s_score
        if first_30s < thresholds["first_30s_retention_percent"]:
            corrective_actions.append(
                f"30s retention is {first_30s:.1f}% (threshold {thresholds['first_30s_retention_percent']}%). "
                "Remove dead air, add micro-hook at 45s mark, implement correction loop for engagement."
            )

        # --- Engagement rate ---
        engagement = self._engagement_rate(metrics)
        eng_score = self._score_ratio(engagement, thresholds["engagement_rate_percent"] * 0.6, baseline["engagement"])
        component_scores["engagement_rate"] = eng_score
        if engagement < thresholds["engagement_rate_percent"]:
            corrective_actions.append(
                f"Engagement rate is {engagement:.1f}% (threshold {thresholds['engagement_rate_percent']}%). "
                "Add CTAs throughout, pose questions, and end with engagement-baiting comment prompt."
            )

        # --- Watch time ---
        wt_minutes = metrics.estimated_minutes_watched
        wt_score = self._score_ratio(wt_minutes / 60, thresholds["watch_time_minutes"] / 60, baseline.get("watch_time", 0.5))
        component_scores["watch_time"] = wt_score

        # --- Overall score ---
        weightings = {
            "ctr": 0.20,
            "avg_view_duration": 0.20,
            "first_15s_retention": 0.25,
            "first_30s_retention": 0.15,
            "engagement_rate": 0.15,
            "watch_time": 0.05,
        }
        overall = sum(component_scores[k] * weightings.get(k, 0) for k in component_scores)
        component_scores["overall"] = overall

        passed = overall >= 70.0 and first_15s >= thresholds["first_15s_retention_percent"]

        summary = self._build_score_summary(metrics, component_scores, passed)

        return ViralityScore(
            video_id=metrics.video_id,
            overall_score=overall,
            component_scores=component_scores,
            passed=passed,
            corrective_actions=corrective_actions,
            summary=summary,
        )

    def _retention_at(self, metrics: VideoMetrics, seconds: float) -> float:
        """Get retention percentage at a given timestamp."""
        if not metrics.retention_points:
            return 0.0
        # Find the closest retention point
        sorted_points = sorted(metrics.retention_points.items(), key=lambda x: abs(x[0] - seconds))
        if not sorted_points:
            return 0.0
        return sorted_points[0][1]

    def _engagement_rate(self, metrics: VideoMetrics) -> float:
        """Calculate engagement rate: (likes + comments + shares) / views * 100."""
        if metrics.views == 0:
            return 0.0
        return ((metrics.likes + metrics.comments + metrics.shares) / metrics.views) * 100

    def _score_ratio(self, actual: float, threshold: float, baseline: float) -> float:
        """
        Score a metric on a 0-100 scale:
          - meets threshold: >= 100
          - exceeds baseline: proportional between 50-100
          - below baseline: below 50
        """
        if threshold <= 0:
            return 100.0 if actual > 0 else 0.0

        # Normalize: 0 = 0% of threshold, 100 = 100%+ of threshold
        ratio = actual / threshold if threshold > 0 else 0
        raw_score = ratio * 100

        # Apply baseline weighting: if above baseline, boost slightly
        if baseline > 0:
            baseline_ratio = actual / baseline if actual > baseline else 0
            # Blend: 70% threshold-based, 30% baseline-relative
            raw_score = raw_score * 0.7 + max(0, baseline_ratio * 70) * 0.3

        return min(100.0, max(0.0, raw_score))

    def _build_score_summary(self, metrics: VideoMetrics, scores: Dict[str, float], passed: bool) -> str:
        """Build a human-readable summary of the virality evaluation."""
        status = "PASS" if passed else "FAIL"
        overall = scores.get("overall", 0)
        return (
            f"[{status}] Video '{metrics.video_id[:11]}...' | "
            f"Overall: {overall:.0f}/100 | "
            f"CTR: {scores.get('ctr', 0):.0f} | "
            f"AVD: {scores.get('avg_view_duration', 0):.0f} | "
            f"15s-ret: {scores.get('first_15s_retention', 0):.0f} | "
            f"Engagement: {scores.get('engagement_rate', 0):.0f} | "
            f"WatchTime: {scores.get('watch_time', 0):.0f}"
        )

    # ------------------------------------------------------------------
    # Historical analysis
    # ------------------------------------------------------------------

    def record_performance(self, metrics: VideoMetrics, score: ViralityScore) -> None:
        """Record a video's performance for future baseline computation."""
        entry = {
            "video_id": metrics.video_id,
            "title": metrics.title,
            "is_short": metrics.is_short,
            "views": metrics.views,
            "likes": metrics.likes,
            "comments": metrics.comments,
            "shares": metrics.shares,
            "subscribers_gained": metrics.subscribers_gained,
            "estimated_minutes_watched": metrics.estimated_minutes_watched,
            "average_view_duration_seconds": metrics.average_view_duration_seconds,
            "avd_ratio": metrics.average_view_duration_seconds / max(metrics.duration_seconds, 1),
            "ctr": metrics.click_through_rate_percent,
            "engagement": self._engagement_rate(metrics),
            "first_15s_retention": self._retention_at(metrics, 15),
            "first_30s_retention": self._retention_at(metrics, 30),
            "tags_used": metrics.tags_used,
            "title_used": metrics.title_used,
            "thumbnail_text": metrics.thumbnail_text,
            "overall_virality_score": score.overall_score,
            "passed": score.passed,
            "published_at": metrics.published_at,
        }
        self.history.append(entry)
        self._save_history()
        log(f"Recorded performance for '{metrics.title[:40]}...' (score={score.overall_score:.0f}/100)")

    def analyze_historical_patterns(self) -> dict:
        """
        Analyze historical data to find what makes content go viral.
        Returns a patterns dict with actionable insights.
        """
        if not self.history:
            return {"insights": ["No historical data yet. Run more videos to build patterns."],
                    "recommendations": []}

        recent = sorted(
            self.history, key=lambda h: h.get("published_at", ""), reverse=True
        )[:30]

        passed = [h for h in recent if h.get("passed", False)]
        failed = [h for h in recent if not h.get("passed", False)]

        insights = []
        recommendations = []

        # CTR analysis
        passed_ctrs = [h.get("ctr", 0) for h in passed if h.get("views", 0) > 100]
        failed_ctrs = [h.get("ctr", 0) for h in failed if h.get("views", 0) > 100]
        if passed_ctrs and failed_ctrs:
            avg_passed = sum(passed_ctrs) / len(passed_ctrs)
            avg_failed = sum(failed_ctrs) / len(failed_ctrs)
            insights.append(
                f"Viral videos average {avg_passed:.1f}% CTR vs {avg_failed:.1f}% for underperformers."
            )
            recommendations.append(
                "Prioritize high-CTR titles: curiosity-gap hooks + primary keyword first."
            )

        # Retention analysis
        passed_15s = [h.get("first_15s_retention", 0) for h in passed]
        failed_15s = [h.get("first_15s_retention", 0) for h in failed]
        if passed_15s and failed_15s:
            avg_pass = sum(passed_15s) / len(passed_15s)
            avg_fail = sum(failed_15s) / len(failed_15s)
            insights.append(
                f"Videos with 15s retention above {avg_pass:.0f}% go viral; "
                f"below {avg_fail:.0f}% they underperform."
            )
            recommendations.append(
                "Apply WTF formula + shocking fact intro + visual change every 1-2s in first 10s."
            )

        # Thumbnail text analysis
        thumb_texts = [h.get("thumbnail_text", "") for h in passed if h.get("views", 0) > 100]
        if thumb_texts:
            from collections import Counter
            word_freq = Counter()
            for text in thumb_texts:
                for word in text.split():
                    word_freq[word] += 1
            top_words = [w for w, _ in word_freq.most_common(5)]
            insights.append(f"Highest-CTR thumbnail words: {', '.join(top_words)}.")

        # Tag analysis
        all_tags = [tag for h in passed for tag in h.get("tags_used", [])]
        from collections import Counter
        tag_freq = Counter(all_tags)
        top_tags = [t for t, _ in tag_freq.most_common(10)]
        insights.append(f"Most common tags in viral videos: {', '.join(top_tags)}.")

        # Engagement analysis
        passed_eng = [h.get("engagement", 0) for h in passed]
        if passed_eng:
            avg_eng = sum(passed_eng) / len(passed_eng)
            insights.append(f"Average engagement rate for viral videos: {avg_eng:.1f}%.")
            recommendations.append(
                "Boost engagement: add CTAs, pose questions, encourage comments."
            )

        return {
            "total_videos_analyzed": len(recent),
            "viral_count": len(passed),
            "underperforming_count": len(failed),
            "insights": insights,
            "recommendations": recommendations,
        }

    def get_performance_context(self) -> dict:
        """
        Return a context dict summarizing historical performance, ready to be
        passed to the SEO optimizer and script generator for recursive
        improvement.
        """
        patterns = self.analyze_historical_patterns()
        recent = self.history[-5:] if self.history else []

        past_titles = [h["title_used"] for h in recent if h.get("title_used")]
        high_performing_tags = []
        if recent:
            passed = [h for h in recent if h.get("passed")]
            high_performers = passed if passed else recent
            tag_counts: Dict[str, int] = {}
            for h in high_performers:
                for tag in h.get("tags_used", []):
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            high_performing_tags = sorted(tag_counts, key=lambda k: tag_counts[k], reverse=True)[:10]

        best_thumb = ""
        if past_titles:
            titles_with_high_ctr = [h for h in recent if h.get("ctr", 0) >= 5.0]
            if titles_with_high_ctr:
                best_thumb = titles_with_high_ctr[0].get("thumbnail_text", "")

        return {
            "past_titles_with_high_ctr": past_titles,
            "high_performing_tags": high_performing_tags,
            "best_thumbnail_text": best_thumb,
            "avg_ctr_lift": sum(h.get("ctr", 0) for h in recent) / max(len(recent), 1),
            "patterns": patterns,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_evaluator: ViralityEvaluator | None = None


def get_evaluator() -> ViralityEvaluator:
    """Singleton evaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = ViralityEvaluator()
    return _evaluator
