"""
Automated Optimization Engine / Recursive Learning Algorithm.

This module implements the recursive learning loop that:

  1. Evaluates whether a generated Short or regular video satisfies YouTube
     virality metrics (via ViralityEvaluator).
  2. Tracks viewership trends to determine what makes content go viral.
  3. Feeds those insights back into the system, ensuring every new video
     automatically improves based on previous performance data.
  4. If a Short or regular video underperforms, executes a feedback loop that:
     a. Analyzes the failure against historical data
     b. Identifies which specific elements (title, thumbnail, hook, pacing,
        tags, spoken keywords) were weak
     c. Dynamically applies corrective optimizations to the next workflow run

The engine persists a learning state file (work/optimization_state.json) that
accumulates insights across every run, and exposes a `get_optimizations()`
API that the script generator and SEO optimizer call to bias their output
toward historically-successful patterns.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

from virality_metrics import ViralityEvaluator, VideoMetrics, ViralityScore, get_evaluator
from seo_optimizer import SEOOptimizer


LOG_PREFIX = "[optimization_engine]"


def log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", flush=True)


# ---------------------------------------------------------------------------
# Learning state schema
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 70.0  # overall virality score to be considered "passing"
UNDERPERFORMANCE_THRESHOLD = 40.0  # below this = major corrective action needed


@dataclass
class OptimizerConfig:
    """Configuration for the optimization engine."""
    state_file: Path = Path("work/optimization_state.json")
    history_days: int = 30  # how many days of history to retain
    confidence_threshold: float = 0.8  # minimum confidence to apply a learned optimization
    min_samples_for_pattern: int = 5  # minimum videos before a pattern is trusted

    # Thresholds
    pass_threshold: float = DEFAULT_THRESHOLD
    fail_threshold: float = UNDERPERFORMANCE_THRESHOLD

    # Weights for blending historical insight into new generations
    historical_weight: float = 0.3  # how much past success influences new output


class OptimizationEngine:
    """
    The recursive learning algorithm.

    Lifecycle:
      1. evaluate(video_metrics) → records the score + stores performance
      2. learn_from_performance() → analyzes patterns, updates optimization map
      3. get_optimizations(topic, is_short) → returns bias directives for the
         next script/SEO/template generation
      4. Those directives are fed into regular_video_prompt, seo_optimizer,
         and the shorts pipeline
    """

    def __init__(self, config: Optional[OptimizerConfig] = None):
        self.config = config or OptimizerConfig()
        self.config.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._evaluator: ViralityEvaluator | None = None
        self._state: dict = {}
        self._load_state()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        """Load the persistent optimization state from disk."""
        if not self.config.state_file.exists():
            self._state = {
                "initialized_at": time.time(),
                "total_videos_evaluated": 0,
                "passed_count": 0,
                "failed_count": 0,
                "patterns": {},
                "element_effectiveness": {
                    "titles": {},
                    "thumbnails": {},
                    "hooks": {},
                    "tags": {},
                    "spoken_keywords": {},
                    "captions": {},
                    "transitions": {},
                    "templates": {},
                },
                "last_evaluated": {},
                "recommendations": [],
            }
            self._save_state()
            return

        try:
            with open(self.config.state_file, "r") as f:
                self._state = json.load(f)
            log(f"Loaded optimization state ({self._state.get('total_videos_evaluated', 0)} videos evaluated).")
        except (json.JSONDecodeError, OSError) as exc:
            log(f"State file corrupted ({exc}) — starting fresh.")
            self._state = {
                "initialized_at": time.time(),
                "total_videos_evaluated": 0,
                "passed_count": 0,
                "failed_count": 0,
                "patterns": {},
                "element_effectiveness": {
                    "titles": {},
                    "thumbnails": {},
                    "hooks": {},
                    "tags": {},
                    "spoken_keywords": {},
                    "captions": {},
                    "transitions": {},
                    "templates": {},
                },
                "last_evaluated": {},
                "recommendations": [],
            }

    def _save_state(self) -> None:
        """Persist the optimization state to disk."""
        self._state["last_updated"] = time.time()
        try:
            with open(self.config.state_file, "w") as f:
                json.dump(self._state, f, indent=2, default=str)
        except OSError as exc:
            log(f"Could not save optimization state ({exc}).")

    def _get_evaluator(self) -> ViralityEvaluator:
        if self._evaluator is None:
            self._evaluator = get_evaluator()
        return self._evaluator

    # ------------------------------------------------------------------
    # Core evaluation + learning
    # ------------------------------------------------------------------

    def evaluate_and_learn(self, metrics: VideoMetrics) -> ViralityScore:
        """
        The main entry point for the feedback loop:

        1. Evaluate the video against virality metrics
        2. Record performance in the virality history
        3. Run pattern analysis and update the optimization state
        4. Return the ViralityScore (and side-effect: state is updated)

        This is called after a video's YouTube analytics are available
        (typically 24-48 hours after upload).
        """
        evaluator = self._get_evaluator()
        score = evaluator.evaluate(metrics)

        # Record in the virality history (used for baseline computation)
        evaluator.record_performance(metrics, score)

        # Update the optimization state
        self._state["total_videos_evaluated"] = self._state.get("total_videos_evaluated", 0) + 1
        if score.passed:
            self._state["passed_count"] = self._state.get("passed_count", 0) + 1
        else:
            self._state["failed_count"] = self._state.get("failed_count", 0) + 1

        self._state["last_evaluated"] = {
            "video_id": metrics.video_id,
            "title": metrics.title,
            "score": score.overall_score,
            "passed": score.passed,
            "evaluated_at": time.time(),
        }

        # Trace each element's effectiveness
        self._trace_element_effectiveness(metrics, score)

        # Run the recursive learning analysis
        self._learn_from_performance(metrics, score)

        # Persist
        self._save_state()

        log(f"Evaluation complete: {score.summary}")
        if not score.passed:
            log(f"Underperforming video detected — corrective actions queued for next run.")
            for action in score.corrective_actions:
                log(f"  → {action}")

        return score

    def _trace_element_effectiveness(self, metrics: VideoMetrics, score: ViralityScore) -> None:
        """
        Trace which specific creative elements (title keywords, thumbnail text,
        tags, hooks, spoken keywords) correlated with high vs low scores.
        Updates the element_effectiveness map for future bias.
        """
        elements = self._state.setdefault("element_effectiveness", {})

        # Title pattern
        title_used = metrics.title_used or metrics.title
        # Extract the hook pattern (first 5 words)
        hook_pattern = " ".join(title_used.split()[:5]).lower()
        self._update_element_score(elements.setdefault("titles", {}), hook_pattern, score.overall_score)

        # Thumbnail text
        thumb_text = metrics.thumbnail_text or ""
        if thumb_text:
            self._update_element_score(elements.setdefault("thumbnails", {}), thumb_text, score.overall_score)

        # Captions
        caption_used = getattr(metrics, "caption_used", "")
        if caption_used:
            self._update_element_score(elements.setdefault("captions", {}), caption_used, score.overall_score)

        # Tags
        for tag in metrics.tags_used:
            self._update_element_score(elements.setdefault("tags", {}), tag, score.overall_score)

        # Spoken keywords
        for kw in metrics.spoken_keywords:
            self._update_element_score(elements.setdefault("spoken_keywords", {}), kw, score.overall_score)

    def _update_element_score(self, element_map: dict, key: str, score: float) -> None:
        """Update the running average score for a creative element."""
        entry = element_map.get(key, {"count": 0, "total": 0.0, "avg": 0.0})
        entry["count"] += 1
        entry["total"] += score
        entry["avg"] = entry["total"] / entry["count"]
        element_map[key] = entry

    def _learn_from_performance(self, metrics: VideoMetrics, score: ViralityScore) -> None:
        """
        Analyze the video's performance and extract optimization patterns.
        This is the "recursive" part: insights from this run inform the next.
        """
        patterns = self._state.setdefault("patterns", {})

        # If the video passed, reinforce the patterns that were used
        if score.passed:
            self._reinforce_patterns(metrics, patterns)
        else:
            self._generate_failure_analysis(metrics, score, patterns)

        # Generate forward-looking recommendations
        recommendations = self._generate_recommendations(patterns)
        self._state["recommendations"] = recommendations

    def _reinforce_patterns(self, metrics: VideoMetrics, patterns: dict) -> None:
        """When a video passes, reinforce the elements that contributed to success."""
        # Track which title patterns consistently pass
        title_used = metrics.title_used or metrics.title
        hook = " ".join(title_used.split()[:5]).lower()

        hook_stats = patterns.setdefault("successful_title_hooks", {})
        entry = hook_stats.get(hook, {"uses": 0, "passes": 0})
        entry["uses"] += 1
        if True:  # this video passed
            entry["passes"] += 1
        hook_stats[hook] = entry

        # Track successful thumbnail words
        thumb_text = metrics.thumbnail_text or ""
        for word in thumb_text.split():
            word = word.upper()
            word_stats = patterns.setdefault("successful_thumbnail_words", {})
            entry = word_stats.get(word, {"uses": 0, "passes": 0})
            entry["uses"] += 1
            entry["passes"] += 1
            word_stats[word] = entry

        # Track successful tags
        tag_stats = patterns.setdefault("successful_tags", {})
        for tag in metrics.tags_used:
            entry = tag_stats.get(tag, {"uses": 0, "passes": 0})
            entry["uses"] += 1
            entry["passes"] += 1
            tag_stats[tag] = entry

        # Retention curve analysis
        if metrics.retention_points:
            # Identify the drop-off point
            sorted_retention = sorted(metrics.retention_points.items())
            sharp_drops = []
            for i in range(1, len(sorted_retention)):
                prev_t, prev_r = sorted_retention[i - 1]
                curr_t, curr_r = sorted_retention[i]
                if prev_r - curr_r > 10:  # >10% drop = sharp
                    sharp_drops.append({"at_second": curr_t, "drop_percent": prev_r - curr_r})
            patterns.setdefault("retention_sharp_drops", []).extend(sharp_drops)

        log(f"Reinforced {len(title_used.split()[:5])} title hooks, "
            f"{len(metrics.tags_used)} tags from successful video.")

    def _generate_failure_analysis(self, metrics: VideoMetrics, score: ViralityScore, patterns: dict) -> None:
        """Analyze why a video underperformed and queue corrective optimizations."""
        failure_analysis = patterns.setdefault("failure_analysis", [])

        analysis = {
            "video_id": metrics.video_id,
            "title": metrics.title,
            "overall_score": score.overall_score,
            "failed_components": [
                {"component": k, "score": v, "corrective_action": a}
                for k, v in score.component_scores.items()
                for a in score.corrective_actions
                if k in a.lower()
            ],
            "timestamp": time.time(),
        }
        failure_analysis.append(analysis)
        if len(failure_analysis) > 50:  # cap the list
            failure_analysis.pop(0)

        log(f"Failure analysis recorded for '{metrics.title[:40]}...' "
            f"(score={score.overall_score:.0f}/100)")

        # Generate specific corrective optimizations
        corrective_optimizations = patterns.setdefault("corrective_optimizations", {})
        for action in score.corrective_actions:
            # Extract the metric category from the action text
            if "CTR" in action:
                key = "title"
            elif "retention" in action.lower() or "AVD" in action:
                key = "hook"
            elif "engagement" in action:
                key = "caption"
            elif "thumbnail" in action.lower():
                key = "thumbnail"
            elif "tag" in action.lower():
                key = "tags"
            else:
                key = "general"

            if key not in corrective_optimizations:
                corrective_optimizations[key] = []
            if action not in corrective_optimizations[key]:
                corrective_optimizations[key].append(action)

    def _generate_recommendations(self, patterns: dict) -> list:
        """Generate forward-looking optimization recommendations."""
        recommendations = []

        # Analyze successful title hooks
        hook_stats = patterns.get("successful_title_hooks", {})
        if hook_stats:
            best_hook = max(hook_stats.items(), key=lambda x: x[1]["passes"] / max(x[1]["uses"], 1))
            confidence = best_hook[1]["passes"] / max(best_hook[1]["uses"], 1)
            if confidence >= 0.5 and best_hook[1]["uses"] >= self.config.min_samples_for_pattern:
                recommendations.append({
                    "type": "title",
                    "insight": f"Title pattern '{best_hook[0]}' has {confidence:.0%} success rate",
                    "action": "Bias future title generation toward this pattern",
                })

        # Analyze successful thumbnail words
        word_stats = patterns.get("successful_thumbnail_words", {})
        if word_stats:
            best_words = sorted(
                word_stats.items(),
                key=lambda x: x[1]["passes"] / max(x[1]["uses"], 1),
                reverse=True,
            )
            for word, stats in best_words[:3]:
                confidence = stats["passes"] / max(stats["uses"], 1)
                if confidence >= 0.6 and stats["uses"] >= self.config.min_samples_for_pattern:
                    recommendations.append({
                        "type": "thumbnail",
                        "insight": f"Thumbnail word '{word}' has {confidence:.0%} success rate",
                        "action": f"Prioritize '{word}' in future thumbnail text",
                    })

        # Analyze retention patterns
        drops = patterns.get("retention_sharp_drops", [])
        if drops:
            common_drop_times = [d["at_second"] for d in drops if d["at_second"] > 0]
            if common_drop_times:
                from collections import Counter
                drop_counts = Counter(int(t // 15) * 15 for t in common_drop_times)
                most_common_drop = drop_counts.most_common(1)[0]
                recommendations.append({
                    "type": "retention",
                    "insight": f"Most videos drop off at ~{most_common_drop[0]}s",
                    "action": "Add micro-hook before this point in future videos",
                })

        return recommendations

    # ------------------------------------------------------------------
    # Public API: get optimizations for the next run
    # ------------------------------------------------------------------

    def get_optimizations(self, topic: str, is_short: bool = True) -> dict:
        """
        Return optimization directives for the next video generation run.

        These directives are consumed by:
          - regular_video_prompt.py (bias the prompt toward proven hooks)
          - seo_optimizer.py (bias title/thumbnail/tags toward high-performers)
          - viral_templates.py (bias template selection for shorts)

        Returns a dict with keys:
          - title_pattern_bias: list of successful title patterns to prefer
          - thumbnail_words_bias: list of high-CTR words for thumbnails
          - tag_bias: list of high-performing tags to include
          - hook_bias: list of successful hook approaches
          - recommended_transitions: list of transitions that retained viewers
          - recommended_templates: list of templates that performed well
          - corrective_actions: list of specific fixes from recent failures
          - historical_confidence: float 0-1 indicating how much to trust patterns
        """
        patterns = self._state.get("patterns", {})
        elements = self._state.get("element_effectiveness", {})

        # Title pattern bias: top-performing title patterns by average score
        title_scores = elements.get("titles", {})
        title_bias = sorted(title_scores.items(), key=lambda x: x[1]["avg"], reverse=True)[:5]
        title_pattern_bias = [t[0] for t in title_bias if t[1]["count"] >= 2]

        # Thumbnail word bias
        thumb_scores = elements.get("thumbnails", {})
        thumb_bias = sorted(thumb_scores.items(), key=lambda x: x[1]["avg"], reverse=True)[:10]
        thumbnail_words_bias = [t[0] for t in thumb_bias if t[1]["count"] >= 2]

        # Tag bias
        tag_scores = elements.get("tags", {})
        tag_bias = sorted(tag_scores.items(), key=lambda x: x[1]["avg"], reverse=True)[:15]
        tag_bias_list = [t[0] for t in tag_bias if t[1]["count"] >= 2]

        # Hook bias (from title patterns, first 3 words)
        hook_scores = {}
        for title, stats in title_scores.items():
            hook = " ".join(title.split()[:3])
            entry = hook_scores.setdefault(hook, {"count": 0, "total": 0})
            entry["count"] = stats["count"]
            entry["total"] += stats["total"]
        hook_bias_list = sorted(
            [(h, s["total"] / s["count"]) for h, s in hook_scores.items() if s["count"] >= 2],
            key=lambda x: x[1], reverse=True,
        )[:5]
        hook_bias = [h[0] for h in hook_bias_list]

        # Recommended transitions
        # (These would be populated from retention analysis if transition
        # tracking was added to VideoMetrics)

        # Corrective actions from recent failures
        corrective_optimizations = patterns.get("corrective_optimizations", {})
        recent_corrective = []
        for category, actions in corrective_optimizations.items():
            recent_corrective.extend(actions[:3])  # top 3 per category

        # Historical confidence
        total = self._state.get("total_videos_evaluated", 0)
        confidence = min(1.0, total / 20.0)  # reach full confidence at 20 videos

        return {
            "title_pattern_bias": title_pattern_bias,
            "thumbnail_words_bias": thumbnail_words_bias,
            "tag_bias": tag_bias_list,
            "hook_bias": hook_bias,
            "recommended_transitions": [],
            "recommended_templates": [],
            "corrective_actions": recent_corrective[-10:],  # last 10 corrective actions
            "historical_confidence": round(confidence, 2),
            "total_videos_evaluated": total,
            "pass_rate": round(
                self._state.get("passed_count", 0) / max(total, 1), 2
            ),
            "recent_recommendations": self._state.get("recommendations", []),
        }

    def get_performance_summary(self) -> dict:
        """Return a summary of the optimization engine's state."""
        total = self._state.get("total_videos_evaluated", 0)
        passed = self._state.get("passed_count", 0)
        failed = self._state.get("failed_count", 0)
        return {
            "total_videos_evaluated": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / max(total, 1), 2),
            "state_file": str(self.config.state_file),
            "last_evaluated": self._state.get("last_evaluated", {}),
            "recommendations": self._state.get("recommendations", []),
        }

    # ------------------------------------------------------------------
    # YouTube Analytics integration
    # ------------------------------------------------------------------

    def fetch_youtube_analytics(
        self,
        video_id: str,
        yt_client,
        publish_date: str,
    ) -> VideoMetrics:
        """
        Fetch analytics data for a video from the YouTube Data API v3 and
        return a VideoMetrics object. This bridges the gap between the
        optimization engine and YouTube's actual data.
        """
        # Fetch video metadata
        video_resp = yt_client.videos().list(
            part="snippet,contentDetails,statistics,status",
            id=video_id,
        ).execute()

        if not video_resp.get("items"):
            raise ValueError(f"Video {video_id} not found on YouTube.")

        item = video_resp["items"][0]
        stats = item.get("statistics", {})
        snippet = item.get("snippet", {})
        content_details = item.get("contentDetails", {})

        duration_iso = content_details.get("duration", "PT0S")
        duration_seconds = self._parse_iso_duration(duration_iso)

        # Fetch analytics (requires youtubeAnalytics scope)
        # We try to get retention data separately
        retention_points = {}
        try:
            analytics = yt_client (
                # youtubeAnalytics endpoint — requires the analytics scope
                # This is best-effort; analytics may need separate auth
            ).execute()
            # Parse retention points from analytics response
            # ... (would be populated from the actual API response)
        except Exception:
            pass

        is_short = bool(snippet.get("tags", [])) and any(
            "shorts" in tag.lower() for tag in (snippet.get("tags", []))
        )

        metrics = VideoMetrics(
            video_id=video_id,
            title=snippet.get("title", ""),
            is_short=is_short,
            duration_seconds=duration_seconds,
            views=int(stats.get("viewCount", 0)),
            likes=int(stats.get("likeCount", 0)),
            comments=int(stats.get("commentCount", 0)),
            shares=int(stats.get("shareCount", 0)) if "shareCount" in stats else 0,
            subscribers_gained=0,  # requires analytics API
            estimated_minutes_watched=float(stats.get("viewCount", 0)) * 0.5 / 60,  # estimate
            average_view_duration_seconds=duration_seconds * 0.6,  # estimate
            click_through_rate_percent=0.0,  # requires analytics
            traffic_source_organic_percent=0.0,
            retention_points=retention_points,
            published_at=snippet.get("publishedAt", ""),
        )

        return metrics

    @staticmethod
    def _parse_iso_duration(iso_duration: str) -> float:
        """Parse ISO 8601 duration (e.g. 'PT5M30S') to seconds."""
        import re
        match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso_duration)
        if not match:
            return 0.0
        hours, minutes, seconds = match.groups("0")
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine: OptimizationEngine | None = None


def get_engine() -> OptimizationEngine:
    """Get or create the singleton optimization engine."""
    global _engine
    if _engine is None:
        _engine = OptimizationEngine()
    return _engine
