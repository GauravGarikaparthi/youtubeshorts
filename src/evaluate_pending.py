"""
Evaluate pending Shorts and long-form videos from the optimization engine.

This script is designed to be run in CI as a separate step or cron job.
It reads work/short_uploads.json and work/longform_uploads.json, evaluates
each video that hasn't been evaluated yet, and feeds the results into the
recursive learning engine.

In a production environment with real YouTube Analytics, this would fetch
actual CTR, retention, watch time, and engagement data via the YouTube
Data API v3 + YouTube Analytics API. Currently it evaluates with the
metrics available at upload time and marks videos as "pending real analytics".
"""

from __future__ import annotations

import sys
import os
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimization_engine import get_engine
from virality_metrics import VideoMetrics, get_evaluator
from upload_youtube import _get_credentials
from googleapiclient.discovery import build


def _try_fetch_youtube_analytics(video_id: str, record: dict) -> VideoMetrics | None:
    """Try to fetch real YouTube analytics. Returns None on failure."""
    try:
        creds = _get_credentials()
        yt = build("youtube", "v3", credentials=creds)
        analytics = build("youtubeanalytics", "v2", credentials=creds)

        # Get video stats
        video_data = yt.videos().list(
            part="snippet,contentDetails,statistics,status",
            id=video_id,
        ).execute()

        if not video_data.get("items"):
            return None

        item = video_data["items"][0]
        stats = item.get("statistics", {})
        snippet = item.get("snippet", {})
        cd = item.get("contentDetails", {})

        # Parse duration
        duration_str = cd.get("duration", "PT0S")
        duration_seconds = _parse_iso_duration(duration_str)

        # Get retention
        retention_points = {}
        try:
            retention_resp = analytics.videos().getVideoRetention(
                ids=video_id,
            ).execute()
            for point in retention_resp.get("retention", []):
                retention_points[float(point["elapsedSeconds"])] = float(point["percent"])
        except Exception:
            pass

        # Get analytics: CTR, watch time, engagement
        today = datetime.utcnow().date()
        seven_days_ago = today - __import__("datetime").timedelta(days=7)

        ctr = 0.0
        watch_time_min = 0.0
        try:
            report = analytics.reports().get(
                ids="channel==MINE",
                startDate=seven_days_ago.strftime("%Y-%m-%d"),
                endDate=today.strftime("%Y-%m-%d"),
                metrics="views,estimatedMinutesWatched,averageViewDuration,clickThroughRate",
                filters=f"video=={video_id}",
            ).execute()
            for row in report.get("row", {}).get("values", [{}]):
                ctr = float(row.get("clickThroughRate", 0))
                watch_time_min = float(row.get("estimatedMinutesWatched", 0))
        except Exception:
            pass

        is_short = "shorts" in (
            (snippet.get("title", "") + " " + " ".join(snippet.get("tags", []))).lower()
        )

        return VideoMetrics(
            video_id=video_id,
            title=record.get("title", ""),
            is_short=is_short,
            duration_seconds=duration_seconds,
            views=int(stats.get("viewCount", 0)),
            likes=int(stats.get("likeCount", 0)),
            comments=int(stats.get("commentCount", 0)),
            shares=0,
            subscribers_gained=int(stats.get("newSubscriberCount", 0)) if "newSubscriberCount" in stats else 0,
            estimated_minutes_watched=watch_time_min,
            average_view_duration_seconds=float(stats.get("averageViewDuration", 0) or 0),
            click_through_rate_percent=ctr,
            traffic_source_organic_percent=0.0,
            retention_points=retention_points,
            tags_used=record.get("tags", []),
            title_used=record.get("title", ""),
            description_used=record.get("description_used", ""),
            thumbnail_text=record.get("thumbnail_text", ""),
            spoken_keywords=record.get("spoken_keywords", []),
            published_at=snippet.get("publishedAt", ""),
        )
    except Exception:
        return None


def _fall_back_metrics(video_id: str, record: dict) -> VideoMetrics:
    """Create metrics from upload record when YouTube API is unavailable."""
    is_short = record.get("is_short", not "longform" in record.get("video_path", ""))
    duration = 55.0 if is_short else 600.0

    return VideoMetrics(
        video_id=video_id,
        title=record.get("title", ""),
        is_short=is_short,
        duration_seconds=duration,
        views=int(record.get("mock_views", 0)),
        likes=0,
        comments=0,
        shares=0,
        subscribers_gained=0,
        estimated_minutes_watched=0.0,
        average_view_duration_seconds=0.0,
        click_through_rate_percent=0.0,
        retention_points={},
        tags_used=record.get("tags", []),
        title_used=record.get("title", ""),
        description_used=record.get("description_used", ""),
        thumbnail_text=record.get("thumbnail_text", ""),
        spoken_keywords=record.get("spoken_keywords", []),
        published_at=record.get("uploaded_at", ""),
    )


def _parse_iso_duration(iso: str) -> float:
    import re
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "PT0S")
    if not match:
        return 0.0
    h, m, s = match.groups("0")
    return int(h) * 3600 + int(m) * 60 + int(s)


def evaluate_pending_videos() -> None:
    """
    Check all uploaded videos (both Shorts and long-form) that haven't been
    evaluated yet, fetch their analytics, and feed results into the
    optimization engine.
    """
    engine = get_engine()
    evaluator = get_evaluator()

    total_evaluated = 0

    for record_file in ["work/shorts_uploads.json", "work/longform_uploads.json"]:
        record_path = Path(record_file)
        if not record_path.exists():
            continue

        try:
            with open(record_path, "r") as f:
                records = json.load(f)
        except (json.JSONDecodeError, OSError):
            log(f"Could not read {record_path} — skipping.")
            continue

        pending = [r for r in records if r.get("status") == "uploaded_pending_analytics"]
        if not pending:
            continue

        log(f"Found {len(pending)} pending videos in {record_file}.")

        # Try to get real YouTube credentials
        yt_available = False
        try:
            _get_credentials()
            yt_available = True
        except Exception:
            log("YouTube credentials not available — using available data.")

        for record in pending:
            video_id = record["video_id"]
            log(f"Evaluating: {record.get('title', '')[:50]}...")

            if yt_available:
                metrics = _try_fetch_youtube_analytics(video_id, record)
                if metrics is None:
                    log(f"  Could not fetch analytics for {video_id} — using fallback.")
                    metrics = _fall_back_metrics(video_id, record)
            else:
                metrics = _fall_back_metrics(video_id, record)

            score = engine.evaluate_and_learn(metrics)

            record["status"] = "evaluated"
            record["evaluation"] = score.to_dict()
            record["evaluated_at"] = datetime.utcnow().isoformat()
            total_evaluated += 1

        # Save updated records
        with open(record_path, "w") as f:
            json.dump(records, f, indent=2, default=str)

    if total_evaluated > 0:
        log(f"Evaluated {total_evaluated} videos. Optimization engine updated.")
        summary = engine.get_performance_summary()
        log(f"Performance summary: {summary['total_videos_evaluated']} videos, "
            f"pass rate: {summary['pass_rate']}")
    else:
        log("No pending videos to evaluate.")


if __name__ == "__main__":
    evaluate_pending_videos()
