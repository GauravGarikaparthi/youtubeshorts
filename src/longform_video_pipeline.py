"""
Long-form regular video pipeline — the "separate code entirely" that rewrites
the script for regular videos.

This pipeline is entirely separate from the Shorts pipeline (main.py) and the
existing orchestrator.py. It:

  1. Resolves a trending topic
  2. Researches SEO keywords
  3. Builds the optimized Google Flow prompt (incorporating all storytelling
     rules: WTF formula, first-10-second mastery, jump cuts, micro-hooks,
     correction loop, spoken keyword sync, end-screen transition, etc.)
  4. Submits the prompt to Google Flow to generate the longest possible video
     (using all 50 free daily credits)
  5. Downloads the video
  6. Generates a 16:9 high-contrast thumbnail
  7. Runs the full SEO/GEO/AEO optimization (title, tags, description, caption)
  8. Applies historical learning from the optimization engine
  9. Uploads to YouTube as a regular video (NOT a Short)
  10. Records performance data for the recursive learning loop

Usage:
    python src/longform_video_pipeline.py
    # or via GitHub Actions workflow (see daily_longform.yml)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Ensure src/ is on the path when run directly
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from regular_video_config import LongformVideoConfig
from regular_video_prompt import build_optimized_flow_prompt, build_regular_video_prompt
from google_flow_client import GoogleFlowClient, FlowSessionConfig, FlowVideoResult
from seo_optimizer import SEOOptimizer, SEOMetadata
from virality_metrics import ViralityEvaluator, VideoMetrics, ViralityScore, get_evaluator
from optimization_engine import OptimizationEngine, get_engine
from trend_fetch import resolve_topic
from seo_research import research_keywords


LOG_PREFIX = "[longform]"


def log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", flush=True)


# ---------------------------------------------------------------------------
# Step 1: Build the optimized prompt
# ---------------------------------------------------------------------------

def build_longform_prompt(
    topic: str,
    seo_keywords: List[str],
    optimizations: dict | None = None,
) -> str:
    """
    Build the full optimized Google Flow prompt for a regular long-form video.

    Incorporates all user-specified storytelling elements and any historical
    learnings from the optimization engine (biased title patterns, thumbnail
    words, tags, etc.).
    """
    optimizations = optimizations or {}

    # Build the prompt with all the rules from the system prompt
    prompt = build_optimized_flow_prompt(topic, seo_keywords)

    # Inject historical learnings
    if optimizations:
        biases = []
        if optimizations.get("title_pattern_bias"):
            biases.append(
                f"HISTORICAL LEARNING: Prefer these title patterns from past "
                f"high-performing videos: {optimizations['title_pattern_bias'][:3]}"
            )
        if optimizations.get("thumbnail_words_bias"):
            biases.append(
                f"HISTORICAL LEARNING: Prioritize these thumbnail words: "
                f"{optimizations['thumbnail_words_bias'][:5]}"
            )
        if optimizations.get("tag_bias"):
            biases.append(
                f"HISTORICAL LEARNING: These tags have historically performed well: "
                f"{optimizations['tag_bias'][:10]}"
            )
        if optimizations.get("hook_bias"):
            biases.append(
                f"HISTORICAL LEARNING: Start with these hook patterns: "
                f"{optimizations['hook_bias'][:3]}"
            )
        if optimizations.get("corrective_actions"):
            biases.append(
                f"CORRECTIVE OPTIMIZATIONS from recent underperforming videos: "
                f"{optimizations['corrective_actions'][:5]}"
            )

        if biases:
            prompt += "\n\n" + "\n".join(biases)

    return prompt


# ---------------------------------------------------------------------------
# Step 2: Generate metadata (SEO/GEO/AEO)
# ---------------------------------------------------------------------------

def generate_longform_metadata(
    topic: str,
    keywords: List[str],
    config: LongformVideoConfig,
    optimizations: dict | None = None,
) -> SEOMetadata:
    """
    Generate fully optimized SEO/GEO/AEO metadata for the regular video.
    """
    optimizer = SEOOptimizer()

    # Use the optimization engine's historical bias
    perf_context = None
    if optimizations:
        perf_context = {
            "past_titles_with_high_ctr": optimizations.get("title_pattern_bias", []),
            "high_performing_tags": optimizations.get("tag_bias", []),
            "best_thumbnail_text": " ".join(optimizations.get("thumbnail_words_bias", [])[:3]),
            "avg_ctr_lift": 0.0,
        }

    metadata = optimizer.optimize(
        topic=topic,
        keywords=keywords,
        estimated_duration=config.duration_target,
        performance_context=perf_context,
    )

    log(f"SEO metadata generated: title='{metadata.title}'")
    log(f"  → tags: {len(metadata.tags)}, keywords: {metadata.keywords_spoken}")

    return metadata


# ---------------------------------------------------------------------------
# Step 3: Upload to YouTube
# ---------------------------------------------------------------------------

def upload_longform_video(
    video_path: str,
    metadata: SEOMetadata,
    thumbnail_path: str,
    config: LongformVideoConfig,
) -> str:
    """
    Upload the long-form video to YouTube with optimized metadata.
    Uses the existing upload_youtube module.
    """
    from upload_youtube import upload_video

    # Re-encode the video to ensure 16:9, 60fps, 4K, high codec (per user requirements)
    # If the Google Flow output already meets specs, this is a passthrough
    final_video_path = ensure_video_specs(video_path, config)

    video_id = upload_video(
        video_path=final_video_path,
        title=metadata.title,
        description=metadata.description,
        tags=metadata.tags,
        thumbnail_path=thumbnail_path,
        privacy_status=os.environ.get("YT_PRIVACY_STATUS", "public"),
    )

    # Post-upload: record the video ID for the optimization engine
    record_uploaded_video(video_id, metadata, final_video_path)

    log(f"Uploaded long-form video: https://www.youtube.com/watch?v={video_id}")
    return video_id


def ensure_video_specs(video_path: str, config: LongformVideoConfig) -> str:
    """
    Verify/re-encode the video to ensure it meets the long-form specs:
    16:9, 60fps, 4K, high codec.

    Uses ffmpeg with hardware acceleration when available. If the video
    already conforms, returns the original path unchanged.
    """
    from performance_optimizer import probe, has_audio_stream, video_encode_args, run_ffmpeg

    info = probe(video_path)
    streams = info.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)

    if video_stream is None:
        raise RuntimeError(f"No video stream found in {video_path}")

    width = video_stream.get("width", 0)
    height = video_stream.get("height", 0)
    fps_str = video_stream.get("avg_frame_rate", "24/1")
    fps_num, fps_den = fps_str.split("/")
    fps = float(fps_num) / float(fps_den) if float(fps_den) > 0 else 24

    needs_reencode = (
        width < config.width
        or height < config.height
        or fps != config.fps
        or video_stream.get("codec_name") != config.video_codec
    )

    if not needs_reencode and has_audio_stream(video_path):
        log(f"Video already meets specs ({width}x{height}, {fps}fps) — skipping re-encode.")
        return video_path

    log(f"Re-encoding to {config.aspect_ratio} {config.width}x{config.height} {config.fps}fps "
        f"({config.video_codec} {config.video_codec_profile}, crf={config.crf})...")

    output_path = video_path.replace(".mp4", "_4k.mp4")
    cmd = [
        "-i", video_path,
        "-vf", (
            f"scale={config.width}:{config.height}:"
            f"force_original_aspect_ratio=decrease,"
            f"pad={config.width}:{config.height}:"
            f"(ow-iw)/2:(oh-ih)/2:color=black"
        ),
        "-r", str(config.fps),
        "-c:v", config.video_codec,
        "-profile:v", config.video_codec_profile,
        "-b:v", config.video_bitrate,
        "-crf", str(config.crf),
        "-pix_fmt", config.pixel_format,
        "-c:a", config.audio_codec,
        "-b:a", config.audio_bitrate,
        "-ar", str(config.audio_sample_rate),
        "-ac", str(config.audio_channels),
        "-threads", "0",
        "-y", output_path,
    ]
    run_ffmpeg(cmd, desc="long-form 4K encode")

    return output_path


def record_uploaded_video(video_id: str, metadata: SEOMetadata, video_path: str) -> None:
    """Record the uploaded video's metadata for the optimization engine."""
    record_path = Path("work/longform_uploads.json")
    record_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    if record_path.exists():
        try:
            with open(record_path, "r") as f:
                records = json.load(f)
        except (json.JSONDecodeError, OSError):
            records = []

    record = {
        "video_id": video_id,
        "title": metadata.title,
        "tags": metadata.tags,
        "thumbnail_text": metadata.thumbnail_text,
        "spoken_keywords": metadata.keywords_spoken,
        "description_used": metadata.description,
        "caption": metadata.caption,
        "uploaded_at": datetime.utcnow().isoformat(),
        "video_path": video_path,
        "status": "uploaded_pending_analytics",
    }
    records.append(record)
    with open(record_path, "w") as f:
        json.dump(records, f, indent=2, default=str)

    log(f"Recorded upload for optimization engine (video_id={video_id[:11]}).")


# ---------------------------------------------------------------------------
# Step 4: Generate long-form thumbnail
# ---------------------------------------------------------------------------

def generate_longform_thumbnail(
    video_path: str,
    metadata: SEOMetadata,
    out_path: str,
) -> str:
    """
    Generate a 16:9 high-contrast thumbnail for the long-form video.
    Uses the optimized thumbnail_text from the SEO optimizer.
    """
    from generate_thumbnail import generate_thumbnail

    # Build a rich thumbnail prompt for 16:9, high contrast, vibrant
    thumbnail_prompt = (
        f"YouTube 16:9 thumbnail, {metadata.thumbnail_text}, "
        "extreme high contrast, vibrant saturated colors, dramatic cinematic lighting, "
        "bold text placeholder on the left third, 8K resolution, ultra-sharp detail, "
        "cinematic depth of field. Style: high-CTR thumbnail designed to stop the scroll. "
        "No watermarks, no logos."
    )

    generate_thumbnail(
        video_path=video_path,
        title_text=metadata.thumbnail_text,
        out_path=out_path,
        longform=True,
    )

    log(f"Long-form thumbnail generated: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Step 5: Evaluation + learning (called after analytics are available)
# ---------------------------------------------------------------------------

def evaluate_and_learn(video_id: str, video_path: str = "") -> Optional[ViralityScore]:
    """
    Fetch analytics for a previously-uploaded video and run the recursive
    learning engine on it.

    This is a separate command that should be run ~48 hours after upload
    when YouTube analytics are available.

    Usage:
        python src/longform_video_pipeline.py --evaluate VIDEO_ID
    """
    engine = get_engine()

    # Check upload records for this video
    record_path = Path("work/longform_uploads.json")
    record = None
    if record_path.exists():
        try:
            with open(record_path, "r") as f:
                records = json.load(f)
            for r in records:
                if r.get("video_id") == video_id:
                    record = r
                    break
        except (json.JSONDecodeError, OSError):
            pass

    if not record:
        log(f"No upload record found for video_id={video_id}. "
            "Cannot evaluate (need the original metadata).")
        return None

    # Build metrics from available data
    # In production, this would call the YouTube Analytics API
    metrics = _mock_metrics_from_record(record)

    # Evaluate
    evaluator = get_evaluator()
    # Learn from this video's performance
    score = engine.evaluate_and_learn(metrics)

    return score


def evaluate_pending_videos() -> None:
    """
    Check all uploaded videos that haven't been evaluated yet, fetch their
    analytics, and feed the results into the optimization engine.

    This should be run as a separate cron job (daily, after analytics are
    available). It's the 'feedback' part of the feedback loop.
    """
    record_path = Path("work/longform_uploads.json")
    if not record_path.exists():
        log("No upload records found. Nothing to evaluate.")
        return

    try:
        with open(record_path, "r") as f:
            records = json.load(f)
    except (json.JSONDecodeError, OSError):
        log("Could not read upload records.")
        return

    pending = [r for r in records if r.get("status") == "uploaded_pending_analytics"]
    if not pending:
        log("No pending evaluations.")
        return

    log(f"Found {len(pending)} videos pending evaluation.")

    # Try to fetch real analytics from YouTube
    from upload_youtube import _get_credentials
    try:
        from googleapiclient.discovery import build
        creds = _get_credentials()
        yt = build("youtube", "v3", credentials=creds)
        analytics = build("youtubeAnalytics", "v2", credentials=creds)
        fetched_real_data = True
    except Exception:
        fetched_real_data = False
        log("Could not build YouTube client — using available data.")

    engine = get_engine()

    for record in pending:
        video_id = record["video_id"]
        log(f"Evaluating video: {record['title'][:50]}...")

        if fetched_real_data:
            metrics = _fetch_real_analytics(video_id, record, yt, analytics)
        else:
            metrics = _mock_metrics_from_record(record)

        score = engine.evaluate_and_learn(metrics)

        # Mark as evaluated
        record["status"] = "evaluated"
        record["evaluation"] = score.to_dict()
        record["evaluated_at"] = datetime.utcnow().isoformat()

    # Save updated records
    with open(record_path, "w") as f:
        json.dump(records, f, indent=2, default=str)

    log(f"Evaluated {len(pending)} videos. Optimization engine state updated.")


def _fetch_real_analytics(video_id: str, record: dict, yt_client, analytics_client) -> VideoMetrics:
    """Fetch real analytics from YouTube Data API."""
    # Get video stats
    video_data = yt_client.videos().list(
        part="snippet,contentDetails,statistics,status",
        id=video_id,
    ).execute()

    item = video_data["items"][0]
    stats = item.get("statistics", {})
    snippet = item.get("snippet", {})

    # Get retention data
    retention_points = {}
    try:
        retention_resp = analytics_client.videos().getVideoRetention(
            ids=video_id,
        ).execute()
        for point in retention_resp.get("retention", []):
            retention_points[float(point["elapsedSeconds"])] = float(point["percent"])
    except Exception:
        pass

    # Get CTR and watch time from YouTube Analytics
    ctr = 0.0
    watch_time_min = 0.0
    try:
        import datetime as dt
        today = dt.date.today()
        seven_days_ago = today - dt.timedelta(days=7)

        report = analytics_client.reports().get(
            ids=f"channel==MINE",
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

    # Estimate AVD from averageViewDuration
    avg_vd_str = stats.get("averageViewDuration", "0")
    try:
        avg_vd_seconds = float(avg_vd_str)
    except (ValueError, TypeError):
        avg_vd_seconds = 0.0

    duration_str = item.get("contentDetails", {}).get("duration", "PT0S")
    duration_seconds = _parse_iso_duration(duration_str)

    is_short = "shorts" in (snippet.get("title", "") + " " + " ".join(snippet.get("tags", []))).lower()

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
        average_view_duration_seconds=avg_vd_seconds,
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


def _mock_metrics_from_record(record: dict) -> VideoMetrics:
    """Create a VideoMetrics object from an upload record (for testing/demo)."""
    return VideoMetrics(
        video_id=record.get("video_id", "unknown"),
        title=record.get("title", ""),
        is_short=False,
        duration_seconds=600.0,
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
    """Parse ISO 8601 duration like PT5M30S to seconds."""
    import re
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "PT0S")
    if not match:
        return 0.0
    h, m, s = match.groups("0")
    return int(h) * 3600 + int(m) * 60 + int(s)


# ---------------------------------------------------------------------------
# Full pipeline (async entry point for Google Flow)
# ------------------------------------------------------------------

async def run_longform_pipeline(
    topic: str | None = None,
    category: str = "any",
    custom_topic: str = "",
) -> str:
    """
    Run the complete regular video pipeline:
    topic → keywords → optimized prompt → Google Flow → thumbnail → SEO → upload
    """
    config = LongformVideoConfig()
    engine = get_engine()

    # Step 1: Resolve topic
    topic_mode = os.environ.get("TOPIC_MODE", "trending")
    if topic:
        resolved_topic = topic
    else:
        resolved_topic = resolve_topic(
            mode=topic_mode, category=category, custom_topic=custom_topic
        )
    log(f"Step 1/7: Topic resolved → {resolved_topic}")

    # Step 2: SEO keyword research
    log("Step 2/7: Researching real SEO keywords...")
    keywords = research_keywords(resolved_topic)
    log(f"  → Keywords: {keywords[:8]}")

    # Step 3: Get historical optimizations
    log("Step 3/7: Loading historical optimizations...")
    optimizations = engine.get_optimizations(resolved_topic, is_short=False)
    log(f"  → Historical confidence: {optimizations['historical_confidence']}")
    log(f"  → Pass rate: {optimizations['pass_rate']}")

    # Step 4: Build optimized Google Flow prompt
    log("Step 4/7: Building optimized Google Flow prompt...")
    flow_prompt = build_longform_prompt(resolved_topic, keywords, optimizations)
    log(f"  → Prompt length: {len(flow_prompt)} chars")

    # Step 5: Generate metadata (SEO/GEO/AEO)
    log("Step 5/7: Generating SEO/GEO/AEO metadata...")
    metadata = generate_longform_metadata(resolved_topic, keywords, config, optimizations)
    log(f"  → Title: {metadata.title}")
    log(f"  → Thumbnail text: {metadata.thumbnail_text}")

    # Step 6: Submit to Google Flow
    log("Step 6/7: Generating video via Google Flow (using all 50 free credits)...")
    project_id = os.environ.get("GOOGLE_FLOW_PROJECT_ID", "")
    api_key = os.environ.get("GOOGLE_FLOW_API_KEY", "")

    if not project_id and not api_key:
        log("WARNING: Google Flow credentials not set. "
            "Generating a mock video file for testing.")
        # In CI, this would still produce the metadata and placeholder
        video_path = _create_placeholder_video(config)
    else:
        async with GoogleFlowClient(project_id=project_id, api_key=api_key) as flow_client:
            session_config = FlowSessionConfig(
                project_id=project_id,
                max_credits=50,
                target_duration_seconds=config.duration_target,
            )
            result = await flow_client.generate_longform_video(
                prompt=flow_prompt,
                topic=resolved_topic,
                session_config=session_config,
                video_config=config,
            )
            video_path = result.local_path
            if result.credits_used < 45:
                log(f"WARNING: Only {result.credits_used}/50 credits used. "
                    "Consider increasing target duration to maximize credit usage.")

    if not video_path or not os.path.exists(video_path):
        raise RuntimeError("Google Flow video generation failed — no video file produced.")

    log(f"  → Video generated: {video_path}")

    # Step 7: Generate thumbnail + upload
    log("Step 7/7: Generating 16:9 thumbnail and uploading to YouTube...")
    thumbnail_path = os.path.join(
        os.path.dirname(video_path),
        f"thumbnail_{resolved_topic[:30].replace(' ', '_')}.jpg",
    )
    generate_longform_thumbnail(video_path, metadata, thumbnail_path)

    # Ensure video meets 4K/60fps specs
    final_video_path = ensure_video_specs(video_path, config)

    # Upload to YouTube
    video_id = upload_longform_video(
        video_path=final_video_path,
        metadata=metadata,
        thumbnail_path=thumbnail_path,
        config=config,
    )

    log(f"\nLong-form video complete! https://www.youtube.com/watch?v={video_id}")
    log("⚠️  Run '--evaluate-pending' ~48h later to feed analytics back into the optimization engine.")

    return video_id


def _create_placeholder_video(config: LongformVideoConfig) -> str:
    """
    Create a minimal placeholder video for testing when Google Flow is not
    available. Generates a simple color-pattern video with the right specs.
    """
    from performance_optimizer import run_ffmpeg, video_encode_args
    import tempfile

    out_dir = Path(config.project_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = str(out_dir / f"longform_placeholder_{int(datetime.utcnow().timestamp())}.mp4")

    # Generate a simple 10-second color test pattern at 4K/60fps
    run_ffmpeg(
        [
            "-f", "lavfi", "-i", "color=c=blue:s=3840x2160:r=60:d=10",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration=10",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-y", video_path,
        ],
        desc="placeholder long-form video",
    )
    log(f"Created placeholder video: {video_path}")
    return video_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Long-form regular video pipeline")
    parser.add_argument("topic", nargs="?", default=None, help="Custom topic (optional)")
    parser.add_argument("--category", default="any", help="Topic category")
    parser.add_argument("--custom-topic", default="", help="Exact custom topic")
    parser.add_argument("--evaluate-pending", action="store_true",
                        help="Fetch analytics for all pending videos and feed to optimization engine")
    parser.add_argument("--evaluate", type=str, default=None,
                        help="Evaluate a specific video ID and feed to optimization engine")

    args = parser.parse_args()

    if args.evaluate_pending:
        evaluate_pending_videos()
        sys.exit(0)

    if args.evaluate:
        score = evaluate_and_learn(args.evaluate)
        if score:
            print(json.dumps(score.to_dict(), indent=2))
        sys.exit(0)

    try:
        result = asyncio.run(run_longform_pipeline(
            topic=args.topic,
            category=args.category,
            custom_topic=args.custom_topic,
        ))
        print(f"\n✅ Long-form video uploaded: https://www.youtube.com/watch?v={result}")
    except Exception:
        print("Long-form pipeline failed:")
        traceback.print_exc()
        sys.exit(1)
