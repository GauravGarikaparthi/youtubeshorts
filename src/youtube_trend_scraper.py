"""
YouTube trend scraper for high-demand viral trend detection.

Scrapes YouTube for a given topic and filters by:
- Upload Date → This Month
- Sort by Most Viewed
- Detects if channels with under 50,000 subscribers are pulling 100,000+ views
  → indicates high-demand viral trend suitable for the next video

Uses YouTube Data API v3 (same credentials as upload). Falls back gracefully
if credentials are missing.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import googleapiclient.discovery
from google.oauth2.credentials import Credentials

LOG_PREFIX = "[youtube_trend_scraper]"


def log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", flush=True)


YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"


def _get_youtube_client():
    """Return an authenticated YouTube API client if credentials exist."""
    try:
        client_id = os.environ.get("YT_CLIENT_ID")
        client_secret = os.environ.get("YT_CLIENT_SECRET")
        refresh_token = os.environ.get("YT_REFRESH_TOKEN")
        if not all([client_id, client_secret, refresh_token]):
            return None
        creds = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
            scopes=["https://www.googleapis.com/auth/youtube.readonly"],
        )
        return googleapiclient.discovery.build(
            YOUTUBE_API_SERVICE_NAME,
            YOUTUBE_API_VERSION,
            credentials=creds,
            cache_discovery=False,
        )
    except Exception as exc:
        log(f"Failed to build YouTube client: {exc}")
        return None


def _compute_published_after() -> str:
    """Published after start of this month in ISO 8601 format."""
    now = datetime.utcnow()
    start_of_month = datetime(now.year, now.month, 1)
    # Format as RFC 3339
    return start_of_month.strftime("%Y-%m-%dT%H:%M:%SZ")


def scrape_trends(
    topic: str,
    max_results: int = 25,
) -> Dict:
    """
    Scrape YouTube for trends on a topic.

    Returns dict with:
    - items: list of video/channel info
    - viral_trend_found: bool (under 50k subs, 100k+ views in this month)
    - recommendation: str (what to do next)
    """
    youtube = _get_youtube_client()
    if not youtube:
        log("YouTube API client not available; skipping trend scrape")
        return {
            "items": [],
            "viral_trend_found": False,
            "recommendation": "No credentials; proceed with existing topic",
        }

    try:
        published_after = _compute_published_after()
        # Search for videos this month, most viewed
        search_response = youtube.search().list(
            q=topic,
            part="snippet",
            type="video",
            order="viewCount",
            publishedAfter=published_after,
            maxResults=max_results,
        ).execute()

        items = []
        video_ids = []
        for search_result in search_response.get("items", []):
            vid = search_result["id"]["videoId"]
            video_ids.append(vid)

        if not video_ids:
            return {
                "items": [],
                "viral_trend_found": False,
                "recommendation": f"No recent videos found for '{topic}'",
            }

        # Get video details (views, channel id)
        videos_response = youtube.videos().list(
            part="statistics,snippet",
            id=",".join(video_ids),
        ).execute()

        channel_ids = set()
        for vid in videos_response.get("items", []):
            channel_ids.add(vid["snippet"]["channelId"])

        # Get channel details (subscriber counts)
        channels_response = youtube.channels().list(
            part="statistics,snippet",
            id=",".join(list(channel_ids)[:50]),  # API limit
        ).execute()

        channel_stats = {}
        for ch in channels_response.get("items", []):
            channel_stats[ch["id"]] = {
                "subs": int(ch["statistics"].get("subscriberCount", 0)),
                "title": ch["snippet"]["title"],
            }

        viral_found = False
        for vid in videos_response.get("items", []):
            stats = vid["statistics"]
            views = int(stats.get("viewCount", 0))
            channel_id = vid["snippet"]["channelId"]
            ch_info = channel_stats.get(channel_id, {"subs": 999999})
            subs = ch_info["subs"]
            items.append({
                "video_id": vid["id"],
                "title": vid["snippet"]["title"],
                "channel_title": ch_info.get("title", vid["snippet"]["channelTitle"]),
                "views": views,
                "subs": subs,
            })
            # Detect viral trend: <50k subs, >100k views this month
            if subs < 50000 and views > 100000:
                viral_found = True

        rec = (
            "HIGH-DEMAND VIRAL TREND DETECTED. Consider making next video on this topic."
            if viral_found
            else "No strong viral signal in this month's results."
        )
        return {
            "items": items,
            "viral_trend_found": viral_found,
            "recommendation": rec,
            "topic": topic,
        }
    except Exception as exc:
        log(f"Trend scrape failed: {exc}")
        return {
            "items": [],
            "viral_trend_found": False,
            "recommendation": f"Error during scrape: {exc}",
        }
