"""
Uploads a finished video to YouTube using the YouTube Data API v3.

Auth uses a long-lived OAuth refresh token (generated once locally via
get_refresh_token.py) so the pipeline can run unattended in CI.
"""

from __future__ import annotations

import os
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from _sanitize import sanitize_credential

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def _get_credentials():
    return Credentials(
        token=None,
        refresh_token=sanitize_credential(os.environ["YT_REFRESH_TOKEN"]),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=sanitize_credential(os.environ["YT_CLIENT_ID"]),
        client_secret=sanitize_credential(os.environ["YT_CLIENT_SECRET"]),
        scopes=SCOPES,
    )


def upload_video(
    video_path: str,
    title: str,
    description: str,
    tags: list[str],
    thumbnail_path: str | None = None,
    privacy_status: str = "public",
    category_id: str = "22",
    default_language: str | None = None,
):
    try:
        creds = _get_credentials()
        youtube = build("youtube", "v3", credentials=creds)
    except RefreshError as exc:
        raise RuntimeError(
            "YouTube OAuth refresh token is invalid or expired. "
            "Regenerate it locally with: python get_refresh_token.py, "
            "then update the YT_REFRESH_TOKEN secret in GitHub repo settings."
        ) from exc

    # YouTube uses defaultLanguage/defaultAudioLanguage to match the video to
    # the right audience. Without it the Short gets no language signal and
    # essentially no impressions -- the #1 reason new uploads get zero views.
    language = (
        default_language
        or os.environ.get("YT_DEFAULT_LANGUAGE")
        or "en-US"
    )

    body = {
        "snippet": {
            "title": title[:100],
            "description": description,
            "tags": tags,
            "categoryId": category_id,
            "defaultLanguage": language,
            "defaultAudioLanguage": language,
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": False,
        },
    }
    print(f"[upload_youtube] Upload settings: language={language} category={category_id} privacy={privacy_status}")

    media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/mp4")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    try:
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"[upload_youtube] Upload progress: {int(status.progress() * 100)}%")
    except RefreshError as exc:
        raise RuntimeError(
            "YouTube OAuth refresh token is invalid or expired. "
            "Regenerate it locally with: python get_refresh_token.py, "
            "then update the YT_REFRESH_TOKEN secret in GitHub repo settings."
        ) from exc

    video_id = response["id"]
    print(f"[upload_youtube] Uploaded: https://www.youtube.com/watch?v={video_id}")

    if thumbnail_path:
        try:
            youtube.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(thumbnail_path)).execute()
        except Exception as e:
            # The video itself is already live -- a thumbnail failure (e.g. the
            # channel isn't phone-verified yet, see youtube.com/verify) should
            # not be treated as a fatal pipeline error and shouldn't hide a
            # successful upload behind a red "failed" run.
            print(f"[upload_youtube] WARNING: could not set custom thumbnail ({e}). "
                  f"Video uploaded fine with the auto-generated thumbnail instead.")

    return video_id


if __name__ == "__main__":
    print("Run via main.py with a real video path.")
