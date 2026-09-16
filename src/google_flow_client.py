"""
Google Flow client for long-form video generation.

Integrates with Google Flow (Google's AI video generation platform) to:
  - Create/manage a project session
  - Submit an optimized prompt for animated illustration explainer generation
  - Poll for completion
  - Download the final video (longest possible, using all 50 free daily credits)
  - Track credit usage per session

Google Flow API surface (as of the current public integration):
  - REST endpoint: https://flow.googleapis.com/v1/projects/{project}/videos
  - Authentication: Application Default Credentials (service account JSON key
    or ambient ADC on Cloud Run / GitHub Actions with Workload Identity Federation)
  - Free tier: 50 credits per day → each generated video segment costs credits
    proportional to duration + quality. We target the maximum duration to
    consume all 50 credits.

Fallback: if the Google Flow API is unavailable, the client logs a clear
message explaining that a manual session must be used.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

from regular_video_config import (
    GOOGLE_FLOW_FREE_CREDITS,
    GOOGLE_FLOW_PROJECT_DIR,
    LongformVideoConfig,
)


LOG_PREFIX = "[google_flow]"


def log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", flush=True)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

FLOW_BASE_URL = "https://flow.googleapis.com/v1"
FLOW_API_VERSION = "v1"


@dataclass
class FlowSessionConfig:
    """Configuration for a Google Flow project session."""

    project_id: str
    session_id: Optional[str] = None
    output_dir: Path = Path(GOOGLE_FLOW_PROJECT_DIR)
    max_credits: int = GOOGLE_FLOW_FREE_CREDITS
    target_duration_seconds: float = 600.0  # 10 minutes — longest possible
    aspect_ratio: str = "16:9"
    quality: str = "high"  # high | medium | standard
    style: str = "animated_illustration"  # animated_illustration | live_action | storyboard

    def to_api_payload(self) -> dict:
        return {
            "aspectRatio": self.aspect_ratio,
            "quality": self.quality,
            "style": self.style,
            "maxCredits": self.max_credits,
            "targetDurationSeconds": int(self.target_duration_seconds),
        }


@dataclass
class FlowVideoResult:
    """Result of a Google Flow video generation session."""

    session_id: str
    video_url: Optional[str]
    local_path: Optional[str]
    duration_seconds: float
    credits_used: int
    credits_remaining: int
    status: str
    metadata: dict = field(default_factory=dict)


class FlowAPIError(Exception):
    """Raised when the Google Flow API returns an error."""
    pass


class GoogleFlowClient:
    """
    Client for the Google Flow video generation API.

    Usage:
        client = GoogleFlowClient(project_id="my-gcp-project")
        result = client.generate_video(prompt="...", topic="...")
        # result.local_path contains the downloaded video
    """

    def __init__(
        self,
        project_id: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
    ):
        self.project_id = project_id or os.environ.get("GOOGLE_FLOW_PROJECT_ID", "")
        self.api_key = api_key or os.environ.get("GOOGLE_FLOW_API_KEY", "")
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=30.0),
            base_url=FLOW_BASE_URL,
        )

    # ------------------------------------------------------------------
    # Authentication helpers
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        """Build auth headers from either API key or Application Default Creds."""
        if self.api_key:
            return {"x-api-key": self.api_key, "Content-Type": "application/json"}
        # Prefer ADC via google-auth; fall back to env var.
        try:
            from google.auth import default as adc_default
            from google.auth.transport.requests import Request

            creds, _ = adc_default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            creds.refresh(Request())
            return {"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json"}
        except Exception:
            log("No Google Cloud credentials found. Google Flow requires "
                "GOOGLE_FLOW_API_KEY or Application Default Credentials.")
            raise FlowAPIError(
                "Google Flow authentication unavailable. Set GOOGLE_FLOW_API_KEY or "
                "provide Application Default Credentials."
            )

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def create_session(self, config: FlowSessionConfig) -> str:
        """
        Create a new Google Flow project session.
        Returns the session_id.
        """
        if not self.project_id:
            raise FlowAPIError("GOOGLE_FLOW_PROJECT_ID must be set.")

        headers = self._auth_headers()
        headers["Content-Type"] = "application/json"

        payload = config.to_api_payload()
        payload["projectId"] = self.project_id
        payload["sessionName"] = f"daily-longform-{int(time.time())}"

        log(f"Creating Google Flow session in project '{self.project_id}'...")
        try:
            resp = await self._client.post(
                f"/projects/{self.project_id}/sessions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            session_id = data.get("sessionId") or data.get("name", "").split("/")[-1]
            log(f"Session created: {session_id}")
            config.session_id = session_id
            return session_id
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise FlowAPIError(
                    "Google Flow API not yet publicly available or project not enrolled. "
                    "A manual Google Flow session is required until the API is GA."
                ) from exc
            raise FlowAPIError(f"Failed to create Flow session: {exc}") from exc

    # ------------------------------------------------------------------
    # Video generation
    # ------------------------------------------------------------------

    async def submit_video_job(
        self,
        session_id: str,
        prompt: str,
        topic: str,
        config: LongformVideoConfig | None = None,
    ) -> str:
        """
        Submit a video generation job to the Flow session using the optimized
        prompt. Returns the job_id.
        """
        config = config or LongformVideoConfig()
        headers = self._auth_headers()

        payload = {
            "prompt": prompt,
            "metadata": {
                "topic": topic,
                "aspectRatio": config.aspect_ratio,
                "resolution": f"{config.width}x{config.height}",
                "fps": config.fps,
                "videoCodec": config.video_codec,
                "videoCodecProfile": config.video_codec_profile,
                "videoBitrate": config.video_bitrate,
                "audioCodec": config.audio_codec,
                "audioBitrate": config.audio_bitrate,
                "audioSampleRate": config.audio_sample_rate,
                "audioChannels": config.audio_channels,
                "pixelFormat": config.pixel_format,
                "crf": config.crf,
                "preset": config.preset,
                "maxCredits": GOOGLE_FLOW_FREE_CREDITS,
                "targetDurationSeconds": int(config.duration_target),
            },
        }

        log(f"Submitting video job to session '{session_id}'...")
        resp = await self._client.post(
            f"/projects/{self.project_id}/sessions/{session_id}/videos",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        job_id = data.get("jobId") or data.get("name", "").split("/")[-1]
        log(f"Video job submitted: {job_id} (using up to {GOOGLE_FLOW_FREE_CREDITS} credits)")
        return job_id

    async def poll_job(
        self,
        session_id: str,
        job_id: str,
        poll_interval: float = 15.0,
        max_wait: float = 900.0,
    ) -> FlowVideoResult:
        """
        Poll the job until completion. Returns a FlowVideoResult with the
        download URL and credit usage.
        """
        headers = self._auth_headers()
        log(f"Polling job '{job_id}' (max wait {max_wait}s)...")

        start = time.monotonic()
        while time.monotonic() - start < max_wait:
            resp = await self._client.get(
                f"/projects/{self.project_id}/sessions/{session_id}/videos/{job_id}",
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

            status = data.get("status", "unknown")
            credits_used = data.get("creditsUsed", 0)
            credits_remaining = data.get("creditsRemaining", 0)

            if status == "SUCCEEDED":
                video_url = data.get("output", {}).get("videoUrl")
                duration = data.get("output", {}).get("durationSeconds", 0.0)
                metadata = data.get("output", {}).get("metadata", {})
                log(f"Job succeeded | credits_used={credits_used} | "
                    f"credits_remaining={credits_remaining} | duration={duration}s")
                return FlowVideoResult(
                    session_id=session_id,
                    video_url=video_url,
                    local_path=None,
                    duration_seconds=duration,
                    credits_used=credits_used,
                    credits_remaining=credits_remaining,
                    status=status,
                    metadata=metadata,
                )

            if status in ("FAILED", "CANCELLED"):
                error = data.get("error", {}).get("message", "unknown error")
                raise FlowAPIError(f"Job '{job_id}' {status.lower()}: {error}")

            log(f"  ...status={status} | credits_used={credits_used}/{GOOGLE_FLOW_FREE_CREDITS} "
                f"| elapsed={time.monotonic() - start:.0f}s")
            time.sleep(poll_interval)

        raise FlowAPIError(f"Job '{job_id}' timed out after {max_wait}s.")

    # ------------------------------------------------------------------
    # Credit awareness
    # ------------------------------------------------------------------

    async def get_credit_balance(self) -> dict:
        """Fetch the current free-tier credit balance for the project."""
        headers = self._auth_headers()
        try:
            resp = await self._client.get(
                f"/projects/{self.project_id}/credits",
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                log("Credit balance endpoint not available — assuming 50 free credits.")
                return {"total": GOOGLE_FLOW_FREE_CREDITS, "used": 0, "remaining": GOOGLE_FLOW_FREE_CREDITS}
            raise FlowAPIError(f"Failed to fetch credit balance: {exc}") from exc

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    async def download_video(self, video_url: str, output_path: str) -> str:
        """Download the generated video from the Flow output URL."""
        log(f"Downloading video to {output_path}...")
        resp = await self._client.get(video_url, timeout=600.0)
        resp.raise_for_status()

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(resp.content)
        log(f"Download complete: {output_path} ({len(resp.content)} bytes)")
        return output_path

    # ------------------------------------------------------------------
    # Full pipeline helper
    # ------------------------------------------------------------------

    async def generate_longform_video(
        self,
        prompt: str,
        topic: str,
        session_config: FlowSessionConfig | None = None,
        video_config: LongformVideoConfig | None = None,
    ) -> FlowVideoResult:
        """
        End-to-end: create session → submit job → poll → download.
        Maximizes credit usage toward all 50 free daily credits.
        """
        session_config = session_config or FlowSessionConfig(
            project_id=self.project_id,
        )
        video_config = video_config or LongformVideoConfig()

        # Check credit balance first
        balance = await self.get_credit_balance()
        remaining = balance.get("remaining", GOOGLE_FLOW_FREE_CREDITS)
        log(f"Google Flow credit balance: {remaining}/{GOOGLE_FLOW_FREE_CREDITS} free credits remaining.")

        if remaining <= 0:
            raise FlowAPIError(
                f"No Google Flow credits remaining ({remaining}). "
                "Wait until the next daily reset or upgrade your plan."
            )

        # Create session
        session_id = await self.create_session(session_config)

        # Submit job with optimized prompt — target max duration to use all credits
        job_id = await self.submit_video_job(session_id, prompt, topic, video_config)

        # Poll until done
        result = await self.poll_job(session_id, job_id)

        # Download the video
        if result.video_url:
            session_config.output_dir.mkdir(parents=True, exist_ok=True)
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in topic)[:60]
            output_path = str(session_config.output_dir / f"{safe_name}_{int(time.time())}.mp4")
            local_path = await self.download_video(result.video_url, output_path)
            result.local_path = local_path

        return result

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
