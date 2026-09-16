from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import httpx

from api_clients import (
    APIError,
    AssetManager,
    ImageGenClient,
    LLMClient,
    TTSClient,
    VideoGenClient,
)
from config import load_settings
from models import AnimationParams, ScriptPackage, VideoMetadata, VideoPipelineResult
from prompts import IMAGE_GEN_PROMPT_TEMPLATE


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger("longform_video_pipeline")


def _ffmpeg_video_args() -> list[str]:
    """
    Best-available H.264 encode args for direct ffmpeg subprocess use,
    cached per process:
      - macOS + VideoToolbox -> hardware encode
      - NVIDIA + NVENC       -> hardware encode
      - otherwise            -> libx264 fast preset (speed prioritized for CI)
    Every render in the pipeline goes through this for uniform acceleration.
    """
    import platform
    encoders = _ffmpeg_encoders()
    system = platform.system()
    if system == "Darwin" and "h264_videotoolbox" in encoders:
        return ["-c:v", "h264_videotoolbox", "-b:v", "8M", "-pix_fmt", "yuv420p"]
    if "h264_nvenc" in encoders and _nvenc_available():
        return ["-c:v", "h264_nvenc", "-b:v", "8M", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p"]


def _moviepy_encode_config() -> tuple[str, list[str], int]:
    """
    Returns (codec, ffmpeg_params, threads) for MoviePy's write_videofile,
    auto-detecting hardware encoders. Uses threads=0 to leverage all cores,
    and -preset fast for libx264 to prevent GitHub runner timeouts.
    """
    import platform
    encoders = _ffmpeg_encoders()
    system = platform.system()
    if system == "Darwin" and "h264_videotoolbox" in encoders:
        return ("h264_videotoolbox", ["-b:v", "8M"], 0)
    if "h264_nvenc" in encoders and _nvenc_available():
        return ("h264_nvenc", ["-b:v", "8M"], 0)
    return ("libx264", ["-preset", "fast", "-crf", "23"], 0)


_ENCODERS_CACHE: str | None = None
_NVENC_CACHE: bool | None = None


def _ffmpeg_encoders() -> str:
    global _ENCODERS_CACHE
    if _ENCODERS_CACHE is None:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, check=False,
        )
        _ENCODERS_CACHE = result.stdout
    return _ENCODERS_CACHE


def _nvenc_available() -> bool:
    import ctypes.util
    import platform
    global _NVENC_CACHE
    if _NVENC_CACHE is None:
        if platform.system() != "Linux":
            _NVENC_CACHE = False
        else:
            _NVENC_CACHE = ctypes.util.find_library("cuda") is not None
    return _NVENC_CACHE


def _ffprobe_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return 0.0
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _ffprobe_has_audio(path: str) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=False,
    )
    return "audio" in result.stdout


class Pipeline:
    def __init__(self, settings, logger: logging.Logger):
        self.settings = settings
        self.logger = logger
        self.assets = AssetManager(settings.work_dir, settings.output_dir)
        self.result = VideoPipelineResult(success=False)

    async def run(self, topic: str) -> VideoPipelineResult:
        start = time.time()
        self.logger.info(
            f"Pipeline started | topic={topic!r} | "
            f"llm={self.settings.llm_provider} | "
            f"tts={self.settings.tts_provider} | "
            f"image={self.settings.image_gen_provider} | "
            f"video={self.settings.video_gen_provider} | "
            f"thumbnail={self.settings.thumbnail_gen_provider} | "
            f"duration={self.settings.target_duration_seconds}s | "
            f"resolution={self.settings.resolution}"
        )

        shared_client = httpx.AsyncClient(timeout=120.0, connect_timeout=30.0)

        try:
            async with shared_client as client:
                llm = LLMClient(self.settings, client)
                script = await self._stage_expand_script(topic, llm)
                voiceover_path, voice_duration = await self._stage_voiceover(script, client)
                image_paths = await self._stage_generate_images(script, client)
                video_segments = await self._stage_animate_images(script, image_paths, client)
                final_path = await self._stage_assemble(video_segments, voiceover_path, voice_duration, script)
                _, metadata = await self._stage_package_metadata(
                    script, final_path, voice_duration, llm, client
                )

            self.result.success = True
            self.result.video_path = final_path
            self.result.metadata = metadata
            self.result.metadata_json_path = self.assets.metadata_path()
            self.result.total_duration_seconds = time.time() - start
            self.result.logs.append(f"Pipeline completed successfully in {self.result.total_duration_seconds:.1f}s")
            self.logger.info(f"Pipeline success | output={final_path} | metadata={self.assets.metadata_path()}")
            return self.result
        except Exception as exc:
            self.result.error = str(exc)
            self.result.logs.append(f"Pipeline failed: {exc}")
            self.logger.error(f"Pipeline failed: {exc}\n{traceback.format_exc()}")
            return self.result

    async def _stage_expand_script(self, topic: str, llm: LLMClient) -> ScriptPackage:
        self.logger.info("Stage 1/5: Expanding script and visual scene breakdown...")
        script = await llm.generate_script(topic, self.settings.target_duration_seconds)
        self.logger.info(f"  -> Title: {script.title} | Sections: {len(script.sections)}")
        for section in script.sections:
            self.assets.log_asset(
                scene_num=section.section_num,
                asset_type="script_section",
                voiceover_text=section.voiceover_text,
                image_prompt=section.image_prompt,
            )
        return script

    async def _stage_voiceover(self, script: ScriptPackage, client: httpx.AsyncClient) -> tuple[Path, float]:
        self.logger.info("Stage 2/5: Generating voiceover narration...")
        full_text = "\n\n".join(s.voiceover_text for s in script.sections)
        output_path = self.assets.voiceover_path()
        tts = TTSClient(self.settings, client)
        duration = await tts.synthesize(full_text, output_path)
        self.assets.log_asset(scene_num=0, asset_type="audio", file_path=output_path, duration_seconds=duration)
        self.logger.info(f"  -> Voiceover saved | duration={duration:.1f}s path={output_path}")
        return output_path, duration

    async def _stage_generate_images(self, script: ScriptPackage, client: httpx.AsyncClient) -> dict[int, Path]:
        self.logger.info("Stage 3/5: Generating base frames for each scene (parallel)...")
        img = ImageGenClient(self.settings, client)

        prompts: list[tuple[str, Path, int, int]] = []
        for section in script.sections:
            out = self.assets.scene_image_path(section.section_num)
            enhanced_prompt = IMAGE_GEN_PROMPT_TEMPLATE.substitute(scene_description=section.image_prompt)
            prompts.append((enhanced_prompt, out, 1920, 1080))
            self.logger.info(f"  -> Queued image for scene {section.section_num}...")

        results = await img.generate_images_batch(prompts)

        image_paths: dict[int, Path] = {}
        for i, result in enumerate(results):
            section = script.sections[i]
            if result is None:
                raise APIError(f"Failed to generate image for scene {section.section_num}", provider="pipeline")
            image_paths[section.section_num] = result
            self.assets.log_asset(scene_num=section.section_num, asset_type="image", file_path=result)
            self.logger.info(f"  -> Image ready for scene {section.section_num}")
        return image_paths

    async def _stage_animate_images(
        self, script: ScriptPackage, image_paths: dict[int, Path], client: httpx.AsyncClient
    ) -> dict[int, Path]:
        self.logger.info("Stage 4/5: Animating images to video segments (parallel)...")
        vid = VideoGenClient(self.settings, client)

        jobs: list[tuple[Path, str, Path, AnimationParams]] = []
        for section in script.sections:
            img_path = image_paths.get(section.section_num)
            if not img_path:
                raise APIError(f"Missing image for scene {section.section_num}", provider="pipeline")
            out = self.assets.scene_video_path(section.section_num)
            jobs.append((img_path, section.image_prompt, out, section.animation))
            self.logger.info(f"  -> Queued animation for scene {section.section_num}...")

        results = await vid.generate_videos_batch(jobs)

        video_paths: dict[int, Path] = {}
        for i, result in enumerate(results):
            section = script.sections[i]
            if result is None:
                raise APIError(f"Failed to animate scene {section.section_num}", provider="pipeline")
            video_paths[section.section_num] = result
            self.assets.log_asset(
                scene_num=section.section_num,
                asset_type="video_segment",
                file_path=result,
                duration_seconds=section.animation.duration_seconds,
            )
            self.logger.info(f"  -> Video ready for scene {section.section_num}")
        return video_paths

    async def _stage_assemble(
        self, video_segments: dict[int, Path], voiceover_path: Path, voice_duration: float, script: ScriptPackage
    ) -> Path:
        self.logger.info("Stage 5/5: Assembling final video with narration...")
        final_path = self.assets.final_video_path()

        sorted_paths = [video_segments[i] for i in sorted(video_segments)]
        if not sorted_paths:
            raise APIError("No video segments to assemble", provider="pipeline")

        from moviepy.editor import (
            AudioFileClip,
            CompositeAudioClip,
            VideoFileClip,
            concatenate_videoclips,
        )

        clips = []
        for path in sorted_paths:
            clip = VideoFileClip(str(path))
            clips.append(clip)

        combined = concatenate_videoclips(clips, method="compose")
        combined = combined.resize(newsize=self._parse_resolution(self.settings.resolution))

        audio = AudioFileClip(str(voiceover_path))
        if audio.duration > combined.duration:
            audio = audio.subclip(0, combined.duration)

        combined.audio = CompositeAudioClip([audio])

        codec, ffmpeg_params, threads = _moviepy_encode_config()
        combined.write_videofile(
            str(final_path),
            fps=self.settings.fps,
            codec=codec,
            audio_codec="aac",
            threads=threads,
            ffmpeg_params=ffmpeg_params,
            logger=None,
        )

        final_duration = combined.duration
        for clip in clips:
            clip.close()
        audio.close()
        combined.close()

        self.assets.log_asset(scene_num=0, asset_type="final", file_path=final_path, duration_seconds=final_duration)
        self.logger.info(f"  -> Final video saved: {final_path}")
        return final_path

    def _parse_resolution(self, resolution: str) -> tuple[int, int]:
        parts = resolution.lower().split("x")
        if len(parts) != 2:
            raise ValueError(f"Invalid resolution format: {resolution}")
        return int(parts[0]), int(parts[1])

    async def _stage_package_metadata(
        self,
        script: ScriptPackage,
        video_path: Path,
        voice_duration: float,
        llm: LLMClient,
        client: httpx.AsyncClient,
    ) -> tuple[Path, VideoMetadata]:
        self.logger.info("Stage 6/5: Generating AI metadata + high-CTR thumbnail...")

        refined = await llm.generate_metadata(
            script.title, script.hook, script.description, voice_duration
        )
        title = refined.get("refined_title", script.title)
        description = refined.get("refined_description", script.description)
        tags = refined.get("refined_tags", script.tags)

        img = ImageGenClient(self.settings, client)
        thumbnail_path = self.assets.thumbnail_path()
        await img.generate_thumbnail(
            script.thumbnail_prompt,
            thumbnail_path,
            width=1280,
            height=720,
        )
        self.assets.log_asset(scene_num=0, asset_type="thumbnail", file_path=thumbnail_path)
        self.logger.info(f"  -> Thumbnail saved: {thumbnail_path}")

        metadata_path = self.assets.write_metadata(title, description, tags, thumbnail_path, video_path)
        self.logger.info(f"  -> Metadata saved: {metadata_path}")

        metadata = VideoMetadata(
            title=title,
            description=description,
            tags=tags,
            thumbnail_path=thumbnail_path,
            video_path=video_path,
        )
        return thumbnail_path, metadata


def run(topic: str) -> VideoPipelineResult:
    settings = load_settings()
    logger = setup_logging(settings.log_level)
    pipeline = Pipeline(settings, logger)
    return asyncio.run(pipeline.run(topic))


if __name__ == "__main__":
    topic = os.environ.get("TOPIC", "The future of artificial intelligence in healthcare")
    try:
        result = run(topic)
        if result.success:
            print(f"SUCCESS: {result.video_path}")
            print(f"METADATA: {result.metadata_json_path}")
            print(f"THUMBNAIL: {result.metadata.thumbnail_path if result.metadata else 'N/A'}")
            sys.exit(0)
        else:
            print(f"FAILED: {result.error}")
            sys.exit(1)
    except Exception as exc:
        print(f"FATAL: {exc}")
        traceback.print_exc()
        sys.exit(1)
