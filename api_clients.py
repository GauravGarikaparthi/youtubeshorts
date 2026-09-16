from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import Settings
from models import AnimationParams, ScriptPackage
from prompts import (
    METADATA_REFINEMENT_SYSTEM_PROMPT,
    THUMBNAIL_GEN_PROMPT_TEMPLATE,
    VIDEO_GEN_PROMPT_TEMPLATE,
    build_metadata_refinement_prompt,
)


class APIError(Exception):
    def __init__(self, message: str, status_code: int | None = None, provider: str = "unknown"):
        super().__init__(message)
        self.status_code = status_code
        self.provider = provider


class RateLimitError(APIError):
    pass


def _tenacity_retry():
    return retry(
        retry=retry_if_exception_type((RateLimitError, httpx.TimeoutException, httpx.HTTPStatusError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )


class BaseAPIClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.settings = settings
        self.client = client
        self.timeout = httpx.Timeout(120.0, connect=30.0)


class LLMClient(BaseAPIClient):
    @_tenacity_retry()
    async def generate_script(self, topic: str, target_duration_seconds: float) -> ScriptPackage:
        from prompts import (
            SCRIPT_EXPANSION_SYSTEM_PROMPT,
            build_script_expansion_user_prompt,
        )

        prompt = build_script_expansion_user_prompt(topic, target_duration_seconds)

        if self.settings.llm_provider == "gemini":
            return await self._call_gemini(SCRIPT_EXPANSION_SYSTEM_PROMPT, prompt)
        if self.settings.llm_provider == "claude":
            return await self._call_claude(SCRIPT_EXPANSION_SYSTEM_PROMPT, prompt)
        if self.settings.llm_provider == "groq":
            return await self._call_groq(SCRIPT_EXPANSION_SYSTEM_PROMPT, prompt)
        raise ValueError(f"Unsupported LLM provider: {self.settings.llm_provider}")

    async def generate_metadata(
        self, title: str, hook: str, description: str, duration_seconds: float
    ) -> dict[str, Any]:
        """
        Refines the video's metadata (title, description, tags) via LLM
        to maximize SEO and CTR. Returns a dict with refined_title,
        refined_description, and refined_tags.
        """

        prompt = build_metadata_refinement_prompt(title, hook, description, duration_seconds)

        if self.settings.llm_provider == "gemini":
            return await self._call_gemini_json(
                METADATA_REFINEMENT_SYSTEM_PROMPT, prompt, schema_keys=["refined_title", "refined_description", "refined_tags"]
            )
        if self.settings.llm_provider == "claude":
            return await self._call_claude_json(
                METADATA_REFINEMENT_SYSTEM_PROMPT, prompt, schema_keys=["refined_title", "refined_description", "refined_tags"]
            )
        if self.settings.llm_provider == "groq":
            return await self._call_groq_json(
                METADATA_REFINEMENT_SYSTEM_PROMPT, prompt, schema_keys=["refined_title", "refined_description", "refined_tags"]
            )
        raise ValueError(f"Unsupported LLM provider: {self.settings.llm_provider}")

    async def _call_gemini(self, system_prompt: str, user_prompt: str) -> ScriptPackage:
        import google.generativeai as genai
        genai.configure(api_key=self.settings.gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-pro", system_instruction=system_prompt)
        response = await asyncio.to_thread(model.generate_content, user_prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        return ScriptPackage(**data)

    async def _call_gemini_json(
        self, system_prompt: str, user_prompt: str, schema_keys: list[str]
    ) -> dict[str, Any]:
        import google.generativeai as genai
        genai.configure(api_key=self.settings.gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-pro", system_instruction=system_prompt)
        response = await asyncio.to_thread(model.generate_content, user_prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        return {k: data[k] for k in schema_keys if k in data}

    async def _call_claude(self, system_prompt: str, user_prompt: str) -> ScriptPackage:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=self.settings.anthropic_api_key)
        message = await client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = message.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        return ScriptPackage(**data)

    async def _call_claude_json(
        self, system_prompt: str, user_prompt: str, schema_keys: list[str]
    ) -> dict[str, Any]:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=self.settings.anthropic_api_key)
        message = await client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = message.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        return {k: data[k] for k in schema_keys if k in data}

    async def _call_groq(self, system_prompt: str, user_prompt: str) -> ScriptPackage:
        headers = {
            "Authorization": f"Bearer {self.settings.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.4,
            "response_format": {"type": "json_object"},
        }
        resp = await self.client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        return ScriptPackage(**data)

    async def _call_groq_json(
        self, system_prompt: str, user_prompt: str, schema_keys: list[str]
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.settings.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.4,
            "response_format": {"type": "json_object"},
        }
        resp = await self.client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        return {k: data[k] for k in schema_keys if k in data}


class TTSClient(BaseAPIClient):
    async def synthesize(self, text: str, output_path: Path) -> float:
        if self.settings.tts_provider == "kokoro":
            return await self._kokoro(text, output_path)
        if self.settings.tts_provider == "piper":
            return await self._piper(text, output_path)
        if self.settings.tts_provider == "elevenlabs":
            return await self._elevenlabs(text, output_path)
        if self.settings.tts_provider == "edge_tts":
            return await self._edge_tts(text, output_path)
        raise ValueError(f"Unsupported TTS provider: {self.settings.tts_provider}")

    async def _kokoro(self, text: str, output_path: Path) -> float:
        import soundfile as sf
        from kokoro_onnx import Kokoro

        model_path = str(self.settings.kokoro_model_dir / "kokoro-v1.0.onnx")
        voices_path = str(self.settings.kokoro_model_dir / "voices-v1.0.bin")

        def _run():
            kokoro = Kokoro(model_path, voices_path)
            samples, sample_rate = kokoro.create(
                text,
                voice=self.settings.kokoro_voice,
                speed=1.0,
                lang=self.settings.kokoro_lang,
            )
            sf.write(str(output_path), samples, sample_rate)

        await asyncio.to_thread(_run)
        return self._get_duration(output_path)

    async def _piper(self, text: str, output_path: Path) -> float:
        import wave

        from piper import PiperVoice

        def _run():
            tts_voice = PiperVoice.load(
                str(self.settings.piper_voice_dir / f"{self.settings.piper_voice}.onnx")
            )
            with wave.open(str(output_path), "wb") as wav_file:
                tts_voice.synthesize_wav(text, wav_file)

        await asyncio.to_thread(_run)
        return self._get_duration(output_path)

    async def _elevenlabs(self, text: str, output_path: Path) -> float:
        from elevenlabs import ElevenLabs
        client = ElevenLabs(api_key=self.settings.elevenlabs_api_key)
        voice_id = self.settings.elevenlabs_voice_id or "21m00Tcm4TlvDq8ikWAM"
        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_turbo_v2_5",
        )
        with open(output_path, "wb") as f:
            f.writelines(audio)
        return self._get_duration(output_path)

    async def _edge_tts(self, text: str, output_path: Path) -> float:
        import edge_tts
        voice = self.settings.edge_tts_voice or "en-US-GuyNeural"
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_path))
        return self._get_duration(output_path)

    def _get_duration(self, path: Path) -> float:
        try:
            import soundfile as sf
            with sf.SoundFile(str(path)) as f:
                return len(f) / f.samplerate
        except Exception:
            return 0.0


class ImageGenClient(BaseAPIClient):
    @_tenacity_retry()
    async def generate_image(self, prompt: str, output_path: Path, width: int = 1920, height: int = 1080) -> Path:
        if self.settings.image_gen_provider == "replicate_flux":
            return await self._replicate_flux(prompt, output_path, width, height)
        if self.settings.image_gen_provider == "together_flux":
            return await self._together_flux(prompt, output_path, width, height)
        raise ValueError(f"Unsupported image provider: {self.settings.image_gen_provider}")

    async def generate_thumbnail(self, prompt: str, output_path: Path, width: int = 1280, height: int = 720) -> Path:
        """
        Generates a high-CTR thumbnail image. Uses the configured image provider
        with a thumbnail-optimized prompt emphasizing dramatic lighting and
        high contrast.
        """
        enhanced_prompt = THUMBNAIL_GEN_PROMPT_TEMPLATE.substitute(subject=prompt)
        return await self.generate_image(enhanced_prompt, output_path, width=width, height=height)

    async def generate_images_batch(
        self, prompts: list[tuple[str, Path, int, int]]
    ) -> list[Path | None]:
        """
        Generates multiple images concurrently using a single shared httpx
        client. Each element of ``prompts`` is (prompt, output_path, width, height).
        Returns results in the same order as the input list -- None for any
        that fail after retry so callers can handle gaps gracefully.
        """
        tasks = [
            self._generate_with_backoff(prompt, out, w, h)
            for prompt, out, w, h in prompts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        output: list[Path | None] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                if isinstance(result, RateLimitError):
                    try:
                        result = await self._generate_with_backoff(
                            prompts[i][0], prompts[i][1], prompts[i][2], prompts[i][3]
                        )
                    except Exception:
                        result = None
                else:
                    result = None
            output.append(result if isinstance(result, Path) else None)
        return output

    async def _generate_with_backoff(
        self, prompt: str, output_path: Path, width: int, height: int
    ) -> Path:
        try:
            return await self.generate_image(prompt, output_path, width, height)
        except RateLimitError:
            await asyncio.sleep(10)
            return await self.generate_image(prompt, output_path, width, height)

    @_tenacity_retry()
    async def _replicate_flux(self, prompt: str, output_path: Path, width: int, height: int) -> Path:
        headers = {
            "Authorization": f"Token {self.settings.replicate_api_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "version": "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            "input": {"prompt": prompt, "width": width, "height": height, "num_inference_steps": 28},
        }
        resp = await self.client.post(
            "https://api.replicate.com/v1/predictions", json=payload, headers=headers, timeout=self.timeout
        )
        resp.raise_for_status()
        prediction = resp.json()
        status_url = prediction["urls"]["get"]

        for _ in range(60):
            await asyncio.sleep(3)
            status_resp = await self.client.get(status_url, headers=headers, timeout=self.timeout)
            status_resp.raise_for_status()
            data = status_resp.json()
            if data["status"] == "succeeded":
                url = data["output"]
                img_resp = await self.client.get(url, timeout=self.timeout)
                img_resp.raise_for_status()
                output_path.write_bytes(img_resp.content)
                return output_path
            if data["status"] == "failed":
                raise APIError(f"Replicate image generation failed: {data.get('error')}", provider="replicate")
        raise APIError("Replicate image generation timed out", provider="replicate")

    @_tenacity_retry()
    async def _together_flux(self, prompt: str, output_path: Path, width: int, height: int) -> Path:
        headers = {
            "Authorization": f"Bearer {self.settings.together_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "black-forest-labs/FLUX.1-schnell",
            "prompt": prompt,
            "width": width,
            "height": height,
            "n": 1,
        }
        resp = await self.client.post(
            "https://api.together.xyz/v1/images/generations",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        url = data["data"][0]["url"]
        img_resp = await self.client.get(url, timeout=self.timeout)
        img_resp.raise_for_status()
        output_path.write_bytes(img_resp.content)
        return output_path

    async def _upload_to_replicate(self, image_path: Path) -> str:
        with open(image_path, "rb") as f:
            content = base64.b64encode(f.read()).decode()
        return f"data:application/octet-stream;base64,{content}"

    async def _upload_to_together(self, image_path: Path) -> str:
        with open(image_path, "rb") as f:
            content = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{content}"


class VideoGenClient(BaseAPIClient):
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        super().__init__(settings, client)
        self._video_cache: dict[str, Path] = {}

    @_tenacity_retry()
    async def generate_video_from_image(
        self,
        image_path: Path,
        prompt: str,
        output_path: Path,
        animation: AnimationParams,
    ) -> Path:
        cache_key = f"{image_path.name}:{prompt[:64]}:{animation.duration_seconds}"
        if cache_key in self._video_cache and self._video_cache[cache_key].exists():
            return self._video_cache[cache_key]

        if self.settings.video_gen_provider == "replicate":
            result = await self._replicate_video(image_path, prompt, output_path, animation)
        elif self.settings.video_gen_provider == "together":
            result = await self._together_video(image_path, prompt, output_path, animation)
        elif self.settings.video_gen_provider == "huggingface":
            result = await self._huggingface_video(image_path, prompt, output_path, animation)
        else:
            raise ValueError(f"Unsupported video provider: {self.settings.video_gen_provider}")

        self._video_cache[cache_key] = result
        return result

    async def generate_videos_batch(
        self,
        jobs: list[tuple[Path, str, Path, AnimationParams]],
    ) -> list[Path | None]:
        """
        Generates multiple video segments concurrently using a shared httpx
        client. Each job is (image_path, prompt, output_path, animation).
        Returns results in input order -- None for any that fail after
        retry so callers can handle gaps.
        """
        tasks = [
            self._generate_video_with_backoff(img, prompt, out, anim)
            for img, prompt, out, anim in jobs
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        output: list[Path | None] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                if isinstance(result, RateLimitError):
                    try:
                        result = await self._generate_video_with_backoff(
                            jobs[i][0], jobs[i][1], jobs[i][2], jobs[i][3]
                        )
                    except Exception:
                        result = None
                else:
                    result = None
            output.append(result if isinstance(result, Path) else None)
        return output

    async def _generate_video_with_backoff(
        self, image_path: Path, prompt: str, output_path: Path, animation: AnimationParams
    ) -> Path:
        try:
            return await self.generate_video_from_image(image_path, prompt, output_path, animation)
        except RateLimitError:
            await asyncio.sleep(10)
            return await self.generate_video_from_image(image_path, prompt, output_path, animation)

    @_tenacity_retry()
    async def _replicate_video(
        self, image_path: Path, prompt: str, output_path: Path, animation: AnimationParams
    ) -> Path:
        headers = {
            "Authorization": f"Token {self.settings.replicate_api_token}",
            "Content-Type": "application/json",
        }
        image_url = await self._upload_to_replicate(image_path)
        motion_prompt = VIDEO_GEN_PROMPT_TEMPLATE.substitute(
            scene_description=prompt,
            motion_instruction=animation.camera_movement or "dynamic cinematic push-in with lateral drift",
        )
        payload = {
            "version": "anotherjesse/zeroscope-v2-xl:9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351",
            "input": {
                "image": image_url,
                "prompt": motion_prompt,
                "num_frames": int(animation.duration_seconds * animation.fps),
                "fps": animation.fps,
            },
        }
        resp = await self.client.post(
            "https://api.replicate.com/v1/predictions", json=payload, headers=headers, timeout=self.timeout
        )
        resp.raise_for_status()
        prediction = resp.json()
        status_url = prediction["urls"]["get"]

        for _ in range(120):
            await asyncio.sleep(5)
            status_resp = await self.client.get(status_url, headers=headers, timeout=self.timeout)
            status_resp.raise_for_status()
            data = status_resp.json()
            if data["status"] == "succeeded":
                url = data["output"]
                vid_resp = await self.client.get(url, timeout=self.timeout)
                vid_resp.raise_for_status()
                output_path.write_bytes(vid_resp.content)
                return output_path
            if data["status"] == "failed":
                raise APIError(f"Replicate video generation failed: {data.get('error')}", provider="replicate")
        raise APIError("Replicate video generation timed out", provider="replicate")

    @_tenacity_retry()
    async def _together_video(
        self, image_path: Path, prompt: str, output_path: Path, animation: AnimationParams
    ) -> Path:
        headers = {
            "Authorization": f"Bearer {self.settings.together_api_key}",
            "Content-Type": "application/json",
        }
        motion_prompt = VIDEO_GEN_PROMPT_TEMPLATE.substitute(
            scene_description=prompt,
            motion_instruction=animation.camera_movement or "dynamic cinematic push-in with lateral drift",
        )
        payload = {
            "model": "tencent/HunyuanVideo",
            "prompt": motion_prompt,
            "input_image": await self._upload_to_together(image_path),
            "duration": min(int(animation.duration_seconds), 10),
        }
        resp = await self.client.post(
            "https://api.together.xyz/v1/videos/generations",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        generation_id = data.get("id")
        status_url = f"https://api.together.xyz/v1/videos/generations/{generation_id}"

        for _ in range(60):
            await asyncio.sleep(5)
            status_resp = await self.client.get(status_url, headers=headers, timeout=self.timeout)
            status_resp.raise_for_status()
            sdata = status_resp.json()
            if sdata.get("status") == "completed":
                url = sdata["output"]["video_url"]
                vid_resp = await self.client.get(url, timeout=self.timeout)
                vid_resp.raise_for_status()
                output_path.write_bytes(vid_resp.content)
                return output_path
            if sdata.get("status") == "failed":
                raise APIError("Together video generation failed", provider="together")
        raise APIError("Together video generation timed out", provider="together")

    @_tenacity_retry()
    async def _huggingface_video(
        self, image_path: Path, prompt: str, output_path: Path, animation: AnimationParams
    ) -> Path:
        headers = {"Authorization": f"Bearer {self.settings.huggingface_api_key}"}
        motion_prompt = VIDEO_GEN_PROMPT_TEMPLATE.substitute(
            scene_description=prompt,
            motion_instruction=animation.camera_movement or "dynamic cinematic push-in with lateral drift",
        )
        async with self.client.post(
            "https://api-inference.huggingface.co/models/ByteDance/AnimateDiff-Lightning",
            headers=headers,
            json={"inputs": motion_prompt},
            timeout=self.timeout,
        ) as resp:
            if resp.status_code == 503:
                raise RateLimitError("HuggingFace model loading", provider="huggingface")
            resp.raise_for_status()
            output_path.write_bytes(await resp.read())
            return output_path


class AssetManager:
    def __init__(self, work_dir: Path, output_dir: Path):
        self.work_dir = work_dir
        self.output_dir = output_dir
        self.assets: list[dict] = []

    def scene_image_path(self, scene_num: int) -> Path:
        return self.work_dir / f"scene_{scene_num:03d}.png"

    def scene_video_path(self, scene_num: int) -> Path:
        return self.work_dir / f"scene_{scene_num:03d}.mp4"

    def voiceover_path(self) -> Path:
        return self.work_dir / "voiceover.wav"

    def final_video_path(self) -> Path:
        return self.output_dir / "final_video.mp4"

    def metadata_path(self) -> Path:
        return self.output_dir / "metadata.json"

    def thumbnail_path(self) -> Path:
        return self.output_dir / "thumbnail.png"

    def write_metadata(self, title: str, description: str, tags: list[str], thumbnail_path: Path | None = None, video_path: Path | None = None) -> Path:
        metadata = {
            "title": title,
            "description": description,
            "tags": tags,
            "thumbnail_path": str(thumbnail_path) if thumbnail_path else None,
            "video_path": str(video_path) if video_path else None,
        }
        path = self.metadata_path()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return path

    def log_asset(self, **kwargs) -> None:
        self.assets.append(kwargs)
