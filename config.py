from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


def _to_upper_alias(name: str) -> str:
    """Generate the canonical env-var alias for a field (e.g. llm_provider -> LLM_PROVIDER).

    This lets the model be populated from both conventional uppercase environment
    variables (LLM_PROVIDER, TTS_PROVIDER, ...) and the raw field names
    (llm_provider, tts_provider, ...).
    """
    return name.upper()


class Settings(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=_to_upper_alias,
    )
    llm_provider: str = Field(default="gemini", description="LLM provider: gemini | claude | groq")
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")

    tts_provider: str = Field(default="kokoro", description="TTS provider: kokoro | piper | elevenlabs | edge_tts")
    elevenlabs_api_key: Optional[str] = Field(default=None, alias="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: Optional[str] = Field(default="21m00Tcm4TlvDq8ikWAM", description="Default ElevenLabs voice")
    kokoro_model_dir: Path = Field(default=Path("models"), description="Directory containing kokoro-v1.0.onnx and voices-v1.0.bin")
    kokoro_voice: str = Field(default="am_adam", description="Kokoro voice ID (e.g., am_adam, bf_emma, etc.)")
    kokoro_lang: str = Field(default="en-us", description="Kokoro language code")
    piper_voice: str = Field(default="en_US-amy-low", description="Piper voice ID (e.g., en_US-amy-low)")
    piper_voice_dir: Path = Field(default=Path("voices"), description="Directory for Piper voice downloads")
    edge_tts_voice: str = Field(default="en-US-GuyNeural", description="edge-tts voice identifier")

    image_gen_provider: str = Field(default="replicate_flux", description="Image provider: replicate_flux | together_flux")
    replicate_api_token: Optional[str] = Field(default=None, alias="REPLICATE_API_TOKEN")
    together_api_key: Optional[str] = Field(default=None, alias="TOGETHER_API_KEY")

    video_gen_provider: str = Field(default="replicate", description="Video provider: replicate | together | huggingface")
    huggingface_api_key: Optional[str] = Field(default=None, alias="HUGGINGFACE_API_KEY")

    thumbnail_gen_provider: str = Field(default="replicate_flux", description="Thumbnail provider: replicate_flux | together_flux")

    work_dir: Path = Field(default=Path("work"))
    output_dir: Path = Field(default=Path("output"))
    log_level: str = Field(default="INFO")

    target_duration_seconds: float = Field(default=180.0, ge=30.0, le=600.0, description="Target total video length")
    resolution: str = Field(default="1920x1080", description="Target resolution e.g. 1920x1080 or 1080x1920")
    fps: int = Field(default=24, ge=12, le=60)

    @field_validator("work_dir", "output_dir", "kokoro_model_dir", "piper_voice_dir", mode="before")
    @classmethod
    def resolve_path(cls, v):
        return Path(v).resolve()

    def ensure_dirs(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.kokoro_model_dir.mkdir(parents=True, exist_ok=True)
        self.piper_voice_dir.mkdir(parents=True, exist_ok=True)

    def require_llm(self) -> None:
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when llm_provider=gemini")
        if self.llm_provider == "claude" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when llm_provider=claude")
        if self.llm_provider == "groq" and not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required when llm_provider=groq")

    def require_image_gen(self) -> None:
        if self.image_gen_provider == "replicate_flux" and not self.replicate_api_token:
            raise ValueError("REPLICATE_API_TOKEN is required when image_gen_provider=replicate_flux")
        if self.image_gen_provider == "together_flux" and not self.together_api_key:
            raise ValueError("TOGETHER_API_KEY is required when image_gen_provider=together_flux")

    def require_video_gen(self) -> None:
        if self.video_gen_provider == "replicate" and not self.replicate_api_token:
            raise ValueError("REPLICATE_API_TOKEN is required when video_gen_provider=replicate")
        if self.video_gen_provider == "together" and not self.together_api_key:
            raise ValueError("TOGETHER_API_KEY is required when video_gen_provider=together")
        if self.video_gen_provider == "huggingface" and not self.huggingface_api_key:
            raise ValueError("HUGGINGFACE_API_KEY is required when video_gen_provider=huggingface")

    def require_tts(self) -> None:
        if self.tts_provider == "elevenlabs" and not self.elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY is required when tts_provider=elevenlabs")
        if self.tts_provider == "kokoro":
            if not (self.kokoro_model_dir / "kokoro-v1.0.onnx").exists():
                raise ValueError(
                    f"Kokoro model not found in {self.kokoro_model_dir}. "
                    "Download from https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
                )
            if not (self.kokoro_model_dir / "voices-v1.0.bin").exists():
                raise ValueError(
                    f"Kokoro voices file not found in {self.kokoro_model_dir}. "
                    "Download from https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
                )
        if self.tts_provider == "piper" and not self.piper_voice.strip():
            raise ValueError("piper_voice must be set when tts_provider=piper")

    @property
    def metadata_json_path(self) -> Path:
        return self.output_dir / "metadata.json"

    @property
    def thumbnail_path(self) -> Path:
        return self.output_dir / "thumbnail.png"

    @property
    def final_video_path(self) -> Path:
        return self.output_dir / "final_video.mp4"


def load_settings() -> Settings:
    try:
        settings = Settings(**os.environ)
    except ValidationError as exc:
        raise ValueError(f"Invalid pipeline configuration from environment: {exc}") from exc
    settings.ensure_dirs()
    settings.require_llm()
    settings.require_image_gen()
    settings.require_video_gen()
    settings.require_tts()
    return settings

