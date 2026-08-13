"""Application configuration and persistence."""

from pathlib import Path
from typing import Any, Literal

import bumblehive
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .storage import load_json, save_json


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AudioConfig(_ConfigModel):
    input_device: int | str | None = None
    output_device: int | str | None = None
    block_ms: int = Field(default=20, gt=0)
    capture_buffer_ms: int = Field(default=500, gt=0)
    latency: float | Literal["low", "high"] = "low"
    echo_cancellation: Literal["disabled", "preferred", "required"] = "preferred"


class VADConfig(_ConfigModel):
    threshold: float = Field(default=0.5, ge=0, le=1)
    speech_start_ms: int = Field(default=200, gt=0)
    speech_end_ms: int = Field(default=400, gt=0)


class SenseVoiceSettings(_ConfigModel):
    model_dir: Path
    threads: int = Field(default=4, gt=0)
    language: Literal["auto", "zh", "en", "ja", "ko", "yue"] = "auto"
    use_itn: bool = True


class SenseVoiceASRConfig(_ConfigModel):
    provider: Literal["sensevoice"]
    settings: SenseVoiceSettings


class VolcengineTTSSettings(_ConfigModel):
    api_key: str = Field(min_length=1)
    voice: str = "zh_female_vv_uranus_bigtts"
    resource_id: str = "seed-tts-2.0"
    sample_rate: Literal[8000, 16000, 22050, 24000, 32000, 44100, 48000] = 48000


class VolcengineTTSConfig(_ConfigModel):
    provider: Literal["volcengine"]
    settings: VolcengineTTSSettings


class AppConfig(_ConfigModel):
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    asr: SenseVoiceASRConfig
    bumblehive: dict[str, Any]
    tts: VolcengineTTSConfig

    @field_validator("bumblehive")
    @classmethod
    def validate_bumblehive(cls, value: dict[str, Any]) -> dict[str, Any]:
        config = bumblehive.BumblehiveConfig.from_mapping(value)
        if not config.provider.api_key:
            raise ValueError("bumblehive.provider.api_key is required")
        return value


def load_config(path: Path) -> AppConfig | None:
    """Load the current configuration, or return None when it does not exist."""

    data = load_json(path)
    return AppConfig.model_validate(data) if data is not None else None


async def save_config(path: Path, config: AppConfig) -> None:
    """Atomically replace the application configuration."""

    await save_json(path, config.model_dump(mode="json"))
