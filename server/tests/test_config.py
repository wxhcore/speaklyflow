import json
from pathlib import Path

import pytest
from speaklyflow_server.config import (
    AppConfig,
    load_config,
    save_config,
)


@pytest.mark.asyncio
async def test_config_round_trip_preserves_api_keys(
    tmp_path: Path,
    app_config: AppConfig,
) -> None:
    path = tmp_path / "nested" / "config.json"

    await save_config(path, app_config)

    assert load_config(path) == app_config
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["tts"]["settings"]["api_key"] == "tts-secret"
    assert saved["bumblehive"]["provider"]["api_key"] == "agent-secret"
    assert path.stat().st_mode & 0o777 == 0o600


def test_invalid_config_is_not_silently_replaced(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"tts": {}}), encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(path)


def test_flat_provider_settings_are_rejected(config_data: dict[str, object]) -> None:
    asr = config_data["asr"]
    assert isinstance(asr, dict)
    settings = asr.pop("settings")
    assert isinstance(settings, dict)
    asr.update(settings)

    with pytest.raises(ValueError):
        AppConfig.model_validate(config_data)


def test_bumblehive_api_key_is_required(config_data: dict[str, object]) -> None:
    bumblehive = config_data["bumblehive"]
    assert isinstance(bumblehive, dict)
    provider = bumblehive["provider"]
    assert isinstance(provider, dict)
    provider.pop("api_key")

    with pytest.raises(ValueError, match="bumblehive.provider.api_key is required"):
        AppConfig.model_validate(config_data)
