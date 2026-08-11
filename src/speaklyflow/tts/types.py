"""Result types shared by text-to-speech providers."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TTSResult:
    """Summary of one synthesis stream.

    Parameters:
        input_characters: Characters successfully submitted to the provider.
        audio_bytes: PCM bytes received from the provider.
        completed: Whether the provider finished the synthesis normally.
        provider_usage: Optional provider-specific numeric usage values.
    """

    input_characters: int
    audio_bytes: int
    completed: bool
    provider_usage: Mapping[str, int | float] | None = None
