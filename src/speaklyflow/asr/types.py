"""Speech recognition value objects."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Transcript:
    """Normalized text produced by a speech recognizer."""

    text: str
    is_final: bool = True
    language: str | None = None
