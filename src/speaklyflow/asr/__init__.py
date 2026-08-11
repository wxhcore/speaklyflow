"""Speech recognition interfaces and implementations."""

from .errors import ASRError, ASRFormatError, ASRStateError
from .protocols import ASR
from .sensevoice import SenseVoiceASR
from .types import Transcript

__all__ = [
    "ASR",
    "ASRError",
    "ASRFormatError",
    "ASRStateError",
    "SenseVoiceASR",
    "Transcript",
]
