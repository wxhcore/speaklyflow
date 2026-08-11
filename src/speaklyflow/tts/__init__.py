"""Text-to-speech interfaces and implementations."""

from .errors import TTSError, TTSStateError
from .protocols import TTS, TextInput, TTSStream
from .types import TTSResult
from .volcengine import VolcengineTTS

__all__ = [
    "TTS",
    "TTSError",
    "TTSResult",
    "TTSStateError",
    "TTSStream",
    "TextInput",
    "VolcengineTTS",
]
