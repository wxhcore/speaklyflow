"""Public protocol implemented by speech recognizers."""

from typing import Protocol

from ..audio import AudioChunk, AudioFormat
from .types import Transcript


class ASR(Protocol):
    """Provider-neutral segmented speech recognizer."""

    async def start(self, input_format: AudioFormat) -> None:
        """Initialize the recognizer for an input audio format."""

        ...

    async def transcribe(self, audio: AudioChunk) -> Transcript:
        """Transcribe one complete speech segment."""

        ...

    async def close(self) -> None:
        """Release model and worker resources."""

        ...
