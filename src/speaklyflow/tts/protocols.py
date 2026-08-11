"""Public protocol implemented by speech synthesizers."""

from collections.abc import AsyncIterable
from typing import Protocol, Self, TypeAlias

from ..audio import AudioChunk, AudioFormat
from .types import TTSResult

TextInput: TypeAlias = str | AsyncIterable[str]


class TTSStream(Protocol):
    """Audio iterator with explicit cancellation and final results."""

    def __aiter__(self) -> Self: ...

    async def __anext__(self) -> AudioChunk: ...

    async def aclose(self) -> None:
        """Cancel synthesis and release its resources."""

        ...

    async def result(self) -> TTSResult:
        """Wait for synthesis to stop and return its final result."""

        ...


class TTS(Protocol):
    """Provider-neutral streaming text-to-speech interface."""

    @property
    def output_format(self) -> AudioFormat:
        """Format of synthesized audio chunks."""

        ...

    async def start(self) -> None:
        """Initialize the provider before synthesis."""

        ...

    def synthesize(self, text: TextInput) -> TTSStream:
        """Stream audio for text chunks."""

        ...

    async def close(self) -> None:
        """Cancel active synthesis and release provider resources."""

        ...
