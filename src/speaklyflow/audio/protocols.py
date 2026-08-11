"""Public protocol implemented by audio backends."""

from collections.abc import AsyncIterator
from typing import Protocol

from .types import AudioChunk, AudioFormat


class AudioIO(Protocol):
    """Duplex audio boundary used by a voice session."""

    @property
    def input_format(self) -> AudioFormat:
        """Format produced by :meth:`capture`."""

        ...

    @property
    def output_format(self) -> AudioFormat:
        """Format accepted by :meth:`write`."""

        ...

    async def start(self) -> None:
        """Open input and output resources."""

        ...

    def capture(self) -> AsyncIterator[AudioChunk]:
        """Yield microphone chunks until the audio component closes."""

        ...

    async def write(self, chunk: AudioChunk) -> None:
        """Write a chunk to the output device."""

        ...

    async def interrupt_playback(self) -> None:
        """Stop playback and flush the device output buffer."""

        ...

    async def close(self) -> None:
        """Close resources and unblock pending operations."""

        ...
