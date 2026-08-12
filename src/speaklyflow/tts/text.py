"""Segment streamed agent text for incremental speech synthesis."""

_STRONG_BREAKS = frozenset("。！？!?；;\n")
_SOFT_BREAKS = frozenset("，,、：:")


class TextSegmenter:
    """Collect text deltas into chunks suitable for streaming TTS."""

    def __init__(
        self,
        *,
        first_chunk_chars: int = 6,
        chunk_chars: int = 24,
    ) -> None:
        """Configure the first and subsequent hard chunk limits."""

        if first_chunk_chars <= 0:
            raise ValueError("first_chunk_chars must be greater than zero")
        if chunk_chars <= 0:
            raise ValueError("chunk_chars must be greater than zero")

        self._first_chunk_chars = first_chunk_chars
        self._chunk_chars = chunk_chars
        self._limit = first_chunk_chars
        self._buffer = ""

    def push(self, delta: str) -> list[str]:
        """Append one text delta and return chunks ready for synthesis."""

        self._buffer += delta
        chunks: list[str] = []

        while self._buffer:
            end = self._break_index()
            if end is None:
                break

            text = self._take(end)
            if text:
                chunks.append(text)
                self._limit = self._chunk_chars

        return chunks

    def flush(self) -> str | None:
        """Return remaining text and prepare for a new response."""

        text = self._take(len(self._buffer))
        self._limit = self._first_chunk_chars
        return text or None

    def reset(self) -> None:
        """Discard remaining text and prepare for a new response."""

        self._buffer = ""
        self._limit = self._first_chunk_chars

    def _break_index(self) -> int | None:
        for index, char in enumerate(self._buffer[: self._limit]):
            end = index + 1
            if char in _STRONG_BREAKS:
                return end
            if char in _SOFT_BREAKS and end >= 3:
                return end

        if len(self._buffer) >= self._limit:
            return self._limit
        return None

    def _take(self, end: int) -> str:
        text = self._buffer[:end].strip()
        self._buffer = self._buffer[end:]
        return text
