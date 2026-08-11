"""Bumblehive runtime integration."""

from collections.abc import AsyncIterator, Iterable
from dataclasses import replace
from typing import Self

import bumblehive
from bumblehive.agent import AgentRunResult
from bumblehive.config import ConfigInput
from bumblehive.observability import AgentEvent, AsyncEventStream
from bumblehive.tools import Tool


class AgentTurn:
    """Bumblehive stream with an isolated conversation history."""

    def __init__(
        self,
        stream: AsyncEventStream[AgentRunResult],
        history: bumblehive.MessageHistory,
    ) -> None:
        self._stream = stream
        self._history = history

    def __aiter__(self) -> AsyncIterator[AgentEvent]:
        return self._stream.__aiter__()

    async def result(self) -> AgentRunResult:
        """Return the native result with messages safe for conversation storage."""

        result = await self._stream.result()
        return replace(result, messages=self._history.get_history())

    async def aclose(self) -> None:
        """Cancel unfinished work for this turn."""

        await self._stream.aclose()


class BumblehiveAgent:
    """Create streamed Bumblehive turns without committing history early."""

    def __init__(
        self,
        config: ConfigInput = None,
        *,
        tools: Iterable[Tool] = (),
    ) -> None:
        self._runtime = bumblehive.from_config(config)
        for tool in tools:
            self._runtime.tools.register(tool)

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()

    async def start(self) -> None:
        """Initialize Bumblehive tools and MCP connections."""

        await self._runtime.initialize_tools()

    def stream(
        self,
        prompt: str,
        *,
        history: bumblehive.MessageHistory | None = None,
    ) -> AgentTurn:
        """Start a turn using a private snapshot of the supplied history."""

        local_history = bumblehive.MessageHistory(
            history.get_history() if history is not None else None,
            conversation_id=(history.conversation_id if history is not None else None),
        )
        stream = self._runtime.stream(prompt, history=local_history)
        return AgentTurn(stream, local_history)

    async def close(self) -> None:
        """Release Bumblehive resources."""

        await self._runtime.close()
