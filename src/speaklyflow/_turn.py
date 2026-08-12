"""Run one interruptible Bumblehive response through TTS and audio output."""

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

import bumblehive
from bumblehive.agent import AgentRunResult
from bumblehive.observability import (
    MODEL_RESPONSE_FINISHED,
    MODEL_STREAM_CONTENT_DELTA,
    MODEL_STREAM_REFUSAL_DELTA,
    TOOL_CALL_FINISHED,
    TOOL_CALLS_FINISHED,
    TOOL_CALLS_STARTED,
    AgentEvent,
)

from .agent import AgentTurn, BumblehiveAgent
from .audio import AudioChunk, AudioIO
from .tts import TTS, TextSegmenter, TTSStream, TTSTextMark

_TEXT_EVENTS = (MODEL_STREAM_CONTENT_DELTA, MODEL_STREAM_REFUSAL_DELTA)
_Message = dict[str, Any]


@dataclass(frozen=True, slots=True)
class _TurnOutcome:
    interrupted: bool
    messages: list[_Message]


@dataclass(frozen=True, slots=True)
class _TextDelta:
    response_index: int
    text: str


@dataclass(frozen=True, slots=True)
class _ResponseEnd:
    response_index: int


class _AgentDone:
    __slots__ = ()


@dataclass(frozen=True, slots=True)
class _ResponsePlayback:
    text: str


_AGENT_DONE = _AgentDone()
_SpeechItem = _TextDelta | _ResponseEnd | _AgentDone


class _PlaybackTracker:
    """Map TTS text marks onto output progress for one synthesis stream."""

    def __init__(self, audio: AudioIO) -> None:
        self._audio = audio
        self._start_frame = audio.played_frames
        self._marks: list[TTSTextMark] = []
        self._submitted: dict[int, str] = {}

    def submit(self, response_index: int, text: str) -> None:
        self._submitted[response_index] = self._submitted.get(response_index, "") + text

    def mark(self, mark: TTSTextMark) -> None:
        self._marks.append(mark)

    def response_playback(self, count: int) -> list[_ResponsePlayback]:
        remaining = self._spoken_text()
        responses: list[_ResponsePlayback] = []
        for index in range(count):
            submitted = self._submitted.get(index, "")
            heard = remaining[: len(submitted)]
            remaining = remaining[len(heard) :]
            responses.append(_ResponsePlayback(text=heard))
        return responses

    def _spoken_text(self) -> str:
        played = self._audio.played_frames - self._start_frame
        if played <= 0:
            return ""
        return "".join(mark.text for mark in self._marks if mark.at_frame <= played)


class _TurnJournal:
    """Project partial Bumblehive events into valid interrupted history."""

    def __init__(self, prompt: str) -> None:
        self._messages: list[_Message] = [{"role": "user", "content": prompt}]
        self._draft_text = ""
        self._response_index = 0
        self._started_tool_ids: set[str] = set()
        self._active_tool_ids: set[str] = set()
        self._tools_finished = asyncio.Event()
        self._tools_finished.set()

    @property
    def response_index(self) -> int:
        return self._response_index

    @property
    def response_count(self) -> int:
        count = sum(message.get("role") == "assistant" for message in self._messages)
        return count + bool(self._draft_text)

    @property
    def tools_active(self) -> bool:
        return bool(self._active_tool_ids)

    async def wait_until_tools_finish(self) -> None:
        await self._tools_finished.wait()

    def record(self, event: AgentEvent) -> None:
        if event.kind in _TEXT_EVENTS:
            delta = event.payload.get("delta")
            if isinstance(delta, str):
                self._draft_text += delta
            return

        if event.kind == MODEL_RESPONSE_FINISHED:
            message = event.payload.get("message")
            if isinstance(message, Mapping):
                self._messages.append(dict(message))
            else:
                self._messages.append(
                    {"role": "assistant", "content": self._draft_text}
                )
            self._draft_text = ""
            self._response_index += 1
            return

        if event.kind == TOOL_CALLS_STARTED:
            tool_ids = self._latest_tool_call_ids()
            self._started_tool_ids.update(tool_ids)
            self._active_tool_ids = set(tool_ids)
            if tool_ids:
                self._tools_finished.clear()
            return

        if event.kind == TOOL_CALL_FINISHED:
            message = event.payload.get("tool_result")
            if isinstance(message, Mapping):
                self._messages.append(dict(message))
            return

        if event.kind == TOOL_CALLS_FINISHED:
            self._active_tool_ids.clear()
            self._tools_finished.set()

    def interrupted_messages(
        self,
        playback: list[_ResponsePlayback],
    ) -> list[_Message]:
        messages = [dict(message) for message in self._messages]
        if self._draft_text:
            messages.append({"role": "assistant", "content": self._draft_text})

        tool_results = _tool_results(messages)
        interrupted: list[_Message] = []
        response_index = 0

        for message in messages:
            role = message.get("role")
            if role == "tool":
                continue
            if role != "assistant":
                interrupted.append(message)
                continue

            heard = (
                playback[response_index]
                if response_index < len(playback)
                else _ResponsePlayback("")
            )
            response_index += 1
            tool_ids = _tool_call_ids(message)

            if tool_ids:
                executed = set(tool_ids) <= self._started_tool_ids and all(
                    tool_id in tool_results for tool_id in tool_ids
                )
                if executed:
                    assistant = dict(message)
                    assistant["content"] = heard.text or None
                    interrupted.append(assistant)
                    interrupted.extend(tool_results[tool_id] for tool_id in tool_ids)
                elif heard.text:
                    interrupted.append({"role": "assistant", "content": heard.text})
                continue

            if heard.text:
                interrupted.append({"role": "assistant", "content": heard.text})

        return interrupted

    def _latest_tool_call_ids(self) -> list[str]:
        for message in reversed(self._messages):
            if message.get("role") == "assistant":
                return _tool_call_ids(message)
        return []


class _TurnRunner:
    """Coordinate one agent turn without committing conversation history."""

    def __init__(
        self,
        *,
        prompt: str,
        history: bumblehive.MessageHistory,
        audio: AudioIO,
        agent: BumblehiveAgent,
        tts: TTS,
    ) -> None:
        self._prompt = prompt
        self._history = history
        self._audio = audio
        self._agent = agent
        self._tts = tts

        self._journal = _TurnJournal(prompt)
        self._playback = _PlaybackTracker(audio)
        # Agent events must keep flowing while playback is slower so tool state
        # remains available to interruption cleanup.
        self._speech_items: asyncio.Queue[_SpeechItem] = asyncio.Queue()
        self._interruption = asyncio.Event()
        self._speech_stopped = asyncio.Event()

        self._turn: AgentTurn | None = None
        self._speech: TTSStream | None = None
        self._agent_task: asyncio.Task[AgentRunResult] | None = None
        self._speech_task: asyncio.Task[None] | None = None

    def interrupt(self) -> None:
        """Request interruption without blocking microphone capture."""

        self._interruption.set()

    async def run(self) -> _TurnOutcome:
        self._turn = self._agent.stream(self._prompt, history=self._history)
        try:
            self._speech = self._tts.synthesize(self._speech_text())
        except BaseException:
            await asyncio.gather(self._turn.aclose(), return_exceptions=True)
            raise

        self._agent_task = asyncio.create_task(
            self._consume_agent(),
            name="speaklyflow-agent-turn",
        )
        self._speech_task = asyncio.create_task(
            self._consume_speech(),
            name="speaklyflow-tts-playback",
        )
        completion = asyncio.create_task(
            self._complete(),
            name="speaklyflow-turn-completion",
        )
        interruption = asyncio.create_task(
            self._interruption.wait(),
            name="speaklyflow-turn-interruption",
        )
        outcome_ready = False

        try:
            done, _ = await asyncio.wait(
                {completion, interruption},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if completion in done:
                result = await completion
                outcome = _TurnOutcome(
                    interrupted=False,
                    messages=[dict(message) for message in result.messages],
                )
            else:
                outcome = await self._interrupt_turn()

            outcome_ready = True
            return outcome
        finally:
            interruption.cancel()
            await asyncio.gather(interruption, return_exceptions=True)
            if not completion.done():
                completion.cancel()
            await asyncio.gather(completion, return_exceptions=True)
            await self._close(suppress_errors=not outcome_ready)

    async def _consume_agent(self) -> AgentRunResult:
        turn = self._require_turn()
        try:
            async for event in turn:
                response_index = self._journal.response_index
                self._journal.record(event)
                if self._speech_stopped.is_set():
                    continue

                if event.kind in _TEXT_EVENTS:
                    delta = event.payload.get("delta")
                    if isinstance(delta, str) and delta:
                        self._speech_items.put_nowait(_TextDelta(response_index, delta))
                elif event.kind == MODEL_RESPONSE_FINISHED:
                    self._speech_items.put_nowait(_ResponseEnd(response_index))
            return await turn.result()
        finally:
            self._speech_items.put_nowait(_AGENT_DONE)

    async def _consume_speech(self) -> None:
        speech = self._require_speech()
        async for output in speech:
            if isinstance(output, TTSTextMark):
                self._playback.mark(output)
                continue
            if not isinstance(output, AudioChunk):
                raise TypeError(f"Unsupported TTS output: {type(output).__name__}")
            if self._interruption.is_set():
                continue
            await self._audio.write(output)

        result = await speech.result()
        if not result.completed:
            raise RuntimeError("TTS synthesis did not complete")
        await self._audio.wait_for_playback()

    async def _speech_text(self) -> AsyncIterator[str]:
        segmenter = TextSegmenter()
        response_index = 0

        while True:
            item = await self._speech_items.get()
            if isinstance(item, _AgentDone):
                remaining = segmenter.flush()
                if remaining:
                    self._playback.submit(response_index, remaining)
                    yield remaining
                return

            if isinstance(item, _TextDelta):
                response_index = item.response_index
                for text in segmenter.push(item.text):
                    self._playback.submit(response_index, text)
                    yield text
                continue

            if isinstance(item, _ResponseEnd):
                remaining = segmenter.flush()
                if remaining:
                    self._playback.submit(item.response_index, remaining)
                    yield remaining
                response_index = item.response_index + 1

    async def _complete(self) -> AgentRunResult:
        agent_task = self._require_agent_task()
        speech_task = self._require_speech_task()
        await asyncio.gather(agent_task, speech_task)
        return agent_task.result()

    async def _interrupt_turn(self) -> _TurnOutcome:
        self._speech_stopped.set()
        await self._audio.interrupt_playback()
        await self._stop_speech()
        await asyncio.sleep(0)

        agent_task = self._require_agent_task()
        if self._journal.tools_active and not agent_task.done():
            tools_finished = asyncio.create_task(
                self._journal.wait_until_tools_finish(),
                name="speaklyflow-tools-finished",
            )
            try:
                await asyncio.wait(
                    {agent_task, tools_finished},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                tools_finished.cancel()
                await asyncio.gather(tools_finished, return_exceptions=True)

        if not agent_task.done():
            agent_task.cancel()
        await asyncio.gather(agent_task, return_exceptions=True)

        playback = self._playback.response_playback(self._journal.response_count)
        return _TurnOutcome(
            interrupted=True,
            messages=self._journal.interrupted_messages(playback),
        )

    async def _stop_speech(self) -> None:
        speech_task = self._require_speech_task()
        if not speech_task.done():
            speech_task.cancel()
        await asyncio.gather(speech_task, return_exceptions=True)
        await self._require_speech().aclose()

    async def _close(self, *, suppress_errors: bool) -> None:
        errors: list[Exception] = []
        for task in (self._speech_task, self._agent_task):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(
                task
                for task in (self._speech_task, self._agent_task)
                if task is not None
            ),
            return_exceptions=True,
        )

        for stream in (self._speech, self._turn):
            if stream is None:
                continue
            try:
                await stream.aclose()
            except Exception as error:  # noqa: BLE001 - both streams must close
                errors.append(error)

        if errors and not suppress_errors:
            raise errors[0]

    def _require_turn(self) -> AgentTurn:
        if self._turn is None:
            raise RuntimeError("Agent turn has not started")
        return self._turn

    def _require_speech(self) -> TTSStream:
        if self._speech is None:
            raise RuntimeError("TTS stream has not started")
        return self._speech

    def _require_agent_task(self) -> asyncio.Task[AgentRunResult]:
        if self._agent_task is None:
            raise RuntimeError("Agent task has not started")
        return self._agent_task

    def _require_speech_task(self) -> asyncio.Task[None]:
        if self._speech_task is None:
            raise RuntimeError("Speech task has not started")
        return self._speech_task


def _tool_results(messages: list[_Message]) -> dict[str, _Message]:
    return {
        message["tool_call_id"]: dict(message)
        for message in messages
        if message.get("role") == "tool"
        and isinstance(message.get("tool_call_id"), str)
    }


def _tool_call_ids(message: Mapping[str, Any]) -> list[str]:
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return []

    identifiers: list[str] = []
    for call in calls:
        if not isinstance(call, Mapping):
            continue
        identifier = call.get("id")
        if isinstance(identifier, str) and identifier:
            identifiers.append(identifier)
    return identifiers
