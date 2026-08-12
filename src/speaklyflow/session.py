"""Run a local voice conversation from microphone input to speech output."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable

import bumblehive
from bumblehive.observability import (
    MODEL_STREAM_CONTENT_DELTA,
    MODEL_STREAM_REFUSAL_DELTA,
)

from .agent import AgentTurn, BumblehiveAgent
from .asr import ASR, SpeechSegmenter
from .audio import AudioChunk, AudioFormatError, AudioIO
from .tts import TTS, TextSegmenter, TTSStream
from .vad import VAD

_TEXT_EVENTS = (MODEL_STREAM_CONTENT_DELTA, MODEL_STREAM_REFUSAL_DELTA)
_Close = Callable[[], Awaitable[None]]


class VoiceSession:
    """Run continuous non-interruptible voice turns with managed resources."""

    def __init__(
        self,
        *,
        audio: AudioIO,
        vad: VAD,
        asr: ASR,
        agent: BumblehiveAgent,
        tts: TTS,
        history: bumblehive.MessageHistory | None = None,
    ) -> None:
        """Store conversation components without starting external resources."""

        if audio.output_format != tts.output_format:
            raise AudioFormatError(
                f"VoiceSession audio output requires {tts.output_format!r}, "
                f"received {audio.output_format!r}"
            )

        self._audio = audio
        self._vad = vad
        self._asr = asr
        self._agent = agent
        self._tts = tts
        self._history = history if history is not None else bumblehive.MessageHistory()

        self._segmenter = SpeechSegmenter()
        self._segments: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=1)
        self._listening = asyncio.Event()
        self._capture_task: asyncio.Task[None] | None = None
        self._started_closers: list[_Close] = []
        self._used = False

    @property
    def history(self) -> bumblehive.MessageHistory:
        """Conversation history committed by completed turns."""

        return self._history

    async def run(self) -> None:
        """Start components and run turns until cancelled or an error occurs."""

        if self._used:
            raise RuntimeError("VoiceSession can only be run once")
        self._used = True

        try:
            await self._start_components()
            self._listening.set()
            self._capture_task = asyncio.create_task(
                self._capture(),
                name="speaklyflow-audio-capture",
            )
            await self._run_conversation()
        except BaseException:
            await self._shutdown(suppress_errors=True)
            raise
        else:
            await self._shutdown(suppress_errors=False)

    async def _run_conversation(self) -> None:
        while True:
            segment = await self._next_segment()
            transcript = await self._asr.transcribe(segment)
            if transcript.text:
                await self._run_turn(transcript.text)

            self._segmenter.reset()
            self._listening.set()

    async def _capture(self) -> None:
        async for chunk in self._audio.capture():
            if not self._listening.is_set():
                continue

            state = await self._vad.analyze(chunk)
            segment = self._segmenter.push(chunk, state)
            if segment is not None:
                self._listening.clear()
                await self._segments.put(segment)

    async def _next_segment(self) -> AudioChunk:
        capture_task = self._capture_task
        if capture_task is None:
            raise RuntimeError("VoiceSession audio capture is not running")

        receive = asyncio.create_task(
            self._segments.get(),
            name="speaklyflow-next-segment",
        )
        try:
            done, _ = await asyncio.wait(
                {receive, capture_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if capture_task in done:
                await capture_task
                raise RuntimeError("Audio capture stopped unexpectedly")
            return receive.result()
        finally:
            if not receive.done():
                receive.cancel()
                await asyncio.gather(receive, return_exceptions=True)

    async def _run_turn(self, prompt: str) -> None:
        turn = self._agent.stream(prompt, history=self._history)
        try:
            speech = self._tts.synthesize(self._response_text(turn))
        except BaseException:
            await self._close_turn(turn, suppress_errors=True)
            raise

        try:
            async for chunk in speech:
                await self._audio.write(chunk)

            tts_result = await speech.result()
            if not tts_result.completed:
                raise RuntimeError("TTS synthesis did not complete")

            agent_result = await turn.result()
            self._history.replace(agent_result.messages)
        except BaseException:
            await self._close_turn(turn, speech, suppress_errors=True)
            raise
        else:
            await self._close_turn(turn, speech, suppress_errors=False)

    async def _response_text(self, turn: AgentTurn) -> AsyncIterator[str]:
        segmenter = TextSegmenter()

        async for event in turn:
            if event.kind not in _TEXT_EVENTS:
                continue

            delta = event.payload.get("delta")
            if not isinstance(delta, str):
                continue
            for text in segmenter.push(delta):
                yield text

        remaining = segmenter.flush()
        if remaining:
            yield remaining

    async def _start_components(self) -> None:
        try:
            self._started_closers.append(self._vad.close)
            await self._vad.start(self._audio.input_format)

            self._started_closers.append(self._asr.close)
            await self._asr.start(self._audio.input_format)

            self._started_closers.append(self._agent.close)
            await self._agent.start()

            self._started_closers.append(self._tts.close)
            await self._tts.start()

            self._started_closers.append(self._audio.close)
            await self._audio.start()
        except BaseException:
            await self._close_components(suppress_errors=True)
            raise

    async def _shutdown(self, *, suppress_errors: bool) -> None:
        self._listening.clear()
        capture_task = self._capture_task
        self._capture_task = None
        if capture_task is not None:
            if not capture_task.done():
                capture_task.cancel()
            await asyncio.gather(capture_task, return_exceptions=True)

        await self._close_components(suppress_errors=suppress_errors)

    async def _close_components(self, *, suppress_errors: bool) -> None:
        closers = tuple(reversed(self._started_closers))
        self._started_closers.clear()
        await self._close_all(closers, suppress_errors=suppress_errors)

    @staticmethod
    async def _close_turn(
        turn: AgentTurn,
        speech: TTSStream | None = None,
        *,
        suppress_errors: bool,
    ) -> None:
        closers: list[_Close] = []
        if speech is not None:
            closers.append(speech.aclose)
        closers.append(turn.aclose)
        await VoiceSession._close_all(closers, suppress_errors=suppress_errors)

    @staticmethod
    async def _close_all(
        closers: Iterable[_Close],
        *,
        suppress_errors: bool,
    ) -> None:
        errors: list[Exception] = []
        for close in closers:
            try:
                await close()
            except Exception as error:  # noqa: BLE001 - remaining resources must still close
                errors.append(error)

        if errors and not suppress_errors:
            raise errors[0]
