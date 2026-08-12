"""Run a local voice conversation from microphone input to speech output."""

import asyncio
from collections.abc import Awaitable, Callable, Iterable

import bumblehive

from ._turn import _TurnRunner
from .agent import BumblehiveAgent
from .asr import ASR, SpeechSegmenter
from .audio import AudioChunk, AudioFormatError, AudioIO
from .tts import TTS
from .vad import VAD, VADState

_Close = Callable[[], Awaitable[None]]


class VoiceSession:
    """Run continuous interruptible voice turns with managed resources."""

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
        self._capture_task: asyncio.Task[None] | None = None
        self._active_turn: _TurnRunner | None = None
        self._turn_id = 0
        self._started_closers: list[_Close] = []
        self._used = False

    @property
    def history(self) -> bumblehive.MessageHistory:
        """Conversation history committed by completed and interrupted turns."""

        return self._history

    async def run(self) -> None:
        """Start components and run turns until cancelled or an error occurs."""

        if self._used:
            raise RuntimeError("VoiceSession can only be run once")
        self._used = True

        try:
            await self._start_components()
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

    async def _capture(self) -> None:
        previous = VADState.SILENCE
        async for chunk in self._audio.capture():
            state = await self._vad.analyze(chunk)
            if state is VADState.SPEAKING and previous is VADState.SILENCE:
                active_turn = self._active_turn
                if active_turn is not None:
                    active_turn.interrupt()

            segment = self._segmenter.push(chunk, state)
            if segment is not None:
                await self._segments.put(segment)
            previous = state

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
        self._turn_id += 1
        turn_id = self._turn_id
        runner = _TurnRunner(
            prompt=prompt,
            history=self._history,
            audio=self._audio,
            agent=self._agent,
            tts=self._tts,
        )
        self._active_turn = runner
        task = asyncio.create_task(
            runner.run(),
            name=f"speaklyflow-turn-{turn_id}",
        )
        capture_task = self._capture_task
        if capture_task is None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise RuntimeError("VoiceSession audio capture is not running")

        try:
            done, _ = await asyncio.wait(
                {task, capture_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if capture_task in done:
                await capture_task
                raise RuntimeError("Audio capture stopped unexpectedly")

            outcome = await task
            if self._turn_id != turn_id:
                return
            if outcome.interrupted:
                self._history.extend(outcome.messages)
            else:
                self._history.replace(outcome.messages)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if self._active_turn is runner:
                self._active_turn = None

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
