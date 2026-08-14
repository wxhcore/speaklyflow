"""Run a local voice conversation from microphone input to speech output."""

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass

import bumblehive
import numpy as np

from ..agent import BumblehiveAgent
from ..asr import ASR, SpeechSegmenter
from ..audio import AudioChunk, AudioError, AudioFormatError, AudioIO
from ..observability import (
    Component,
    ComponentEvent,
    ComponentState,
    ErrorEvent,
    InputLevelEvent,
    InputSource,
    MetricsEvent,
    SessionEvent,
    SessionState,
    SpeechEvent,
    SpeechState,
    TranscriptEvent,
    TurnEvent,
    TurnState,
    UserInputEvent,
    VoiceEvent,
    VoiceObserver,
)
from ..observability._dispatcher import _EventDispatcher
from ..tts import TTS
from ..vad import VAD, VADState
from .turn import _TurnFailure, _TurnRunner

_Close = Callable[[], Awaitable[None]]
_INPUT_LEVEL_INTERVAL_SECONDS = 0.1


@dataclass(frozen=True, slots=True)
class _VoiceInput:
    audio: AudioChunk
    speech_stopped_at: float
    estimated_speech_ended_at: float


@dataclass(frozen=True, slots=True)
class _TextInput:
    text: str
    submitted_at: float


_TurnInput = _VoiceInput | _TextInput


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
        observers: Iterable[VoiceObserver] = (),
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
        self._session_id = uuid.uuid4().hex
        self._events = _EventDispatcher(observers)

        self._segmenter = SpeechSegmenter()
        self._inputs: asyncio.Queue[_TurnInput] = asyncio.Queue(maxsize=1)
        self._capture_task: asyncio.Task[None] | None = None
        self._active_turn: _TurnRunner | None = None
        self._run_task: asyncio.Task[None] | None = None
        self._turn_id = 0
        self._started_closers: list[tuple[Component, _Close]] = []
        self._reported_errors: list[BaseException] = []
        self._accepting_inputs = False
        self._stop_requested = False
        self._used = False

    @property
    def history(self) -> bumblehive.MessageHistory:
        """Conversation history committed by completed and interrupted turns."""

        return self._history

    @property
    def session_id(self) -> str:
        """Opaque identifier included in every emitted event."""

        return self._session_id

    async def run(self) -> None:
        """Start components and run turns until cancelled or an error occurs."""

        if self._used:
            raise RuntimeError("VoiceSession can only be run once")
        self._used = True
        self._run_task = asyncio.current_task()
        self._events.start()
        self._emit(
            SessionEvent(session_id=self._session_id, state=SessionState.STARTING)
        )

        try:
            await self._start_components()
            self._capture_task = asyncio.create_task(
                self._capture(),
                name="speaklyflow-audio-capture",
            )
            self._accepting_inputs = True
            self._emit(
                SessionEvent(session_id=self._session_id, state=SessionState.READY)
            )
            await self._run_conversation()
        except asyncio.CancelledError:
            await self._finish(suppress_errors=True)
            if not self._stop_requested:
                raise
        except BaseException as error:
            self._emit_error(
                component=Component.SESSION,
                operation="run",
                error=error,
                fatal=True,
            )
            await self._finish(suppress_errors=True)
            raise
        else:
            await self._finish(suppress_errors=False)
        finally:
            self._accepting_inputs = False
            self._run_task = None

    async def stop(self) -> None:
        """Stop the session and wait for its resources to close."""

        task = self._run_task
        if task is None:
            return
        self._accepting_inputs = False
        self._stop_requested = True
        task.cancel()
        await task

    def interrupt(self) -> bool:
        """Interrupt the active assistant turn, if one is running."""

        turn = self._active_turn
        if turn is None:
            return False
        return turn.interrupt()

    def submit_text(self, text: str) -> None:
        """Queue one text input for the conversation."""

        if not self._accepting_inputs:
            raise RuntimeError("VoiceSession is not ready")
        if not text.strip():
            raise ValueError("Text input must not be empty")
        self._inputs.put_nowait(_TextInput(text, time.perf_counter()))

    async def _run_conversation(self) -> None:
        while True:
            turn_input = await self._next_input()
            if isinstance(turn_input, _TextInput):
                await self._run_turn(
                    turn_input.text,
                    source=InputSource.TEXT,
                    prompt_ready_at=turn_input.submitted_at,
                )
                continue

            asr_started_at = time.perf_counter()
            try:
                transcript = await self._asr.transcribe(turn_input.audio)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - provider failure is input-local
                self._emit_error(
                    component=Component.ASR,
                    operation="transcribe",
                    error=error,
                    fatal=False,
                )
                continue
            asr_finished_at = time.perf_counter()
            if transcript.text:
                self._emit(
                    TranscriptEvent(
                        session_id=self._session_id,
                        text=transcript.text,
                        language=transcript.language,
                    )
                )
                await self._run_turn(
                    transcript.text,
                    source=InputSource.VOICE,
                    prompt_ready_at=asr_finished_at,
                    speech_stopped_at=turn_input.speech_stopped_at,
                    estimated_speech_ended_at=turn_input.estimated_speech_ended_at,
                    asr_started_at=asr_started_at,
                    asr_finished_at=asr_finished_at,
                    asr_audio_seconds=turn_input.audio.duration_seconds,
                )

    async def _capture(self) -> None:
        previous = VADState.SILENCE
        next_level_at = 0.0
        try:
            async for chunk in self._audio.capture():
                now = time.perf_counter()
                if now >= next_level_at:
                    self._emit(
                        InputLevelEvent(
                            session_id=self._session_id,
                            level=_input_level(chunk),
                        )
                    )
                    next_level_at = now + _INPUT_LEVEL_INTERVAL_SECONDS
                try:
                    state = await self._vad.analyze(chunk)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    self._emit_error(
                        component=Component.VAD,
                        operation="analyze",
                        error=error,
                        fatal=True,
                    )
                    raise

                if state is VADState.SPEAKING and previous is VADState.SILENCE:
                    self._emit(
                        SpeechEvent(
                            session_id=self._session_id,
                            state=SpeechState.STARTED,
                        )
                    )
                    self.interrupt()

                segment = self._segmenter.push(chunk, state)
                if segment is not None:
                    speech_stopped_at = time.perf_counter()
                    self._emit(
                        SpeechEvent(
                            session_id=self._session_id,
                            state=SpeechState.STOPPED,
                        )
                    )
                    await self._inputs.put(
                        _VoiceInput(
                            audio=segment,
                            speech_stopped_at=speech_stopped_at,
                            estimated_speech_ended_at=(
                                speech_stopped_at
                                - self._vad.speech_end_confirmation_seconds
                            ),
                        )
                    )
                previous = state
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self._emit_error(
                component=Component.AUDIO,
                operation="capture",
                error=error,
                fatal=True,
            )
            raise
        finally:
            self._emit(InputLevelEvent(session_id=self._session_id, level=0.0))

    async def _next_input(self) -> _TurnInput:
        capture_task = self._capture_task
        if capture_task is None:
            raise RuntimeError("VoiceSession audio capture is not running")

        receive = asyncio.create_task(
            self._inputs.get(),
            name="speaklyflow-next-input",
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

    async def _run_turn(
        self,
        prompt: str,
        *,
        source: InputSource,
        prompt_ready_at: float,
        speech_stopped_at: float | None = None,
        estimated_speech_ended_at: float | None = None,
        asr_started_at: float | None = None,
        asr_finished_at: float | None = None,
        asr_audio_seconds: float | None = None,
    ) -> None:
        self._turn_id += 1
        turn_id = self._turn_id
        self._emit(
            UserInputEvent(
                session_id=self._session_id,
                turn_id=turn_id,
                source=source,
                text=prompt,
            )
        )
        self._emit(
            TurnEvent(
                session_id=self._session_id,
                turn_id=turn_id,
                state=TurnState.STARTED,
            )
        )
        runner = _TurnRunner(
            prompt=prompt,
            history=self._history,
            audio=self._audio,
            agent=self._agent,
            tts=self._tts,
            session_id=self._session_id,
            turn_id=turn_id,
            emit=self._emit,
            prompt_ready_at=prompt_ready_at,
            speech_stopped_at=speech_stopped_at,
            estimated_speech_ended_at=estimated_speech_ended_at,
            asr_started_at=asr_started_at,
            asr_finished_at=asr_finished_at,
            asr_audio_seconds=asr_audio_seconds,
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

            try:
                outcome = await task
            except _TurnFailure as failure:
                self._history.extend(runner.failure_messages())
                self._emit(
                    TurnEvent(
                        session_id=self._session_id,
                        turn_id=turn_id,
                        state=TurnState.FAILED,
                    )
                )
                self._emit(
                    MetricsEvent(
                        session_id=self._session_id,
                        turn_id=turn_id,
                        metrics=runner.metrics(),
                    )
                )
                self._emit_error(
                    component=failure.component,
                    operation=failure.operation,
                    error=failure.error,
                    fatal=failure.fatal,
                    turn_id=turn_id,
                )
                if failure.fatal:
                    raise failure.error
                return
            except AudioError as error:
                self._emit(
                    TurnEvent(
                        session_id=self._session_id,
                        turn_id=turn_id,
                        state=TurnState.FAILED,
                    )
                )
                self._emit_error(
                    component=Component.AUDIO,
                    operation="playback",
                    error=error,
                    fatal=True,
                    turn_id=turn_id,
                )
                raise
            except Exception as error:
                self._emit(
                    TurnEvent(
                        session_id=self._session_id,
                        turn_id=turn_id,
                        state=TurnState.FAILED,
                    )
                )
                self._emit_error(
                    component=Component.SESSION,
                    operation="turn",
                    error=error,
                    fatal=True,
                    turn_id=turn_id,
                )
                raise

            if outcome.interrupted:
                self._history.extend(outcome.messages)
                state = TurnState.INTERRUPTED
            else:
                self._history.replace(outcome.messages)
                state = (
                    TurnState.FAILED
                    if outcome.failure is not None
                    else TurnState.COMPLETED
                )

            self._emit(
                TurnEvent(
                    session_id=self._session_id,
                    turn_id=turn_id,
                    state=state,
                )
            )
            self._emit(
                MetricsEvent(
                    session_id=self._session_id,
                    turn_id=turn_id,
                    metrics=runner.metrics(),
                )
            )
            if outcome.failure is not None:
                self._emit_error(
                    component=outcome.failure.component,
                    operation=outcome.failure.operation,
                    error=outcome.failure.error,
                    fatal=outcome.failure.fatal,
                    turn_id=turn_id,
                )
                if outcome.failure.fatal:
                    raise outcome.failure.error
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if self._active_turn is runner:
                self._active_turn = None

    async def _start_components(self) -> None:
        try:
            await self._start_component(
                Component.VAD,
                lambda: self._vad.start(self._audio.input_format),
                self._vad.close,
            )
            await self._start_component(
                Component.ASR,
                lambda: self._asr.start(self._audio.input_format),
                self._asr.close,
            )
            await self._start_component(
                Component.AGENT,
                self._agent.start,
                self._agent.close,
            )
            await self._start_component(
                Component.TTS,
                self._tts.start,
                self._tts.close,
            )
            await self._start_component(
                Component.AUDIO,
                self._audio.start,
                self._audio.close,
            )
        except BaseException:
            await self._close_components(suppress_errors=True)
            raise

    async def _start_component(
        self,
        component: Component,
        start: Callable[[], Awaitable[None]],
        close: _Close,
    ) -> None:
        self._emit(
            ComponentEvent(
                session_id=self._session_id,
                component=component,
                state=ComponentState.STARTING,
            )
        )
        started_at = time.perf_counter()
        self._started_closers.append((component, close))
        try:
            await start()
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            elapsed = _duration_ms(started_at, time.perf_counter())
            self._emit(
                ComponentEvent(
                    session_id=self._session_id,
                    component=component,
                    state=ComponentState.FAILED,
                    elapsed_ms=elapsed,
                )
            )
            self._emit_error(
                component=component,
                operation="start",
                error=error,
                fatal=True,
            )
            raise
        self._emit(
            ComponentEvent(
                session_id=self._session_id,
                component=component,
                state=ComponentState.READY,
                elapsed_ms=_duration_ms(started_at, time.perf_counter()),
            )
        )

    async def _finish(self, *, suppress_errors: bool) -> None:
        self._emit(
            SessionEvent(session_id=self._session_id, state=SessionState.STOPPING)
        )
        try:
            await self._shutdown(suppress_errors=suppress_errors)
        except BaseException as error:
            self._emit_error(
                component=Component.SESSION,
                operation="shutdown",
                error=error,
                fatal=True,
            )
            self._emit(
                SessionEvent(session_id=self._session_id, state=SessionState.STOPPED)
            )
            await self._events.close()
            raise
        self._emit(
            SessionEvent(session_id=self._session_id, state=SessionState.STOPPED)
        )
        await self._events.close()

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

    async def _close_all(
        self,
        closers: Iterable[tuple[Component, _Close]],
        *,
        suppress_errors: bool,
    ) -> None:
        errors: list[Exception] = []
        for component, close in closers:
            try:
                await close()
            except Exception as error:  # noqa: BLE001 - remaining resources must still close
                errors.append(error)
                self._emit_error(
                    component=component,
                    operation="close",
                    error=error,
                    fatal=not suppress_errors,
                )

        if errors and not suppress_errors:
            raise errors[0]

    def _emit(self, event: VoiceEvent) -> None:
        self._events.emit(event)

    def _emit_error(
        self,
        *,
        component: Component,
        operation: str,
        error: BaseException,
        fatal: bool,
        turn_id: int | None = None,
    ) -> None:
        if any(reported is error for reported in self._reported_errors):
            return
        self._reported_errors.append(error)
        self._emit(
            ErrorEvent(
                session_id=self._session_id,
                component=component,
                operation=operation,
                message=str(error),
                error_type=type(error).__name__,
                fatal=fatal,
                turn_id=turn_id,
            )
        )


def _duration_ms(start: float, end: float) -> float:
    return round(max(end - start, 0) * 1_000, 1)


def _input_level(chunk: AudioChunk) -> float:
    samples = np.frombuffer(chunk.data, dtype="<i2")
    if samples.size == 0:
        return 0.0
    rms = np.sqrt(np.mean(np.square(samples, dtype=np.float64)))
    return min(float(rms / 32_768), 1.0)
