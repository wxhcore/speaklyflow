"""Single-session SpeaklyFlow runtime."""

import asyncio
from collections.abc import Callable
from pathlib import Path

import bumblehive

from speaklyflow import VoiceSession
from speaklyflow.agent import BumblehiveAgent
from speaklyflow.asr import SenseVoiceASR
from speaklyflow.audio import LocalAudio
from speaklyflow.observability import (
    ErrorEvent,
    MetricsEvent,
    VoiceEvent,
    VoiceObserver,
)
from speaklyflow.tts import VolcengineTTS
from speaklyflow.vad import SileroVAD

from .config import AppConfig, load_config, save_config
from .history import ConversationHistory, load_history, save_history
from .protocol import RuntimeView
from .resources import resolve_resource_path

SessionBuilder = Callable[
    [AppConfig, VoiceObserver, bumblehive.MessageHistory],
    VoiceSession,
]


class CommandError(Exception):
    """Expected command rejection returned to the desktop client."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def build_session(
    config: AppConfig,
    observer: VoiceObserver,
    history: bumblehive.MessageHistory,
) -> VoiceSession:
    """Construct the concrete SpeaklyFlow session described by the config."""

    tts_settings = config.tts.settings
    tts = VolcengineTTS(
        api_key=tts_settings.api_key,
        voice=tts_settings.voice,
        resource_id=tts_settings.resource_id,
        sample_rate=tts_settings.sample_rate,
    )
    audio_config = config.audio
    vad_config = config.vad
    asr_settings = config.asr.settings
    return VoiceSession(
        audio=LocalAudio(
            input_device=audio_config.input_device,
            output_device=audio_config.output_device,
            input_sample_rate=16_000,
            output_sample_rate=tts.output_format.sample_rate,
            block_ms=audio_config.block_ms,
            capture_buffer_ms=audio_config.capture_buffer_ms,
            latency=audio_config.latency,
            echo_cancellation=audio_config.echo_cancellation,
        ),
        vad=SileroVAD(
            threshold=vad_config.threshold,
            speech_start_ms=vad_config.speech_start_ms,
            speech_end_ms=vad_config.speech_end_ms,
        ),
        asr=SenseVoiceASR(
            model_dir=resolve_resource_path(asr_settings.model_dir),
            threads=asr_settings.threads,
            language=asr_settings.language,
            use_itn=asr_settings.use_itn,
        ),
        agent=BumblehiveAgent(
            bumblehive.BumblehiveConfig.from_mapping(config.bumblehive)
        ),
        tts=tts,
        history=history,
        observers=[observer],
    )


def _with_default_workspace(config: AppConfig, workspace: Path) -> AppConfig:
    bumblehive_config = dict(config.bumblehive)
    runtime_config = dict(bumblehive_config.get("runtime") or {})
    if not runtime_config.get("workspace"):
        runtime_config["workspace"] = str(workspace)
    bumblehive_config["runtime"] = runtime_config
    return config.model_copy(update={"bumblehive": bumblehive_config})


class RuntimeController:
    """Own one local VoiceSession, configuration, and conversation history."""

    def __init__(
        self,
        config_path: Path,
        *,
        session_builder: SessionBuilder = build_session,
    ) -> None:
        self._config_path = config_path
        self._history_path = config_path.parent / "history.json"
        self._default_workspace = config_path.parent / "workspace"
        self._session_builder = session_builder
        self._lock = asyncio.Lock()
        self._session: VoiceSession | None = None
        self._session_task: asyncio.Task[None] | None = None
        self._config: AppConfig | None = None
        self._config_error: str | None = None

        try:
            self._config = load_config(config_path)
        except (OSError, ValueError, TypeError) as error:
            self._config_error = str(error)

        stored_history = load_history(self._history_path)
        if stored_history is None:
            self._message_history = bumblehive.MessageHistory()
            turns = None
        else:
            self._message_history = bumblehive.MessageHistory(
                stored_history.messages,
                conversation_id=stored_history.conversation_id,
            )
            turns = stored_history.turns

        state = "idle" if self._config is not None else "unconfigured"
        self.view = RuntimeView(state, turns=turns)

    def config_response(self) -> dict[str, object]:
        return {
            "config": (
                self._config.model_dump(mode="json")
                if self._config is not None
                else None
            ),
            "error": self._config_error,
        }

    async def update_config(self, incoming: AppConfig) -> dict[str, object]:
        async with self._lock:
            if self._session_task is not None:
                raise CommandError(
                    "session_active",
                    "Configuration cannot change while the session is active",
                )
            await save_config(self._config_path, incoming)
            self._config = incoming
            self._config_error = None
            self.view.set_runtime_state("idle")
            return self.config_response()

    async def start(self) -> str:
        async with self._lock:
            if self._session_task is not None:
                raise CommandError("session_active", "A session is already active")
            if self._config is None:
                raise CommandError("config_missing", "Server configuration is missing")

            try:
                session_config = _with_default_workspace(
                    self._config,
                    self._default_workspace,
                )
                session = self._session_builder(
                    session_config,
                    self,
                    self._message_history,
                )
            except Exception as error:
                self.view.set_runtime_state(
                    "failed",
                    error={"type": type(error).__name__, "message": str(error)},
                )
                raise

            self._session = session
            self.view.set_runtime_state("starting")
            self._session_task = asyncio.create_task(
                self._run_session(session),
                name="speaklyflow-server-session",
            )
            return session.session_id

    async def stop(self) -> bool:
        async with self._lock:
            session = self._session
            task = self._session_task
            if session is None or task is None:
                return False
            self.view.set_runtime_state("stopping")

        try:
            await session.stop()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        finally:
            try:
                await self._save_history()
            finally:
                async with self._lock:
                    if self._session is session:
                        self._session = None
                        self._session_task = None
                        if self.view.runtime_state != "failed":
                            self.view.set_runtime_state("idle")
        return True

    async def on_event(self, event: VoiceEvent) -> None:
        """Update the desktop projection and persist completed turn state."""

        self.view.on_event(event)
        if isinstance(event, MetricsEvent) or (
            isinstance(event, ErrorEvent) and event.turn_id is not None
        ):
            await self._save_history()

    async def new_conversation(self) -> str:
        """End voice activity and replace the current conversation."""

        await self.stop()
        async with self._lock:
            self._message_history = bumblehive.MessageHistory()
            self.view.new_conversation()
            await self._save_history()
            return self._message_history.conversation_id

    def interrupt(self) -> bool:
        session = self._running_session()
        return session.interrupt()

    def submit_text(self, text: str) -> None:
        session = self._running_session()
        try:
            session.submit_text(text)
        except asyncio.QueueFull as error:
            raise CommandError(
                "input_busy",
                "Another input is already queued",
            ) from error
        except ValueError as error:
            raise CommandError("invalid_text", str(error)) from error
        except RuntimeError as error:
            raise CommandError("session_not_ready", str(error)) from error

    async def _run_session(self, session: VoiceSession) -> None:
        try:
            await session.run()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if self.view.runtime_state != "failed":
                self.view.set_runtime_state(
                    "failed",
                    error={"type": type(error).__name__, "message": str(error)},
                )
        finally:
            async with self._lock:
                if self._session is session:
                    self._session = None
                    self._session_task = None
                    if self.view.runtime_state != "failed":
                        self.view.set_runtime_state("idle")

    def _running_session(self) -> VoiceSession:
        session = self._session
        if session is None or self._session_task is None:
            raise CommandError("session_not_running", "No session is running")
        return session

    async def _save_history(self) -> None:
        await save_history(
            self._history_path,
            ConversationHistory(
                conversation_id=self._message_history.conversation_id,
                messages=self._message_history.get_history(),
                turns=self.view.persisted_turns(),
            ),
        )
