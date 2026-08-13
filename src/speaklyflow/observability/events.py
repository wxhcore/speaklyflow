"""Typed events emitted by a voice session."""

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .metrics import TurnMetrics


class Component(str, Enum):
    """Voice session component that produced an event."""

    SESSION = "session"
    AUDIO = "audio"
    VAD = "vad"
    ASR = "asr"
    AGENT = "agent"
    TTS = "tts"


class SessionState(str, Enum):
    """Lifecycle state of a voice session."""

    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"


class ComponentState(str, Enum):
    """Initialization state of a session component."""

    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"


class SpeechState(str, Enum):
    """Stable user speech transition reported by VAD."""

    STARTED = "started"
    STOPPED = "stopped"


class TurnState(str, Enum):
    """State of one recognized user turn."""

    STARTED = "started"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class InputSource(str, Enum):
    """Source that supplied one user turn."""

    VOICE = "voice"
    TEXT = "text"


class SynthesisState(str, Enum):
    """State of assistant speech synthesis."""

    STARTED = "started"
    FIRST_AUDIO = "first_audio"
    FINISHED = "finished"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class PlaybackState(str, Enum):
    """State of assistant audio playback."""

    STARTED = "started"
    PROGRESS = "progress"
    FINISHED = "finished"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


@dataclass(frozen=True, slots=True, kw_only=True)
class VoiceEvent:
    """Base event with session correlation and wall-clock time."""

    session_id: str
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionEvent(VoiceEvent):
    """Voice session lifecycle transition."""

    state: SessionState


@dataclass(frozen=True, slots=True, kw_only=True)
class ComponentEvent(VoiceEvent):
    """Component initialization transition and elapsed time."""

    component: Component
    state: ComponentState
    elapsed_ms: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class SpeechEvent(VoiceEvent):
    """User speech start or stop transition."""

    state: SpeechState


@dataclass(frozen=True, slots=True, kw_only=True)
class TranscriptEvent(VoiceEvent):
    """Final text recognized from one speech segment."""

    text: str
    language: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class UserInputEvent(VoiceEvent):
    """User text accepted for one voice or text turn."""

    turn_id: int
    source: InputSource
    text: str


@dataclass(frozen=True, slots=True, kw_only=True)
class TurnEvent(VoiceEvent):
    """Assistant turn lifecycle transition."""

    turn_id: int
    state: TurnState


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentRequestEvent(VoiceEvent):
    """Conversation messages about to be sent to the agent."""

    turn_id: int
    messages: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentTextEvent(VoiceEvent):
    """One streamed assistant text delta."""

    turn_id: int
    delta: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCallStartedEvent(VoiceEvent):
    """Bumblehive tool call ready for execution."""

    turn_id: int
    call_id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCallFinishedEvent(VoiceEvent):
    """Completed Bumblehive tool call and its execution result."""

    turn_id: int
    call_id: str
    name: str
    result: str
    succeeded: bool
    elapsed_ms: float


@dataclass(frozen=True, slots=True, kw_only=True)
class SynthesisEvent(VoiceEvent):
    """Assistant speech synthesis transition and elapsed time."""

    turn_id: int
    state: SynthesisState
    elapsed_ms: float


@dataclass(frozen=True, slots=True, kw_only=True)
class PlaybackEvent(VoiceEvent):
    """Assistant playback state and text confirmed as played."""

    turn_id: int
    state: PlaybackState
    spoken_text: str = ""
    delta: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class MetricsEvent(VoiceEvent):
    """Final metrics snapshot for one turn."""

    turn_id: int
    metrics: TurnMetrics


@dataclass(frozen=True, slots=True, kw_only=True)
class ErrorEvent(VoiceEvent):
    """Serializable error information for application presentation."""

    component: Component
    operation: str
    message: str
    error_type: str
    fatal: bool
    turn_id: int | None = None
