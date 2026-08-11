"""Silero voice activity detection using ONNX Runtime."""

import asyncio
import math
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnxruntime as ort

from ..audio import AudioChunk, AudioFormat
from .errors import VADError, VADFormatError, VADStateError
from .types import VADState

_SUPPORTED_SAMPLE_RATES = (8_000, 16_000)


class _SileroOnnxModel:
    """Small stateful wrapper around the bundled Silero ONNX model."""

    def __init__(self, model_path: Path) -> None:
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        providers = None
        if "CPUExecutionProvider" in ort.get_available_providers():
            providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=providers,
        )
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 0), dtype=np.float32)
        self._sample_rate = 0

    def predict(self, pcm: bytes, sample_rate: int) -> float:
        """Return the speech probability for one Silero-sized PCM window."""

        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        samples *= 1.0 / 32_768.0
        samples = samples.reshape(1, -1)

        if self._sample_rate and self._sample_rate != sample_rate:
            self._reset()
        context_size = 64 if sample_rate == 16_000 else 32
        if self._context.shape[1] == 0:
            self._context = np.zeros((1, context_size), dtype=np.float32)

        model_input = np.concatenate((self._context, samples), axis=1)
        outputs = self._session.run(
            None,
            {
                "input": model_input,
                "state": self._state,
                "sr": np.array(sample_rate, dtype=np.int64),
            },
        )
        probability = cast(np.ndarray[Any, Any], outputs[0])
        self._state = cast(np.ndarray[Any, Any], outputs[1])
        self._context = model_input[:, -context_size:]
        self._sample_rate = sample_rate
        return float(probability[0, 0])

    def _reset(self) -> None:
        self._state.fill(0)
        self._context = np.zeros((1, 0), dtype=np.float32)
        self._sample_rate = 0


class SileroVAD:
    """Detect stable speech states from mono 8kHz or 16kHz PCM audio."""

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        speech_start_ms: int = 100,
        speech_end_ms: int = 400,
    ) -> None:
        """Initialize configuration without loading the ONNX model."""

        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between zero and one")
        if speech_start_ms <= 0:
            raise ValueError("speech_start_ms must be greater than zero")
        if speech_end_ms <= 0:
            raise ValueError("speech_end_ms must be greater than zero")

        self._threshold = threshold
        self._speech_start_ms = speech_start_ms
        self._speech_end_ms = speech_end_ms

        self._lock = asyncio.Lock()
        self._executor: ThreadPoolExecutor | None = None
        self._model: _SileroOnnxModel | None = None
        self._input_format: AudioFormat | None = None
        self._buffer = bytearray()

        self._state = VADState.SILENCE
        self._speech_windows = 0
        self._silence_windows = 0
        self._start_windows = 0
        self._end_windows = 0

        self._started = False
        self._closed = False

    async def start(self, input_format: AudioFormat) -> None:
        """Load the model and validate the input format."""

        async with self._lock:
            if self._closed:
                raise VADStateError("SileroVAD has already been closed")
            if self._started:
                if input_format != self._input_format:
                    raise VADStateError("SileroVAD is already running with another audio format")
                return
            self._validate_format(input_format)

            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="speaklyflow-vad")
            try:
                future = asyncio.get_running_loop().run_in_executor(executor, self._load_model)
                self._model = await self._await_worker(future)
            except asyncio.CancelledError:
                await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)
                raise
            except Exception as error:
                await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)
                raise VADError(f"Unable to load Silero VAD model: {error}") from error

            self._executor = executor
            self._input_format = input_format
            window_ms = self._window_samples(input_format.sample_rate) * 1_000
            window_ms /= input_format.sample_rate
            self._start_windows = max(1, math.ceil(self._speech_start_ms / window_ms))
            self._end_windows = max(1, math.ceil(self._speech_end_ms / window_ms))
            self._started = True

    async def analyze(self, chunk: AudioChunk) -> VADState:
        """Analyze one PCM chunk without blocking the asyncio event loop."""

        async with self._lock:
            self._ensure_started()
            if chunk.format != self._input_format:
                raise VADFormatError(
                    f"SileroVAD requires {self._input_format!r}, received {chunk.format!r}"
                )
            if not chunk.data:
                return self._state

            input_format = self._require_input_format()
            window_bytes = self._window_samples(input_format.sample_rate)
            window_bytes *= input_format.frame_bytes
            self._buffer.extend(chunk.data)
            complete_bytes = len(self._buffer) // window_bytes * window_bytes
            if complete_bytes == 0:
                return self._state
            pcm = bytes(self._buffer[:complete_bytes])
            del self._buffer[:complete_bytes]

            executor = self._require_executor()
            try:
                future = asyncio.get_running_loop().run_in_executor(
                    executor, self._analyze_pcm, pcm
                )
                return await self._await_worker(future)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise VADError(f"Unable to analyze audio with SileroVAD: {error}") from error

    async def close(self) -> None:
        """Release the model and inference thread."""

        async with self._lock:
            if self._closed:
                return
            self._closed = True
            self._started = False
            self._model = None
            self._buffer.clear()

            executor = self._executor
            self._executor = None
            if executor is not None:
                await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)

    def _analyze_pcm(self, data: bytes) -> VADState:
        model = self._require_model()
        input_format = self._require_input_format()
        window_bytes = self._window_samples(input_format.sample_rate) * input_format.frame_bytes

        for offset in range(0, len(data), window_bytes):
            window = data[offset : offset + window_bytes]
            confidence = model.predict(window, input_format.sample_rate)
            self._update_state(confidence)
        return self._state

    def _update_state(self, confidence: float) -> None:
        if confidence >= self._threshold:
            self._silence_windows = 0
            if self._state is VADState.SILENCE:
                self._speech_windows += 1
                if self._speech_windows >= self._start_windows:
                    self._state = VADState.SPEAKING
                    self._speech_windows = 0
            else:
                self._speech_windows = 0
        else:
            self._speech_windows = 0
            if self._state is VADState.SPEAKING:
                self._silence_windows += 1
                if self._silence_windows >= self._end_windows:
                    self._state = VADState.SILENCE
                    self._silence_windows = 0
            else:
                self._silence_windows = 0

    def _load_model(self) -> _SileroOnnxModel:
        model_resource = files("speaklyflow.vad.data").joinpath("silero_vad.onnx")
        with as_file(model_resource) as model_path:
            return _SileroOnnxModel(model_path)

    async def _await_worker(self, future: asyncio.Future[Any]) -> Any:
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            await asyncio.shield(future)
            raise

    def _validate_format(self, input_format: AudioFormat) -> None:
        if input_format.channels != 1:
            raise VADFormatError("SileroVAD requires mono audio")
        if input_format.sample_rate not in _SUPPORTED_SAMPLE_RATES:
            supported = ", ".join(str(rate) for rate in _SUPPORTED_SAMPLE_RATES)
            raise VADFormatError(f"SileroVAD requires one of these sample rates: {supported}")

    def _ensure_started(self) -> None:
        if self._closed:
            raise VADStateError("SileroVAD has been closed")
        if not self._started:
            raise VADStateError("SileroVAD has not been started")

    def _require_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            raise VADStateError("SileroVAD inference worker is unavailable")
        return self._executor

    def _require_model(self) -> _SileroOnnxModel:
        if self._model is None:
            raise VADStateError("SileroVAD model is unavailable")
        return self._model

    def _require_input_format(self) -> AudioFormat:
        if self._input_format is None:
            raise VADStateError("SileroVAD input format is unavailable")
        return self._input_format

    @staticmethod
    def _window_samples(sample_rate: int) -> int:
        return 512 if sample_rate == 16_000 else 256
