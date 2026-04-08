"""Voice Listener — Always-on Whisper-based wake word detection and speech transcription."""
from __future__ import annotations

import logging
import threading
import time

import numpy as np
import sounddevice as sd
import whisper

logger = logging.getLogger(__name__)

# Virtual/software mics to skip
_MIC_BLACKLIST = [
    "camo",
    "iriun",
    "virtual",
    "intelligo vac",
    "stereo mix",
    "sound mapper",
    "primary sound",
]


def _find_best_input_device() -> tuple[int | None, float]:
    """Scan audio devices and pick the best physical mic.

    Priority: Bluetooth buds (realme) > ASUS noise-cancelling > Realtek > default.
    Prefers the Windows Audio Session API (WASAPI) variant when available.

    Returns:
        (device_index, native_sample_rate).
    """
    try:
        devices = sd.query_devices()
    except Exception as exc:
        logger.warning("Could not enumerate audio devices: %s", exc)
        return None, 16000.0

    # Preference keywords in priority order
    prefer = [
        "realme",                    # Bluetooth earbuds (user's Realme Buds)
        "ai noise-cancelling input",  # ASUS built-in
        "microphone array (realtek",  # Laptop array mic
        "microphone (realtek",        # Laptop single mic
    ]

    candidates: list[tuple[int, str, float]] = []
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) < 1:
            continue
        name_lower = dev["name"].lower()
        if any(bl in name_lower for bl in _MIC_BLACKLIST):
            continue
        # Skip WDM-KS raw duplicates — they often produce garbled audio
        hostapi = sd.query_hostapis(dev["hostapi"])["name"].lower() if "hostapi" in dev else ""
        if "wdm-ks" in name_lower or "wdm-ks" in hostapi:
            continue
        candidates.append((idx, name_lower, dev["default_samplerate"]))

    if not candidates:
        return None, 16000.0

    def _score(item: tuple[int, str, float]) -> int:
        _, name, _ = item
        for rank, kw in enumerate(prefer):
            if kw in name:
                return rank
        return len(prefer)

    candidates.sort(key=_score)
    best_idx, best_name, best_rate = candidates[0]
    logger.info("Auto-selected microphone: [%d] %s (native rate=%.0f)",
                best_idx, devices[best_idx]["name"], best_rate)
    return best_idx, best_rate


class VoiceListener:
    """Listens for the wake word 'jarvis', then transcribes the following speech.

    Runs Whisper base.en on CPU in a background thread.
    Records at the device's native sample rate, then resamples to 16 kHz for Whisper.
    """

    WHISPER_RATE = 16000  # Whisper always expects 16 kHz

    def __init__(self, config: dict, on_transcription_callback) -> None:
        self.wake_word: str = config["voice"]["wake_word"].lower()
        self.chunk_duration: float = config["voice"]["chunk_duration"]
        self.silence_threshold: int = config["voice"]["silence_threshold"]
        self.whisper_model_name: str = config["voice"]["whisper_model"]
        self.callback = on_transcription_callback
        self.is_listening: bool = False
        self._thread: threading.Thread | None = None
        self._model = None
        self._device: int | None = None
        self._device_rate: float = 16000.0  # actual recording rate (device native)

    def start(self) -> None:
        """Load the Whisper model and start the background listening thread."""
        logger.info("Loading Whisper model '%s' on cpu...", self.whisper_model_name)
        self._model = whisper.load_model(self.whisper_model_name, device="cpu")
        logger.info("Whisper model loaded.")

        self._device, self._device_rate = _find_best_input_device()

        # Log the chosen device
        try:
            dev_info = sd.query_devices(self._device) if self._device is not None else sd.query_devices(kind="input")
            logger.info(
                "Microphone: %s (native rate=%.0f, will resample to %d for Whisper)",
                dev_info["name"], self._device_rate, self.WHISPER_RATE,
            )
        except Exception as exc:
            logger.warning("Could not query microphone info: %s", exc)

        self.is_listening = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logger.info("VoiceListener started — listening for wake word '%s'.", self.wake_word)

    def _rec(self, duration_seconds: float) -> np.ndarray:
        """Record audio at the device’s native rate and resample to 16 kHz.

        Args:
            duration_seconds: How many seconds to record.

        Returns:
            1-D float32 numpy array at 16 kHz.
        """
        num_samples = int(self._device_rate * duration_seconds)
        audio = sd.rec(
            num_samples,
            samplerate=self._device_rate,
            channels=1,
            dtype="float32",
            device=self._device,
        )
        sd.wait()
        audio = audio.flatten()

        # Resample to 16 kHz if needed
        if abs(self._device_rate - self.WHISPER_RATE) > 1.0:
            audio = self._resample(audio, self._device_rate, self.WHISPER_RATE)

        return audio

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: float, target_sr: float) -> np.ndarray:
        """Resample audio from orig_sr to target_sr using linear interpolation."""
        if len(audio) == 0:
            return audio
        duration = len(audio) / orig_sr
        target_len = int(duration * target_sr)
        if target_len <= 0:
            return np.zeros(1, dtype=np.float32)
        x_old = np.arange(len(audio), dtype=np.float64)
        x_new = np.linspace(0, len(audio) - 1, target_len, dtype=np.float64)
        resampled = np.interp(x_new, x_old, audio.astype(np.float64))
        return resampled.astype(np.float32)

    def _listen_loop(self) -> None:
        """Continuously record audio in 2-second chunks, check for wake word."""
        wake_chunk_seconds = 2.0

        while self.is_listening:
            try:
                audio_chunk = self._rec(wake_chunk_seconds)

                rms = float(np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2)) * 32768)
                if rms < 0.5:
                    time.sleep(0.1)
                    continue

                if not self.is_listening:
                    break

                text = self._transcribe(audio_chunk)
                if self.wake_word in text.lower():
                    logger.info("Wake word detected! (heard: %r)", text)
                    full_audio = self._record_until_silence()
                    if full_audio is not None and len(full_audio) > 0:
                        transcription = self._transcribe(full_audio)
                        cleaned = transcription.strip()
                        lower = cleaned.lower()
                        # Strip wake word from beginning if present
                        if lower.startswith(self.wake_word):
                            cleaned = cleaned[len(self.wake_word):].strip(" ,.")
                        if cleaned:
                            logger.info("Transcription: %s", cleaned)
                            self.callback(cleaned)

            except Exception as exc:
                logger.error("VoiceListener error: %s", exc)
                time.sleep(1)

    def _record_until_silence(self) -> np.ndarray | None:
        """Record audio until 1.5 seconds of silence is detected.

        Returns:
            1-D float32 numpy array at 16 kHz, or None on failure.
        """
        silence_limit = 1.5
        silence_chunks = int(silence_limit / self.chunk_duration)
        max_duration = 30.0
        max_chunks = int(max_duration / self.chunk_duration)

        audio_chunks: list[np.ndarray] = []
        consecutive_silence = 0

        for _ in range(max_chunks):
            if not self.is_listening:
                break
            chunk = self._rec(self.chunk_duration)
            audio_chunks.append(chunk)

            rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)) * 32768)
            if rms < self.silence_threshold:
                consecutive_silence += 1
            else:
                consecutive_silence = 0

            if consecutive_silence >= silence_chunks:
                break

        return np.concatenate(audio_chunks) if audio_chunks else None

    def _transcribe(self, audio_array: np.ndarray) -> str:
        """Transcribe a 16 kHz float32 audio array using Whisper.

        Args:
            audio_array: 1-D float32 numpy array at 16 kHz.

        Returns:
            Transcribed text string.
        """
        if self._model is None:
            return ""
        try:
            result = self._model.transcribe(
                audio_array,
                fp16=False,
                language="en",
            )
            return result.get("text", "").strip()
        except Exception as exc:
            logger.error("Whisper transcription error: %s", exc)
            return ""

    def stop(self) -> None:
        """Stop listening and join the background thread."""
        self.is_listening = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("VoiceListener stopped.")
