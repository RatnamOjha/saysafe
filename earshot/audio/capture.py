"""Mic recording via sounddevice."""

import logging

import numpy as np

from earshot.audio.io import SR

log = logging.getLogger(__name__)


def input_device_name(device: int | str | None = None) -> str:
    import sounddevice as sd

    return sd.query_devices(device, kind="input")["name"]


def record(seconds: float, device: int | str | None = None) -> np.ndarray:
    """Block for `seconds` and return 16 kHz mono float32 from the mic."""
    import sounddevice as sd

    log.info("recording %.1f s from %s", seconds, input_device_name(device))
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32", device=device)
    sd.wait()
    return audio[:, 0].copy()


class MicStream:
    """Context manager yielding 32 ms frames (512 samples) from the mic, for StreamingVAD."""

    def __init__(self, frame_samples: int = 512, device: int | str | None = None):
        import queue

        self.frame_samples = frame_samples
        self.device = device
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = None

    def __enter__(self) -> "MicStream":
        import sounddevice as sd

        log.info("mic stream from %s", input_device_name(self.device))
        self._stream = sd.InputStream(
            samplerate=SR,
            channels=1,
            dtype="float32",
            blocksize=self.frame_samples,
            device=self.device,
            callback=lambda data, *_: self._queue.put(data[:, 0].copy()),
        )
        self._stream.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stream.stop()
        self._stream.close()

    def read(self, timeout: float | None = None) -> np.ndarray:
        """Next frame. Raises queue.Empty on timeout."""
        return self._queue.get(timeout=timeout)
