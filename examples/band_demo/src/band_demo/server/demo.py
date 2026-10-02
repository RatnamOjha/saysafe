"""The demo session: one mic, one pipeline, one audience tracker, one worker thread.

The server reads the laptop mic directly (the page and the server run on the same
laptop, so streaming audio from the browser would only add failure points). Every
trace event is pushed to the page over the websocket.

One worker thread owns the mic. It turns speech into turns, runs typed commands
from the page between turns, and is the only thread that touches the pipeline,
except phone approvals, which complete from the server thread like in `earshot live`.
"""

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from band_demo.agent.channels import PhoneChannel
from band_demo.agent.events import EventBus, bus
from band_demo.agent.mock_agent import MockAgent
from band_demo.agent.pipeline import LiveMic, Pipeline
from band_demo.config import env
from band_demo.owner import owner_score_fn
from saysafe.privacy.audience import AudienceTracker
from saysafe.voice.vad import StreamingVAD

log = logging.getLogger(__name__)

DEMO_CODE = "482913"


@dataclass(frozen=True)
class Scene:
    key: int
    title: str
    lines: str


SCENES = [
    Scene(1, "Owner approves", 'Say "order my usual", then "yeah, do it".'),
    Scene(2, "Someone else says yes", 'Say "order my usual". Your friend says "yes". Tap Approve.'),
    Scene(
        3,
        "Replayed word",
        'Say "send fifty dollars to Jake". Play an old word, then say the new one.',
    ),
    Scene(4, "Alone", 'Mic on for 45 s first. Ask "what\'s my verification code?"'),
    Scene(5, "Friend nearby", 'Your friend chats. Ask "what\'s my verification code?" again.'),
    Scene(6, "Injected email", 'Friend still nearby. Ask "read my last email".'),
    Scene(7, "Whisper", 'Whisper "what\'s my balance?" (only if whisper detection exists).'),
]


class DemoSession:
    def __init__(self, events: EventBus = bus, open_mic: bool = True, tts=None, phone=None):
        self.events = events
        self.demo = env("EARSHOT_DEMO") == "1"
        self.tracker = AudienceTracker(score=owner_score_fn())
        self.agent = MockAgent(fixed_code=DEMO_CODE if self.demo else None)
        self.pipeline = Pipeline(
            agent=self.agent, events=events, audience=self.tracker, tts=tts,
            phone=phone or PhoneChannel(events), listen=self._listen,
        )  # fmt: skip
        self.scene = 0
        self.mic_enabled = False
        self._open_mic = open_mic
        self._stream = None
        self._mic: LiveMic | None = None
        self._jobs: queue.Queue[Callable[[], None]] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # lifecycle

    def start(self, warm: bool = True) -> None:
        if warm:
            t0 = time.perf_counter()
            self.pipeline.warm()
            log.info("models warm in %.1f s", time.perf_counter() - t0)
        if self._open_mic:
            from band_demo.audio.capture import MicStream

            self._stream = MicStream().__enter__()
            self._mic = LiveMic(self._stream, on_segment=self._observe)
            self.set_mic(True)
        self._thread = threading.Thread(target=self._work, daemon=True, name="demo-worker")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            self._stream.__exit__(None, None, None)

    # controls from the page

    def set_mic(self, on: bool) -> None:
        self.mic_enabled = on and self._mic is not None
        if self.mic_enabled:
            self._mic.drain()
            self.tracker.mic_on()
        else:
            self.tracker.mic_off()
        self.events.publish("mic", on=self.mic_enabled)

    def set_flags(self, headphones: bool | None = None, discreet_mode: bool | None = None) -> dict:
        if headphones is not None:
            self.pipeline.flags["headphones"] = headphones
        if discreet_mode is not None:
            self.pipeline.flags["discreet_mode"] = discreet_mode
        self.events.publish("flags", **self.pipeline.flags)
        return dict(self.pipeline.flags)

    def reset(self, scene: int | None = None) -> None:
        """New scene: forget voices and the conversation, keep mic time and flags."""
        self.tracker.reset()
        self.agent._awaiting_amount_for = None
        if scene is not None:
            self.scene = scene
        self.events.led("idle")
        self.events.publish("reset", scene=self.scene)

    def submit_text(self, text: str) -> None:
        self._jobs.put(lambda: self.pipeline.run_text(text))

    def snapshot(self) -> dict:
        from dataclasses import asdict

        from band_demo.config import thresholds as demo_thresholds

        t = demo_thresholds()
        return {
            "type": "hello",
            "demo": self.demo,
            "mic": self.mic_enabled,
            "flags": dict(self.pipeline.flags),
            "scene": self.scene,
            "scenes": [asdict(s) for s in SCENES],
            "thresholds": {"t_accept": t.t_accept, "t_reject": t.t_reject, "source": t.source},
            "audience": asdict(self.tracker.state()),
        }

    # the worker

    def _listen(self, timeout_s: float) -> np.ndarray | None:
        if self._mic is None or not self.mic_enabled:
            return None
        return self._mic.listen(timeout_s)

    def _observe(self, segment: np.ndarray) -> None:
        obs = self.tracker.observe_segment(segment)
        self.events.publish(
            "voice",
            label=None if obs is None else obs.label,
            score=None if obs is None or obs.score is None else round(obs.score, 3),
            seconds=round(len(segment) / 16000, 2),
        )

    def _work(self) -> None:
        vad = StreamingVAD()
        while not self._stop.is_set():
            try:
                job = self._jobs.get_nowait()
            except queue.Empty:
                job = None
            if job is not None:
                self._run(job)
                vad = StreamingVAD()
                continue
            if not self.mic_enabled:
                time.sleep(0.05)
                continue
            try:
                frame = self._stream.read(timeout=0.1)
            except queue.Empty:
                continue
            for segment in vad.feed(frame):
                self._observe(segment)
                self._run(lambda s=segment: self.pipeline.run_audio(s))
                vad = StreamingVAD()
                break

    def _run(self, job: Callable[[], None]) -> None:
        try:
            job()
        except Exception:
            log.exception("turn failed")
            self.events.publish("error", message="That turn failed; see the server log.")
            self.events.led("refused")
        finally:
            if self._mic is not None:
                self._mic.drain()  # don't treat the band's own voice as the next command
