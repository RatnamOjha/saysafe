"""Output channels: speaker, headphones, phone (ntfy).

Every delivery is recorded in `sent` and published as a trace event, so tests and
the demo UI see exactly what went where.
"""

import logging
from dataclasses import dataclass, field

import httpx

from earshot.agent.events import EventBus, bus
from earshot.audio.tts import TTS
from earshot.config import env

log = logging.getLogger(__name__)


@dataclass
class Delivery:
    channel: str
    text: str
    volume: float = 1.0
    extra: dict = field(default_factory=dict)


class SpeakerChannel:
    name = "speaker"

    def __init__(self, tts: TTS, events: EventBus = bus):
        self.tts = tts
        self.events = events
        self.sent: list[Delivery] = []

    def deliver(self, text: str, volume: float = 1.0) -> Delivery:
        with self.events.step("tts", channel=self.name, text=text, volume=volume):
            audio = self.tts.synthesize(text)
        d = Delivery(self.name, text, volume)
        self.sent.append(d)
        self.events.publish("spoken", channel=self.name, text=text, volume=volume)
        self.tts.play(audio, volume)
        return d


class HeadphonesChannel(SpeakerChannel):
    """Same audio as the speaker; on the band this would go to paired headphones."""

    name = "headphones"


class PhoneChannel:
    """Notifications via ntfy. Falls back to the console when ntfy isn't configured
    or unreachable, so nothing depends on the network."""

    name = "phone"

    def __init__(
        self,
        events: EventBus = bus,
        console_only: bool = False,
        client: httpx.Client | None = None,
    ):
        self.events = events
        self.server = env("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
        self.topic = None if console_only else env("NTFY_TOPIC")
        self.public_url = env("EARSHOT_PUBLIC_URL", "http://localhost:8000").rstrip("/")
        self.client = client or httpx.Client(timeout=3.0)
        self.sent: list[Delivery] = []

    def notify(self, text: str, title: str = "earshot") -> Delivery:
        return self._send({"title": title, "message": text, "tags": ["lock"]}, text)

    def request_approval(self, action_id: str, summary: str) -> Delivery:
        base = f"{self.public_url}/approvals/{action_id}"
        payload = {
            "title": "Approve?",
            "message": summary,
            "tags": ["warning"],
            "priority": 4,
            "actions": [
                {"action": "http", "label": "Approve", "url": f"{base}/approve",
                 "method": "POST", "clear": True},
                {"action": "http", "label": "Deny", "url": f"{base}/deny",
                 "method": "POST", "clear": True},
            ],
        }  # fmt: skip
        return self._send(payload, summary, action_id=action_id)

    def _send(self, payload: dict, text: str, **extra) -> Delivery:
        via = "console"
        if self.topic:
            try:
                r = self.client.post(self.server, json={"topic": self.topic, **payload})
                r.raise_for_status()
                via = "ntfy"
            except httpx.HTTPError as e:
                log.warning("phone: ntfy failed (%s), printing instead", type(e).__name__)
        if via == "console":
            log.info("[phone] %s: %s", payload.get("title"), text)
        d = Delivery(self.name, text, extra={"via": via, **extra})
        self.sent.append(d)
        self.events.publish("phone", text=text, via=via, title=payload.get("title"), **extra)
        return d
