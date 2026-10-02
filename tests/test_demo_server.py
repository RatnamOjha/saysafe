import time

import pytest
from fastapi.testclient import TestClient

from saysafe.agent.channels import PhoneChannel
from saysafe.agent.events import EventBus
from saysafe.audio.tts import NullTTS
from saysafe.server.app import create_app
from saysafe.server.demo import DEMO_CODE, DemoSession


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("EARSHOT_DEMO", "1")
    events = EventBus()
    session = DemoSession(events=events, open_mic=False, tts=NullTTS(),
                          phone=PhoneChannel(events, console_only=True))  # fmt: skip
    session.start(warm=False)
    with TestClient(create_app(session=session)) as c:
        c.session = session
        yield c
    session.stop()


def receive_until(ws, wanted: str, limit: int = 60) -> list[dict]:
    seen = []
    for _ in range(limit):
        e = ws.receive_json()
        seen.append(e)
        if e.get("type") == wanted:
            return seen
    raise AssertionError(f"no {wanted} in {[e.get('type') for e in seen]}")


def test_page_and_static_files(client):
    assert "earshot" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_hello_on_connect(client):
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["demo"] is True
        assert len(hello["scenes"]) == 7 and "t_accept" in hello["thresholds"]


def test_typed_command_streams_events(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        assert client.post("/text", json={"text": "what's my verification code"}).status_code == 200
        seen = receive_until(ws, "route")
        types = [e["type"] for e in seen]
        assert "transcript" in types and "detection" in types and "audience_state" in types
        route = seen[-1]
        # the mic is off, so the room is unknown: the code goes to the phone
        assert route["channel"] == "phone_only" and DEMO_CODE in route["phone"]


def test_flags_reset_and_mic(client):
    assert client.post("/flags", json={"headphones": True}).json()["headphones"] is True
    assert client.post("/flags", json={"discreet_mode": True}).json() == {
        "headphones": True, "discreet_mode": True,
    }  # fmt: skip
    assert client.post("/reset", json={"scene": 4}).json() == {"scene": 4}
    assert client.post("/mic", json={"on": True}).json() == {"on": False}  # no mic opened
    assert client.post("/text", json={"text": "  "}).status_code == 400


def test_headphones_flag_reaches_the_router(client):
    client.post("/flags", json={"headphones": True})
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        client.post("/text", json={"text": "what's my verification code"})
        assert receive_until(ws, "route")[-1]["channel"] == "headphones_full"


def test_audience_ticks(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        t0 = time.time()
        e = ws.receive_json()
        assert e["type"] == "audience_tick" and time.time() - t0 < 2.5


def test_approval_card_resolves_through_routes(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        client.post("/text", json={"text": "send twenty dollars to Priya"})
        seen = receive_until(ws, "phone")
        action_id = seen[-1]["action_id"]
        assert client.post(f"/approvals/{action_id}/approve").json()["status"] == "approved"
        assert client.post(f"/approvals/{action_id}/approve").status_code == 409
        assert any(e["type"] == "executed" for e in receive_until(ws, "executed"))
