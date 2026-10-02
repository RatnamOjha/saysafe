"""FastAPI app: the demo page, /ws, /flags, /reset, /text, /mic, and the approval routes.

The ntfy Approve / Deny buttons POST to /approvals/{id}/approve and /deny, and so do
the buttons on the page's phone mock.
"""

import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from band_demo.agent.events import Event
from saysafe.approvals.pending import PendingApprovals
from saysafe.approvals.pending import pending as default_pending

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"


class Flags(BaseModel):
    headphones: bool | None = None
    discreet_mode: bool | None = None


class TextIn(BaseModel):
    text: str


class ResetIn(BaseModel):
    scene: int | None = None


class MicIn(BaseModel):
    on: bool


def create_app(pending: PendingApprovals | None = None, session=None) -> FastAPI:
    """Approval routes always; the demo page and controls when a DemoSession is given."""
    pending = pending or default_pending
    sockets: set[WebSocket] = set()
    loop_holder: dict = {}

    async def send_all(message: str) -> None:
        for ws in list(sockets):
            try:
                await ws.send_text(message)
            except Exception:
                sockets.discard(ws)

    def forward(event: Event) -> None:
        """Bus events come from worker threads; hop onto the server's event loop."""
        loop = loop_holder.get("loop")
        if loop is not None and sockets:
            message = json.dumps(event.to_dict(), default=str)
            asyncio.run_coroutine_threadsafe(send_all(message), loop)

    async def audience_ticker() -> None:
        while True:
            await asyncio.sleep(1)
            if sockets and session is not None:
                state = asdict(session.tracker.state())
                await send_all(json.dumps({"type": "audience_tick", **state}, default=str))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop_holder["loop"] = asyncio.get_running_loop()
        unsubscribe = session.events.subscribe(forward) if session is not None else None
        ticker = asyncio.create_task(audience_ticker()) if session is not None else None
        yield
        if ticker:
            ticker.cancel()
        if unsubscribe:
            unsubscribe()

    app = FastAPI(title="earshot", lifespan=lifespan)

    def resolve(action_id: str, verb: str) -> dict:
        r = pending.approve(action_id) if verb == "approve" else pending.deny(action_id)
        if r.status is None:
            raise HTTPException(404, "No such approval request")
        if not r.changed:  # already resolved, or expired
            raise HTTPException(409, f"Request is already {r.status}")
        return {"action_id": action_id, "status": r.status}

    @app.post("/approvals/{action_id}/approve")
    def approve(action_id: str) -> dict:
        return resolve(action_id, "approve")

    @app.post("/approvals/{action_id}/deny")
    def deny(action_id: str) -> dict:
        return resolve(action_id, "deny")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    if session is None:
        return app

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        sockets.add(websocket)
        await websocket.send_text(json.dumps(session.snapshot(), default=str))
        try:
            while True:
                await websocket.receive_text()  # the page only listens; keep the socket open
        except WebSocketDisconnect:
            sockets.discard(websocket)

    @app.post("/flags")
    def flags(body: Flags) -> dict:
        return session.set_flags(body.headphones, body.discreet_mode)

    @app.post("/reset")
    def reset(body: ResetIn) -> dict:
        session.reset(body.scene)
        return {"scene": session.scene}

    @app.post("/text")
    def text(body: TextIn) -> dict:
        if not body.text.strip():
            raise HTTPException(400, "Empty text")
        session.submit_text(body.text.strip())
        return {"queued": True}

    @app.post("/mic")
    def mic(body: MicIn) -> dict:
        session.set_mic(body.on)
        return {"on": session.mic_enabled}

    return app


def serve_in_background(host: str = "0.0.0.0", port: int = 8000) -> threading.Thread:
    """Run the approval routes in a daemon thread (used by `earshot live` for phone taps)."""
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(create_app(), host=host, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True, name="earshot-server")
    thread.start()
    return thread
