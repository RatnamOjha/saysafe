"""FastAPI app: approval routes now; the demo page, /ws, /flags, /reset, /text come in D2.

The ntfy Approve / Deny buttons POST to /approvals/{id}/approve and /deny.
"""

import threading

from fastapi import FastAPI, HTTPException

from earshot.approvals.pending import PendingApprovals
from earshot.approvals.pending import pending as default_pending


def create_app(pending: PendingApprovals | None = None) -> FastAPI:
    pending = pending or default_pending
    app = FastAPI(title="earshot")

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

    return app


def serve_in_background(host: str = "0.0.0.0", port: int = 8000) -> threading.Thread:
    """Run the app in a daemon thread (used by `earshot live` for phone taps)."""
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(create_app(), host=host, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True, name="earshot-server")
    thread.start()
    return thread
