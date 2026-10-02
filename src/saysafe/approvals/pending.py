"""Phone approval requests waiting for a tap. Expire after 120 s; resolve once.

The approvals hook adds a request when it steps up; the server's /approvals routes
resolve it. On approve, a phone_tap token is issued for the exact action and the
owner's `on_approved` callback runs it.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from saysafe.approvals.actions import Action
from saysafe.approvals.tokens import TokenService, default_service
from saysafe.config import load_yaml

Status = Literal["pending", "approved", "denied", "expired"]


@dataclass(frozen=True)
class Resolution:
    status: Status | None  # None: no such request
    changed: bool  # False if it was already resolved or expired


@dataclass
class PendingApproval:
    action: Action
    tier: str
    created_at: float
    expires_at: float
    on_approved: Callable[[Action, str], None]
    on_denied: Callable[[Action], None]
    status: Status = "pending"
    extra: dict = field(default_factory=dict)


class PendingApprovals:
    def __init__(
        self,
        tokens: TokenService | None = None,
        clock: Callable[[], float] = time.time,
        ttl_s: float | None = None,
    ):
        self._tokens = tokens
        self.clock = clock
        self.ttl_s = ttl_s if ttl_s is not None else load_yaml("policy")["phone"]["pending_ttl_s"]
        self._items: dict[str, PendingApproval] = {}
        self._lock = threading.Lock()

    @property
    def tokens(self) -> TokenService:
        return self._tokens or default_service()

    def add(
        self,
        action: Action,
        tier: str,
        on_approved: Callable[[Action, str], None],
        on_denied: Callable[[Action], None] = lambda a: None,
    ) -> PendingApproval:
        now = self.clock()
        p = PendingApproval(action, tier, now, now + self.ttl_s, on_approved, on_denied)
        with self._lock:
            self._items[action.id] = p
        return p

    def get(self, action_id: str) -> PendingApproval | None:
        p = self._items.get(action_id)
        if p is not None and p.status == "pending" and self.clock() >= p.expires_at:
            p.status = "expired"
        return p

    def approve(self, action_id: str) -> Resolution:
        """Only a pending, unexpired request can be approved, and only once."""
        with self._lock:
            p = self.get(action_id)
            if p is None or p.status != "pending":
                return Resolution(p.status if p else None, changed=False)
            p.status = "approved"
        token = self.tokens.issue(p.action, p.tier, method="phone_tap", fused_score=None)
        p.on_approved(p.action, token)
        return Resolution("approved", changed=True)

    def deny(self, action_id: str) -> Resolution:
        with self._lock:
            p = self.get(action_id)
            if p is None or p.status != "pending":
                return Resolution(p.status if p else None, changed=False)
            p.status = "denied"
        p.on_denied(p.action)
        return Resolution("denied", changed=True)


pending = PendingApprovals()
