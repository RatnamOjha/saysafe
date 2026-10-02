"""Action model, canonical_json and action_hash.

An approval is bound to action_hash, so changing any field after approval voids it.
Actions are frozen; "changing" one means model_copy(update=...), which gets a new hash.
"""

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

ActionType = Literal[
    "order_food",
    "send_money",
    "cancel_subscription",
    "book_appointment",
    "send_email",
    "set_reminder",
]
Source = Literal["user_voice", "agent_initiated", "from_content"]


class Action(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    type: ActionType
    counterparty: str | None = None
    amount: Decimal | None = None
    currency: Literal["USD"] = "USD"
    destination: str | None = None
    is_new_counterparty: bool = False
    source: Source = "user_voice"
    params: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("amount")
    @classmethod
    def _cents(cls, v: Decimal | None) -> Decimal | None:
        # 43.2 and 43.20 must hash the same.
        return None if v is None else Decimal(v).quantize(Decimal("0.01"))


def canonical_json(action: Action) -> str:
    """Sorted keys, no whitespace, Decimals and datetimes as strings."""
    return json.dumps(action.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def action_hash(action: Action) -> str:
    return hashlib.sha256(canonical_json(action).encode()).hexdigest()
