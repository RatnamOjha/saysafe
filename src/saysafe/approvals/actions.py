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

# Types the packaged policy and read-backs know about. Any other snake_case type works
# too ("unlock_door", "refund"); write policy rules for it, or it needs a phone tap.
KNOWN_TYPES = (
    "order_food",
    "send_money",
    "cancel_subscription",
    "book_appointment",
    "send_email",
    "set_reminder",
)
Source = Literal["user_voice", "agent_initiated", "from_content"]


class _FrozenDict(dict):
    """A JSON-serializable dict that refuses mutation after an Action is created."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("Action.params is immutable; create a new Action to change it")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


class Action(BaseModel):
    """Something the agent wants to do. source: user_voice (the owner asked for it),
    agent_initiated (the assistant suggested it) or from_content (an email, a web page
    or a message asked for it); the last two need one more tier of proof."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    type: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)
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

    @field_validator("params", mode="after")
    @classmethod
    def _freeze_params(cls, value: dict[str, Any]) -> dict[str, Any]:
        # Pydantic's frozen model setting blocks `action.amount = ...`, but a nested dict
        # would otherwise still allow `action.params["x"] = ...` after approval.
        return _freeze(value)

    def model_copy(self, *, update=None, deep: bool = False):
        # BaseModel.model_copy intentionally skips validation. Re-freeze params changed
        # through that convenience method too, so a copied action stays immutable.
        copied = super().model_copy(update=update, deep=deep)
        if update and "params" in update:
            object.__setattr__(copied, "params", _freeze(copied.params))
        return copied


def canonical_json(action: Action) -> str:
    """Sorted keys, no whitespace, Decimals and datetimes as strings."""
    return json.dumps(action.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def action_hash(action: Action) -> str:
    return hashlib.sha256(canonical_json(action).encode()).hexdigest()
