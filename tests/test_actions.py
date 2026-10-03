from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from saysafe.approvals.actions import Action, action_hash, canonical_json

FIXED = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)


def _action(**kw) -> Action:
    base = dict(
        id="a1", type="send_money", counterparty="Jake", amount=Decimal("50"),
        destination="@jake", params={"note": "dinner", "split": 2}, created_at=FIXED,
    )  # fmt: skip
    return Action(**{**base, **kw})


def test_hash_stable_across_key_order():
    a = _action(params={"note": "dinner", "split": 2})
    b = _action(params={"split": 2, "note": "dinner"})
    assert canonical_json(a) == canonical_json(b)
    assert action_hash(a) == action_hash(b)


def test_amount_normalized_to_cents():
    assert action_hash(_action(amount=Decimal("43.2"))) == action_hash(
        _action(amount=Decimal("43.20"))
    )
    assert '"amount":"50.00"' in canonical_json(_action())


@pytest.mark.parametrize(
    "field, value",
    [
        ("id", "a2"), ("type", "order_food"), ("counterparty", "Priya"),
        ("amount", Decimal("50.01")), ("destination", "@evil"), ("is_new_counterparty", True),
        ("source", "from_content"), ("params", {"note": "dinner", "split": 3}),
        ("created_at", datetime(2026, 9, 26, 12, 0, 1, tzinfo=timezone.utc)),
    ],
)  # fmt: skip
def test_hash_changes_when_any_field_changes(field, value):
    a = _action()
    assert action_hash(a) != action_hash(a.model_copy(update={field: value}))


def test_every_field_is_covered_by_the_test_above():
    covered = {"id", "type", "counterparty", "amount", "destination", "is_new_counterparty",
               "source", "params", "created_at", "currency"}  # fmt: skip
    assert set(Action.model_fields) == covered  # currency is Literal["USD"], can't change


def test_actions_are_frozen():
    with pytest.raises(ValidationError):
        _action().amount = Decimal("1")
