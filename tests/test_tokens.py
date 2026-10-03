import threading
from decimal import Decimal

import pytest

from saysafe import Action, Refused
from saysafe.approvals.tokens import MemoryNonceStore, TokenService


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def order():
    return Action(type="order_food", counterparty="DoorDash", amount=Decimal("43.20"),
                  destination="home")  # fmt: skip


@pytest.fixture(params=["sqlite", "memory"])
def tokens(tmp_path, request):
    nonces = tmp_path / "n.sqlite" if request.param == "sqlite" else MemoryNonceStore()
    return TokenService(b"s" * 32, nonces, clock=Clock(), ttl_s=60)


def test_token_approved_action_runs(tokens):
    a = order()
    assert tokens.verify(tokens.issue(a, "voice", "voice", 0.8), a)["method"] == "voice"


def test_changed_amount_after_approval_is_refused(tokens):
    a = order()
    token = tokens.issue(a, "voice", "voice", 0.8)
    tampered = a.model_copy(update={"amount": Decimal("430.20")})
    with pytest.raises(Refused, match="action_changed"):
        tokens.verify(token, tampered)


@pytest.mark.parametrize("field, value", [("counterparty", "Evil"), ("destination", "work"),
                                          ("id", "other")])  # fmt: skip
def test_any_changed_field_is_refused(tokens, field, value):
    a = order()
    token = tokens.issue(a, "voice", "voice", 0.8)
    with pytest.raises(Refused, match="action_changed"):
        tokens.verify(token, a.model_copy(update={field: value}))


def test_replayed_token_is_refused(tokens):
    a = order()
    token = tokens.issue(a, "voice", "voice", 0.8)
    tokens.verify(token, a)
    with pytest.raises(Refused, match="replayed"):
        tokens.verify(token, a)


def test_expired_token_is_refused(tokens):
    a = order()
    token = tokens.issue(a, "voice", "voice", 0.8)
    tokens.clock.now += 60
    with pytest.raises(Refused, match="expired"):
        tokens.verify(token, a)


@pytest.mark.parametrize("mangle", [
    lambda t: t[:-2] + ("AA" if not t.endswith("AA") else "BB"),  # bad signature
    lambda t: "garbage",
    lambda t: t.split(".")[0],                                    # signature stripped
])  # fmt: skip
def test_forged_token_is_refused(tokens, mangle):
    a = order()
    with pytest.raises(Refused, match="bad_signature"):
        tokens.verify(mangle(tokens.issue(a, "voice", "voice", 0.8)), a)


def test_token_from_another_secret_is_refused(tokens, tmp_path):
    a = order()
    other = TokenService(b"x" * 32, tmp_path / "o.sqlite", clock=Clock())
    with pytest.raises(Refused, match="bad_signature"):
        tokens.verify(other.issue(a, "voice", "voice", 0.8), a)


def test_concurrent_reuse_only_one_wins(tokens):
    a = order()
    token = tokens.issue(a, "voice", "voice", 0.8)
    results = []

    def use():
        try:
            tokens.verify(token, a)
            results.append("ok")
        except Refused as e:
            results.append(e.reason)

    threads = [threading.Thread(target=use) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count("ok") == 1 and results.count("replayed") == 15
