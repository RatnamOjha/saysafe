"""Maps an Action to a proof tier (none / voice / voice_challenge / phone_tap).

The rules are data in config/policy.yaml; this file only evaluates them.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from earshot.approvals.actions import Action
from earshot.config import load_yaml

Tier = Literal["none", "voice", "voice_challenge", "phone_tap"]
TIERS: list[Tier] = ["none", "voice", "voice_challenge", "phone_tap"]


@dataclass(frozen=True)
class RiskAssessment:
    tier: Tier
    rule_id: str
    reasons: list[str] = field(default_factory=list)
    base_tier: Tier | None = None  # before the source bump; None if not bumped

    @property
    def bumped(self) -> bool:
        return self.base_tier is not None


def _matches(when: dict, a: Action) -> bool:
    for key, want in when.items():
        if key not in _KNOWN:
            raise ValueError(f"unknown policy condition {key!r}")
        if key == "type" and a.type != want:
            return False
        if key == "type_in" and a.type not in want:
            return False
        if key == "is_new_counterparty" and a.is_new_counterparty != want:
            return False
        if key == "amount_over" and not (a.amount is not None and a.amount > Decimal(want)):
            return False
        if key == "amount_at_least" and not (a.amount is not None and a.amount >= Decimal(want)):
            return False
        if key == "amount_under_or_missing" and a.amount is not None and a.amount >= Decimal(want):
            return False
    return True


_KNOWN = {
    "type", "type_in", "is_new_counterparty", "amount_over", "amount_at_least",
    "amount_under_or_missing",
}  # fmt: skip


def bump(tier: Tier) -> Tier:
    return TIERS[min(TIERS.index(tier) + 1, len(TIERS) - 1)]


def assess(action: Action, policy: dict | None = None) -> RiskAssessment:
    policy = policy or load_yaml("policy")
    for rule in policy["rules"]:
        if _matches(rule.get("when") or {}, action):
            break
    else:  # a policy file without a default rule still fails closed
        rule = {"id": "default", "tier": "phone_tap", "reason": "No rule matched"}

    amount = f"${action.amount}" if action.amount is not None else "no amount"
    reasons = [rule["reason"].format(counterparty=action.counterparty or "them", amount=amount)]
    tier: Tier = rule["tier"]
    bump_reason = (policy.get("source_bump") or {}).get(action.source)
    if bump_reason:
        reasons.append(bump_reason)
        return RiskAssessment(bump(tier), rule["id"], reasons, base_tier=tier)
    return RiskAssessment(tier, rule["id"], reasons)
