"""ApprovalGuard: proof scaled to risk, one step at a time. The host owns mic and speaker.

    approvals = ApprovalGuard(secret=SECRET)  # 32 random bytes, kept server-side
    action = Action(type="send_money", counterparty="Jake", amount=50)

    step = approvals.start(action, room=Room.ALONE)
    speak(step.say)  # "Jake, fifty dollars. Say 'copper' to confirm."
    if step.status == "awaiting_reply":
        step = approvals.reply(step, transcript, voice_score=0.71)
        speak(step.say)
    if step.status == "awaiting_phone":
        ask_phone(step.id, step.phone_request)  # then approvals.phone_approve(step.id)
    if step.status == "approved":
        run(action, step.token)  # the executor calls approvals.verify(token, action)

Tiers come from the policy (saysafe/data/policy.yaml, or your own):
  none            approved at once
  voice           read back; the owner says yes
  voice_challenge read back with a one-time word; the owner says the word
  phone_tap       tap Approve on the phone
A voice step approves only when the reply matches AND the voice score is in the accept
band; a no rejects; anything else steps up to the phone. Approvals are signed tokens bound
to the exact action: change any field and verify() refuses it.
"""

from __future__ import annotations

import copy
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml

from saysafe.approvals.actions import Action, action_hash
from saysafe.approvals.audit import AuditLog
from saysafe.approvals.challenge import Challenge, ChallengeIssuer
from saysafe.approvals.fusion import Thresholds, fuse, load_thresholds
from saysafe.approvals.policy import _KNOWN, TIERS, RiskAssessment, Tier, assess
from saysafe.approvals.readback import readback, summary
from saysafe.approvals.response import Match, match_response
from saysafe.approvals.tokens import NonceStore, TokenService
from saysafe.config import default_config
from saysafe.privacy.audience import Room
from saysafe.privacy.route import table_cell

Status = Literal["approved", "rejected", "awaiting_reply", "awaiting_phone", "expired"]
Outcome = Literal["approve", "step_up", "reject"]

REJECT_LINE = "Okay, I won't."


@dataclass(frozen=True)
class Step:
    """Where an approval stands. Every ApprovalGuard call returns the latest Step."""

    id: str  # the action's id
    action: Action
    tier: Tier
    status: Status
    say: str = ""  # what to say now; "" for nothing
    token: str | None = None  # set when approved: pass it to the executor
    phone_request: str | None = None  # set on entering awaiting_phone: show it with Approve/Deny
    reasons: tuple[str, ...] = ()
    rule_id: str = ""
    method: str = ""  # how it was decided: none | voice | phone_tap
    expires_at: float | None = None
    challenge_word: str | None = field(default=None, repr=False)
    issued_at: float = 0.0
    version: int = 0

    @property
    def done(self) -> bool:
        return self.status in ("approved", "rejected", "expired")


class StepClosed(Exception):
    """The step can't take this call: it's resolved, expired, waiting for something
    else, or unknown. status is its current status, or None if unknown."""

    def __init__(self, step_id: str, status: str | None, wanted: str):
        self.step_id, self.status = step_id, status
        super().__init__(f"step {step_id} is {status or 'unknown'}, not {wanted}")


class StepStore(Protocol):
    """Where steps live between calls. Memory by default; implement this for a store
    shared by several processes (e.g. a phone webhook served by another worker)."""

    def get(self, step_id: str) -> Step | None: ...

    def put(self, step: Step) -> None: ...

    def swap(self, old: Step, new: Step) -> bool:
        """Atomically replace `old` with `new` if the stored step still has old.version."""
        ...


class MemoryStepStore:
    """Keeps the most recent `max_steps` steps; older ones are forgotten (and so closed)."""

    def __init__(self, max_steps: int = 10_000) -> None:
        self._steps: OrderedDict[str, Step] = OrderedDict()
        self._max = max_steps
        self._lock = threading.Lock()

    def get(self, step_id: str) -> Step | None:
        return self._steps.get(step_id)

    def put(self, step: Step) -> None:
        with self._lock:
            self._steps[step.id] = step
            self._steps.move_to_end(step.id)
            while len(self._steps) > self._max:
                self._steps.popitem(last=False)

    def swap(self, old: Step, new: Step) -> bool:
        with self._lock:
            current = self._steps.get(old.id)
            if current is None or current.version != old.version:
                return False
            self._steps[old.id] = new
            return True


EventHandler = Callable[[str, dict[str, Any]], None]


class ApprovalGuard:
    """Decides whether an action may run, and signs the approval.

    secret: at least 16 random bytes (32 recommended), kept server-side.
    policy: a dict or YAML path with `rules` (and optionally the other sections of
        saysafe/data/policy.yaml); sections you leave out keep the defaults.
    thresholds: voice score bands (saysafe.load_thresholds(path) loads yours).
        The packaged ones are placeholders: calibrate on your own mic and users.
    require_voice_score: True (default) means a spoken yes or word only approves with a
        voice score in the accept band. Set False only if your app knows who's speaking
        another way (e.g. a signed-in phone line); then the reply alone approves.
    nonces: NonceStore or SQLite path for used tokens (default: in memory).
    steps: StepStore (default: in memory).
    audit: AuditLog for decisions (no audio, transcripts or embeddings).
    on_event(name, data): risk_assessed, readback, voice_scored, reply_matched, decision.
    screen: what the user approves on, as said out loud: "phone", "watch", "app".
    """

    def __init__(
        self,
        secret: bytes,
        *,
        policy: dict | str | Path | None = None,
        thresholds: Thresholds | None = None,
        require_voice_score: bool = True,
        nonces: NonceStore | str | Path | None = None,
        steps: StepStore | None = None,
        audit: AuditLog | None = None,
        on_event: EventHandler | None = None,
        screen: str = "phone",
        clock: Callable[[], float] = time.time,
    ):
        self.policy = _load_policy(policy)
        self.clock = clock
        self.tokens = TokenService(secret, nonces, clock, self.policy["tokens"]["ttl_s"])
        challenge = self.policy["challenge"]
        self.issuer = ChallengeIssuer(clock, challenge["ttl_s"], challenge["avoid_recent"])
        self.thresholds = thresholds or _default_thresholds()
        self.require_voice_score = require_voice_score
        self.steps: StepStore = steps or MemoryStepStore()
        self.audit = audit
        self.on_event = on_event
        self.screen = screen

    # lines ---------------------------------------------------------------------

    def _private_line(self) -> str:
        return f"Someone else might be listening. Check your {self.screen} to approve."

    def _step_up_line(self, why: str) -> str:
        if why.startswith("Voice"):
            lead = "I couldn't confirm it's you."
        elif why.startswith("Wrong challenge"):
            lead = "That's not the word I asked for."
        else:
            lead = "I didn't catch that."
        return f"{lead} Check your {self.screen} to approve."

    # the flow ------------------------------------------------------------------

    def assess(self, action: Action) -> RiskAssessment:
        """The tier this action needs, without starting anything."""
        return assess(action, self.policy)

    def start(
        self, action: Action, *, room: Room | str = Room.UNKNOWN, headphones: bool = False
    ) -> Step:
        """Begin approving `action`. room/headphones decide whether a money read-back may
        be said out loud; if not, it goes to the phone instead."""
        now = self.clock()
        risk = self.assess(action)
        self._emit("risk_assessed", action_id=action.id, tier=risk.tier, rule_id=risk.rule_id,
                   reasons=risk.reasons, bumped=risk.bumped,
                   action=action.model_dump(mode="json"))  # fmt: skip
        base = Step(action.id, action, risk.tier, "awaiting_reply", reasons=tuple(risk.reasons),
                    rule_id=risk.rule_id, issued_at=now)  # fmt: skip

        if risk.tier == "none":
            token = self.tokens.issue(action, "none", method="none", fused_score=None)
            step = replace(base, status="approved", token=token, method="none")
            self.steps.put(step)
            return self._decided(step, "approve")

        private = self._private_reason(action, Room(room), headphones)
        if risk.tier == "phone_tap" or private:
            say = (
                self._private_line() if private else readback(action, risk, None, self.screen).text
            )
            reasons = (*base.reasons, private) if private else base.reasons
            step = self._to_phone(replace(base, reasons=reasons), say, now)
            self.steps.put(step)
            self._emit("readback", action_id=action.id, text=say, tier=risk.tier,
                       private=bool(private), challenge_word=None)  # fmt: skip
            return self._decided(step, "step_up", match="private_readback" if private else None)

        word = self.issuer.issue(action.id).word if risk.tier == "voice_challenge" else None
        rb = readback(action, risk, word, self.screen)
        ttl = self.policy["challenge"]["ttl_s"]
        step = replace(base, say=rb.text, expires_at=now + ttl, challenge_word=word)
        self.steps.put(step)
        self._emit("readback", action_id=action.id, text=rb.text, tier=risk.tier, private=False,
                   challenge_word=word)  # fmt: skip
        return step

    def reply(
        self,
        step: Step | str,
        transcript: str | None,
        *,
        voice_score: float | None = None,
        command_score: float | None = None,
    ) -> Step:
        """The owner's reply to a read-back. transcript None means nobody answered in time.

        voice_score: how much the reply sounds like the owner (e.g. cosine to their
            voiceprint), or None if you couldn't score it (too short, typed, no voice ID).
        command_score: the same for the spoken command, if you have it; fused with the reply.
        """
        current = self._get(step, "awaiting_reply")
        now = self.clock()
        audit_extra: dict[str, Any] = {}

        if transcript is None:
            new = replace(current, status="rejected", say=REJECT_LINE, method="voice",
                          reasons=("No reply",), version=current.version + 1)  # fmt: skip
            return self._commit(current, new, "reject", match="timeout")

        weights = self.policy["voice"]
        fused, voice_band = fuse(voice_score, command_score, self.thresholds,
                                 weights["reply_weight"], weights["command_weight"])  # fmt: skip
        if voice_score is None and not self.require_voice_score:
            voice_band = "accept"  # the host vouches for who's speaking
        scores = {"reply": _r(voice_score), "command": _r(command_score), "fused": _r(fused),
                  "band": voice_band, "t_accept": self.thresholds.t_accept,
                  "t_reject": self.thresholds.t_reject}  # fmt: skip
        self._emit("voice_scored", action_id=current.id, **scores)

        challenge = None
        if current.challenge_word:
            challenge = Challenge(current.challenge_word, current.id, current.issued_at,
                                  current.expires_at or now)  # fmt: skip
        match = match_response(transcript, current.tier, challenge, now)
        self._emit("reply_matched", action_id=current.id, kind=match.kind, detail=match.detail)
        audit_extra.update(scores=scores, match=match.kind)

        wanted = "challenge_ok" if current.tier == "voice_challenge" else "affirm"
        late = current.expires_at is not None and now >= current.expires_at
        bump = current.version + 1
        if match.kind == "negate":
            new = replace(current, status="rejected", say=REJECT_LINE, method="voice",
                          reasons=(match.detail,), version=bump)  # fmt: skip
            return self._commit(current, new, "reject", **audit_extra)
        if match.kind == wanted and voice_band == "accept" and not late:
            token = self.tokens.issue(current.action, current.tier, "voice", fused)
            new = replace(current, status="approved", say="", token=token, method="voice",
                          version=bump)  # fmt: skip
            return self._commit(current, new, "approve", **audit_extra)

        why = self._step_up_reason(match, wanted, voice_band, voice_score, late)
        new = self._to_phone(replace(current, reasons=(why,), version=bump),
                             self._step_up_line(why), now)  # fmt: skip
        return self._commit(current, new, "step_up", **audit_extra)

    def phone_approve(self, step: Step | str) -> Step:
        """The owner tapped Approve. Works once, and only before the request expires."""
        current = self._get(step, "awaiting_phone")
        token = self.tokens.issue(
            current.action, current.tier, method="phone_tap", fused_score=None
        )
        new = replace(current, status="approved", say="", token=token, method="phone_tap",
                      version=current.version + 1)  # fmt: skip
        return self._commit(current, new, "approve")

    def phone_deny(self, step: Step | str) -> Step:
        current = self._get(step, "awaiting_phone")
        new = replace(current, status="rejected", say=REJECT_LINE, method="phone_tap",
                      reasons=("Denied on the phone",), version=current.version + 1)  # fmt: skip
        return self._commit(current, new, "reject")

    def cancel(self, step: Step | str) -> Step:
        """Drop a step that's still waiting (the conversation moved on)."""
        current = self._get(step, "awaiting_reply or awaiting_phone")
        new = replace(current, status="rejected", say="", reasons=("Cancelled",),
                      version=current.version + 1)  # fmt: skip
        return self._commit(current, new, "reject")

    def get(self, step_id: str) -> Step | None:
        """The step as it stands now (a waiting step past its expiry reads as expired)."""
        step = self.steps.get(step_id)
        if step is not None and step.status == "awaiting_phone" and step.expires_at is not None:
            if self.clock() >= step.expires_at:
                expired = replace(step, status="expired", say="", version=step.version + 1)
                if self.steps.swap(step, expired):
                    self._emit("decision", action_id=step.id, outcome="reject",
                               method="phone_tap", reasons=["Phone request expired"])  # fmt: skip
                return self.steps.get(step_id)
        return step

    def verify(self, token: str, action: Action) -> dict:
        """For the executor, right before it runs `action`: returns the token's payload, or
        raises Refused (bad_signature, action_changed, expired, replayed). Single use."""
        return self.tokens.verify(token, action)

    # internals -----------------------------------------------------------------

    def _get(self, step: Step | str, wanted: str) -> Step:
        step_id = step if isinstance(step, str) else step.id
        current = self.get(step_id)
        if current is None or current.status not in wanted.split(" or "):
            raise StepClosed(step_id, current.status if current else None, wanted)
        return current

    def _to_phone(self, step: Step, say: str, now: float) -> Step:
        return replace(step, status="awaiting_phone", say=say, phone_request=summary(step.action),
                       method="phone_tap", expires_at=now + self.policy["phone"]["pending_ttl_s"],
                       challenge_word=None)  # fmt: skip

    def _commit(self, old: Step, new: Step, outcome: Outcome, **extra: Any) -> Step:
        if not self.steps.swap(old, new):
            latest = self.steps.get(old.id)
            raise StepClosed(old.id, latest.status if latest else None, old.status)
        return self._decided(new, outcome, **extra)

    def _decided(self, step: Step, outcome: Outcome, **extra: Any) -> Step:
        extra = {k: v for k, v in extra.items() if v is not None}
        self._emit("decision", action_id=step.id, outcome=outcome, method=step.method,
                   reasons=list(step.reasons), **extra)  # fmt: skip
        if self.audit is not None:
            a = step.action
            self.audit.record(action_id=a.id, action_hash=action_hash(a), type=a.type,
                              source=a.source, tier=step.tier, rule_id=step.rule_id,
                              reasons=list(step.reasons), outcome=outcome, method=step.method,
                              **extra)  # fmt: skip
        return step

    def _private_reason(self, action: Action, room: Room, headphones: bool) -> str | None:
        cfg = self.policy.get("private_readback") or {}
        if action.type not in cfg.get("types", []) or headphones:
            return None
        if table_cell(cfg["level"], room.value) == "speak":
            return None
        return f"Read-back kept private ({room.value.replace('_', ' ')})"

    def _step_up_reason(
        self, match: Match, wanted: str, band: str, voice_score: float | None, late: bool
    ) -> str:
        if late:
            return "Reply came after the read-back expired"
        if match.kind == "challenge_wrong":
            return f"Wrong challenge word ({match.detail})"
        if match.kind != wanted:
            return f"Reply wasn't a clear {'challenge word' if wanted == 'challenge_ok' else 'yes'}"
        if voice_score is None:
            return "Voice couldn't be checked (no voice score for the reply)"
        return f"Voice didn't match the owner well enough ({band})"

    def _emit(self, name: str, **data: Any) -> None:
        if self.on_event is not None:
            self.on_event(name, data)


def _load_policy(policy: dict | str | Path | None) -> dict:
    merged = default_config("policy")
    if policy is None:
        return merged
    if isinstance(policy, (str, Path)):
        policy = yaml.safe_load(Path(policy).read_text()) or {}
    for key, value in copy.deepcopy(policy).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    for rule in merged["rules"]:
        if rule.get("tier") not in TIERS:
            raise ValueError(f"policy rule {rule.get('id')!r}: tier must be one of {TIERS}")
        unknown = set(rule.get("when") or {}) - _KNOWN
        if unknown:
            raise ValueError(f"policy rule {rule.get('id')!r}: unknown conditions {unknown}")
        rule.setdefault("reason", rule.get("id", "policy rule"))
    return merged


def _default_thresholds() -> Thresholds:
    return load_thresholds()


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 4)
