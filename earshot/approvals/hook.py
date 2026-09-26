"""before_execute(action): the approval decision flow.

1. Assess risk. Tier none: issue a token and continue.
2. voice / voice_challenge: speak the read-back (with a one-time word if needed),
   listen for the reply, transcribe it.
3. Score the reply voice against the owner, fuse with the command's score.
4. APPROVE only if the fused score is in the accept band, neither score is below
   t_reject, and the reply matched (affirm, or challenge_ok for the challenge tier).
   REJECT on negate or silence. Anything else STEPS UP to a phone tap.
   phone_tap tier goes straight to the phone.

Every decision is audited (no audio, embeddings or transcripts) and traced.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from earshot.approvals.actions import Action, action_hash
from earshot.approvals.audit import AuditLog
from earshot.approvals.challenge import ChallengeIssuer
from earshot.approvals.pending import PendingApprovals
from earshot.approvals.pending import pending as default_pending
from earshot.approvals.policy import RiskAssessment, assess
from earshot.approvals.readback import readback
from earshot.approvals.response import Match, match_response
from earshot.approvals.tokens import TokenService, default_service
from earshot.config import env, load_yaml
from earshot.identity.verify import Thresholds, VerifyResult, thresholds

if TYPE_CHECKING:
    from earshot.agent.pipeline import TurnContext
    from earshot.audio.stt import STT
    from earshot.identity.embed import Embedding
    from earshot.identity.profile_store import Profile

log = logging.getLogger(__name__)

Outcome = Literal["approve", "step_up", "reject"]

STEP_UP_LINE = "I couldn't confirm it's you. Tap on your phone to approve."
REJECT_LINE = "Okay, I won't."


@dataclass
class ApprovalDecision:
    outcome: Outcome
    token: str | None = None  # present only when outcome == "approve"
    message: str | None = None  # what to say after the decision, if anything
    reasons: list[str] = field(default_factory=list)


class VoiceScorer:
    """Scores audio and command embeddings against the owner's profile."""

    def __init__(self, profile: Profile):
        self.profile = profile

    def score_audio(self, audio: np.ndarray) -> VerifyResult:
        """Replies are short, so they use the reply minimum, not the enrollment one."""
        from functools import partial

        from earshot.identity.embed import embed
        from earshot.identity.verify import verify

        minimum = load_yaml("policy")["voice"]["min_reply_speech_s"]
        return verify(audio, self.profile, embedder=partial(embed, min_speech_s=minimum))

    def score_embedding(self, embedding: Embedding) -> float:
        from earshot.identity.embed import cosine

        return cosine(embedding.vector, self.profile.mean)


def load_owner_scorer() -> VoiceScorer | None:
    """The enrolled owner's scorer, or None (which makes every voice check step up)."""
    from earshot.identity.profile_store import ProfileError, ProfileStore

    name = env("EARSHOT_OWNER", "owner")
    try:
        return VoiceScorer(ProfileStore().load(name))
    except ProfileError as e:
        log.warning("approvals: no usable owner profile (%s); voice approvals will step up", e)
        return None


def fuse(
    reply: float | None, command: float | None, t: Thresholds, weights: dict
) -> tuple[float | None, str]:
    """(fused score, band). Missing reply score -> uncertain. Either score under
    t_reject -> never accept."""
    from earshot.identity.verify import band

    if reply is None:
        return None, "uncertain"
    if command is None:
        fused = reply
    else:
        fused = weights["reply_weight"] * reply + weights["command_weight"] * command
    b = band(fused, t)
    if b == "accept" and min(s for s in (reply, command) if s is not None) < t.t_reject:
        b = "uncertain"
    return fused, b


class Approver:
    """before_execute with every dependency injectable (STT, scorer, clock, tokens)."""

    def __init__(
        self,
        *,
        stt: STT | None = None,
        scorer: VoiceScorer | None | Callable[[], VoiceScorer | None] = load_owner_scorer,
        tokens: TokenService | None = None,
        pending: PendingApprovals | None = None,
        issuer: ChallengeIssuer | None = None,
        audit: AuditLog | None = None,
        clock: Callable[[], float] = time.time,
        t: Thresholds | None = None,
    ):
        self._stt = stt
        self._scorer = scorer
        self._tokens = tokens
        self.pending = pending or default_pending
        self.clock = clock
        self.issuer = issuer or ChallengeIssuer(clock=clock)
        self.audit = audit or AuditLog()
        self.t = t
        self.cfg = load_yaml("policy")["voice"]

    @property
    def tokens(self) -> TokenService:
        return self._tokens or default_service()

    @property
    def stt(self) -> STT:
        if self._stt is None:
            from earshot.audio.stt import get_stt

            self._stt = get_stt("reply")
        return self._stt

    @property
    def scorer(self) -> VoiceScorer | None:
        if callable(self._scorer) and not isinstance(self._scorer, VoiceScorer):
            self._scorer = self._scorer()  # load the profile once, on first use
        return self._scorer

    # the flow

    def before_execute(self, action: Action, ctx: TurnContext) -> ApprovalDecision:
        start = time.perf_counter()
        latency: dict[str, float] = {}
        ev = ctx.events
        risk = assess(action)
        ev.publish(
            "risk_assessed", action_id=action.id, tier=risk.tier, rule_id=risk.rule_id,
            reasons=risk.reasons, bumped=risk.bumped, action=action.model_dump(mode="json"),
        )  # fmt: skip
        log_base = dict(
            action_id=action.id, action_hash=action_hash(action), type=action.type,
            source=action.source, tier=risk.tier, rule_id=risk.rule_id, reasons=risk.reasons,
        )  # fmt: skip

        def finish(decision: ApprovalDecision, method: str, **extra) -> ApprovalDecision:
            latency["total"] = round((time.perf_counter() - start) * 1000, 1)
            ev.publish(
                "decision", action_id=action.id, outcome=decision.outcome, method=method,
                latency_ms_by_step=latency, **extra,
            )  # fmt: skip
            self.audit.record(
                **log_base, outcome=decision.outcome, method=method, latency_ms=latency, **extra
            )
            ev.led({"approve": "done", "step_up": "amber", "reject": "off"}[decision.outcome])
            return decision

        if risk.tier == "none":
            token = self.tokens.issue(action, risk.tier, method="none", fused_score=None)
            return finish(ApprovalDecision("approve", token, reasons=risk.reasons), "none")

        if risk.tier == "phone_tap":
            rb = readback(action, risk)
            with ev.step("readback_spoken", text=rb.text, tier=risk.tier):
                ctx.speak(rb.text)
            self._request_phone(action, risk, ctx, rb.body)
            return finish(ApprovalDecision("step_up", reasons=risk.reasons), "phone_tap")

        # voice or voice_challenge
        challenge = self.issuer.issue(action.id) if risk.tier == "voice_challenge" else None
        rb = readback(action, risk, challenge.word if challenge else None)
        t0 = time.perf_counter()
        ctx.speak(rb.text)
        latency["readback"] = _ms(t0)
        ev.publish(
            "readback_spoken", text=rb.text, tier=risk.tier,
            challenge_word=challenge.word if challenge else None,
        )  # fmt: skip

        ev.led("listening")
        t0 = time.perf_counter()
        heard = ctx.listen(self.cfg["reply_timeout_s"])
        latency["listen"] = _ms(t0)
        if heard is None:
            ev.publish("reply_captured", text=None, timed_out=True)
            if challenge:
                self.issuer.consume(challenge)
            ctx.speak(REJECT_LINE)
            return finish(ApprovalDecision("reject", reasons=["No reply"]), "voice",
                          match="timeout")  # fmt: skip

        # transcript + voice score
        if isinstance(heard, str):  # text mode: no audio, so no voice evidence
            transcript, reply_result = heard, None
        else:
            t0 = time.perf_counter()
            transcript = self.stt.transcribe(heard).text
            latency["stt"] = _ms(t0)
            t0 = time.perf_counter()
            reply_result = self.scorer.score_audio(heard) if self.scorer else None
            latency["speaker"] = _ms(t0)
        ev.publish(
            "reply_captured", text=transcript, timed_out=False,
            speech_seconds=None if reply_result is None else round(reply_result.speech_seconds, 2),
        )  # fmt: skip

        t = self.t or thresholds()
        reply_score = reply_result.score if reply_result else None
        command_score = (
            self.scorer.score_embedding(ctx.command_embedding)
            if self.scorer and ctx.command_embedding is not None
            else None
        )
        fused, voice_band = fuse(reply_score, command_score, t, self.cfg)
        scores = {
            "reply": _r(reply_score), "command": _r(command_score), "fused": _r(fused),
            "band": voice_band, "t_accept": t.t_accept, "t_reject": t.t_reject,
        }  # fmt: skip
        speech = reply_result.speech_seconds if reply_result else 0.0
        ev.publish("speaker_scored", **scores, speech_seconds=round(speech, 2),
                   owner_enrolled=self.scorer is not None)  # fmt: skip

        t0 = time.perf_counter()
        match: Match = match_response(transcript, risk.tier, challenge, self.clock())
        latency["match"] = _ms(t0)
        if challenge:
            self.issuer.consume(challenge)  # one reply per word, right or wrong
        ev.publish("reply_matched", kind=match.kind, detail=match.detail)

        audit_extra = dict(scores=scores, speech_seconds=round(speech, 2), match=match.kind)
        wanted = "challenge_ok" if risk.tier == "voice_challenge" else "affirm"

        if match.kind == "negate":
            ctx.speak(REJECT_LINE)
            return finish(ApprovalDecision("reject", reasons=[match.detail]), "voice",
                          **audit_extra)  # fmt: skip
        if match.kind == wanted and voice_band == "accept":
            token = self.tokens.issue(action, risk.tier, method="voice", fused_score=fused)
            return finish(ApprovalDecision("approve", token, reasons=risk.reasons), "voice",
                          **audit_extra)  # fmt: skip

        why = _step_up_reason(match, wanted, voice_band, reply_result, self.scorer is not None)
        ctx.speak(STEP_UP_LINE)
        self._request_phone(action, risk, ctx, rb.body)
        return finish(ApprovalDecision("step_up", reasons=[why]), "phone_tap", **audit_extra)

    def _request_phone(
        self, action: Action, risk: RiskAssessment, ctx: TurnContext, summary: str
    ) -> None:
        def denied(a: Action) -> None:
            ctx.events.publish("decision", action_id=a.id, outcome="reject", method="phone_tap")
            ctx.events.led("off")
            self.audit.record(action_id=a.id, action_hash=action_hash(a), type=a.type,
                              tier=risk.tier, outcome="reject", method="phone_tap")  # fmt: skip

        def approved(a: Action, token: str) -> None:
            self.audit.record(action_id=a.id, action_hash=action_hash(a), type=a.type,
                              tier=risk.tier, outcome="approve", method="phone_tap")  # fmt: skip
            ctx.complete_approved(a, token)

        self.pending.add(action, risk.tier, on_approved=approved, on_denied=denied)
        ctx.phone.request_approval(action.id, summary)


def _step_up_reason(
    match: Match, wanted: str, band: str, reply: VerifyResult | None, enrolled: bool
) -> str:
    if not enrolled:
        return "No owner voice enrolled"
    if reply is None:
        return "No voice to check (text reply)"
    if reply.too_short:
        return "Too little speech to check the voice"
    if match.kind == "challenge_wrong":
        return f"Wrong challenge word ({match.detail})"
    if band != "accept":
        return f"Voice didn't match the owner well enough ({band})"
    return f"Reply wasn't a clear {'challenge word' if wanted == 'challenge_ok' else 'yes'}"


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


_approver: Approver | None = None


def get_approver() -> Approver:
    global _approver
    if _approver is None:
        _approver = Approver()
    return _approver


def before_execute(action: Action, ctx: TurnContext) -> ApprovalDecision:
    """Decide whether `action` may run. See the module docstring for the flow.

    ctx provides: speak(text), listen(timeout_s) -> audio | typed str | None,
    command_embedding, phone, events, complete_approved(action, token).
    """
    return get_approver().before_execute(action, ctx)
