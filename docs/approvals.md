# Approvals

`ApprovalGuard` decides whether an action may run, one step at a time. Your app does the talking and listening in between.

```python
import secrets
from saysafe import Action, ApprovalGuard, Room

approvals = ApprovalGuard(secret=SECRET)  # >= 32 random bytes, fixed, server-side

action = Action(type="send_money", counterparty="Jake", amount=50)
step = approvals.start(action, room=Room.ALONE)
speak(step.say)                              # "Jake, fifty dollars. Say 'copper' to confirm."

if step.status == "awaiting_reply":
    transcript, score = listen_and_score()   # your STT; your voice ID or saysafe.voice
    step = approvals.reply(step, transcript, voice_score=score)
    speak(step.say)                          # "" when approved, "Okay, I won't." on a no

if step.status == "awaiting_phone":
    push_to_phone(step.id, step.phone_request)
    # later, from your phone webhook:
    step = approvals.phone_approve(step.id)  # or phone_deny(step.id)

if step.status == "approved":
    run(action, step.token)
```

And in the executor, right before running:

```python
approvals.verify(token, action)  # raises Refused: bad_signature, action_changed, expired, replayed
```

## Steps

Every call returns the latest `Step`, a frozen snapshot:

| Field | Meaning |
|---|---|
| `status` | `approved`, `rejected`, `awaiting_reply`, `awaiting_phone`, `expired` |
| `say` | what to say now (`""` for nothing) |
| `token` | set when approved: pass it to the executor |
| `phone_request` | set on entering `awaiting_phone`: show it with Approve / Deny |
| `tier`, `reasons`, `rule_id`, `method` | why, and how it was decided |

A call that doesn't fit the step's status (a second phone tap, a reply after a decision, an unknown id) raises `StepClosed`, which carries `.status`. Map it to HTTP 409 (or 404 when `.status` is `None`).

## Tiers and the policy

| Tier | What the owner does |
|---|---|
| `none` | nothing; approved at once |
| `voice` | says yes, and the voice matches |
| `voice_challenge` | says a one-time word (random, 30 s, single use), and the voice matches |
| `phone_tap` | taps Approve on the phone |

A voice step approves only when the reply matches **and** the voice score is in the accept band. A "no" anywhere in the reply rejects ("yes, no wait" is a no). Anything else steps up to the phone: a voice that doesn't match, no voice score, an unclear reply, a wrong or expired word.

The policy maps actions to tiers. Rules are checked in order and the first match wins:

```yaml
rules:
  - id: new_payee
    when: {type: send_money, is_new_counterparty: true}
    tier: phone_tap
    reason: "New payee: you haven't paid {counterparty} before"
  - id: door
    when: {type: unlock_door}
    tier: voice_challenge
    reason: Opens the house
  - id: default
    when: {}
    tier: phone_tap
    reason: Not covered by a rule
```

Conditions: `type`, `type_in`, `is_new_counterparty`, `amount_over`, `amount_at_least`, `amount_under_or_missing`. Pass the policy as a dict or a YAML path: `ApprovalGuard(secret, policy="policy.yaml")`. Sections you leave out keep their defaults; see the [packaged policy](https://github.com/RatnamOjha/saysafe/blob/main/src/saysafe/data/policy.yaml) for all of them (expiry times, voice weights, private read-backs).

Actions whose `source` is `agent_initiated` or `from_content` need one tier more. Set it honestly: it's how prompt injection is kept from moving money with a plain "yes".

## Who's listening, again

A read-back says the amount and the one-time word out loud. For `send_money` (configurable under `private_readback`), if the room isn't one where a sensitive reply may be said, the read-back goes to the phone instead: "Someone else might be listening. Check your phone to approve." Pass `room=` and `headphones=` to `start`, as for `PrivacyGuard`.

## Voice scores

`voice_score` is how much the reply sounds like the owner: cosine similarity to their voiceprint, from [saysafe.voice](voice.md) or your own model. `command_score` (optional) is the same for the spoken command, fused 0.4 / 0.6 with the reply. Either one under `t_reject` blocks an approval. Pass `None` when you couldn't score (too short, typed, nobody enrolled): the step then goes to the phone.

If your app knows who's speaking another way (a signed-in phone line, a paired device), `ApprovalGuard(..., require_voice_score=False)` lets a matching reply approve without a score. A score you do pass still counts.

## Storage, events and audit

- `nonces=`: where used tokens are recorded. Memory by default; a path for SQLite (survives restarts), or your own `NonceStore` (one atomic `burn(nonce, at) -> bool`).
- `steps=`: where steps live between calls. Memory by default (the last 10,000). Implement `StepStore` (`get`, `put`, `swap`) for a store shared by several processes, e.g. when the phone webhook runs in another worker.
- `on_event(name, data)`: `risk_assessed`, `readback`, `voice_scored`, `reply_matched`, `decision`. Good for a trace UI.
- `audit=AuditLog(path)`: one JSON line per decision, from an allow-list of fields: no audio, transcripts or embeddings.

## The phone channel

saysafe can't tell who pressed Approve. Authenticate your phone channel: a signed-in app session or a device key. An unauthenticated approve URL lets anyone who sees it approve.
