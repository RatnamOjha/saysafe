# Before you deploy

The quickstart runs in one Python process. That's good for learning and local demos. A real voice agent usually has a speech worker, an API server, and a phone app, so the host app must connect Saysafe's pieces safely.

## What Saysafe handles

- It decides what reply text may be spoken and what should go to a private screen.
- It asks for proof before a risky action.
- It issues a short-lived token bound to the exact action, and refuses changed or already-used tokens.

## What your app still handles

- **The microphone, speech recognition, speaker, and phone screen.** Saysafe doesn't check that audio was heard correctly or that the phone displayed all the details. Before the user can approve, render every field in `step.phone_details`; don't approve from the shorter `phone_request` notification.
- **Who can tap Approve.** Require a signed-in phone session or a paired-device credential. Never expose an unauthenticated approval URL.
- **The action itself.** Call `approvals.verify(token, action)` in the executor immediately before running it. If verification raises `Refused`, stop. Do not let a language model or another request change `action` after approval.
- **Room and source facts.** Pass accurate `room`, `headphones`, and tool `sources`. If the room is unknown, keep the default cautious behavior. A tool returning a code or bank balance should be tagged even if the reply text doesn't name it.
- **Secret management and delivery.** Keep a stable, randomly generated signing secret on the server. Keep it out of source control and never generate a new one every request. Deliver private text only to the authenticated user's device; if delivery fails, don't read the private text aloud.

## Storage: choose for your deployment

| Setup | Approval steps | Used-token tracking | Suitable for |
|---|---|---|---|
| Defaults | `MemoryStepStore` | `MemoryNonceStore` | A local demo in one process |
| One server, restart survival needed | Still memory unless you provide a `StepStore` | SQLite via `nonces="/path/to/nonces.sqlite"` | A single-host prototype; steps are lost on restart |
| Several workers or replicas | A shared `StepStore` implementing atomic `get` / `put` / `swap` | A shared `NonceStore` whose `burn` is atomic across workers | A multi-worker service |

SQLite nonce storage preserves used-token records across a restart, but it does not make approval steps persistent or coordinate several machines. For multiple workers, use stores backed by a shared database or Redis, and make `swap` and `burn` atomic. `swap(old, new)` must only succeed if the stored version is still `old.version`; otherwise two simultaneous replies could both advance the same approval.

## Simple deployment checklist

1. Use a fixed server-side signing key and the same key in every worker.
2. Use shared, atomic stores if requests can land on different workers or the process can restart mid-approval.
3. Authenticate phone approval and show the complete action details before the Approve button.
4. Verify the token against the final action immediately before the external tool runs.
5. Pass accurate room/headphone state and tag sensitive tool sources.
6. Fail closed: if approval verification or private delivery fails, don't perform the action or speak the private result.
7. Test the whole host-app path, including concurrent approvals, restarts, lost phone sessions, and failed private delivery.

This guide describes integration requirements; Saysafe does not ship a phone app, hosted service, distributed step store, or deployment-ready authentication system.
