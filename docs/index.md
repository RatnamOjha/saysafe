# saysafe

Privacy and approval guardrails for voice agents. Rules only: no models, no network.

Voice agents speak out loud and act on what they hear. A screen used to do two quiet jobs for them: keep what's shown private, and show who approved something. saysafe does both for voice:

- **Who's listening.** A reply with a login code, a balance, a diagnosis or an address isn't read out when other people may hear. It's redacted or sent to the phone, and the rest is still said. [Private replies →](privacy.md)
- **Who's speaking.** Risky actions need proof scaled to the risk (the owner's voice, a one-time word, or a phone tap), and the approval is bound to the exact action. [Approvals →](approvals.md)

saysafe never touches the mic or the speaker. Your app passes text (and, optionally, voice scores) in and gets back what to say.

## Install

```bash
pip install saysafe                # core: Python 3.10+, no torch
pip install "saysafe[pipecat]"     # Pipecat step (Pipecat needs Python 3.11+)
pip install "saysafe[voice]"       # voice ID: torch (CPU), speechbrain
```

## In 30 seconds

```python
from saysafe import PrivacyGuard, Room

guard = PrivacyGuard()
d = guard.check("Your Chase code is 482913.", room=Room.OTHERS_PRESENT)
d.say            # "I sent it to your phone."
d.send_to_phone  # "Your Chase code is 482913."
```

```python
import secrets
from saysafe import Action, ApprovalGuard

approvals = ApprovalGuard(secret=secrets.token_bytes(32))
step = approvals.start(Action(type="send_money", counterparty="Jake", amount=50))
step.say  # "Someone else might be listening. Check your phone to approve."
          # (the room is unknown, so the amount isn't read out)
```

## Who it's for

- **Devices that talk to one owner out loud:** wearables, home assistants, car assistants, robots. Both halves fit, voice ID included.
- **Agents that take actions by voice,** including phone agents: read-backs, one-time words, phone taps and approvals bound to the action.
- **Anything that sends LLM text to text-to-speech:** the privacy check as an output filter.

## What it isn't

Voice ID is a convenience, not security: recordings and live voice clones can fool it. Rules catch the patterns they know, in English. Read the [threat model](threat-model.md) before relying on it.
