# saysafe

Privacy and approval guardrails for voice agents. Rules only: no models, no network.

Voice agents speak out loud and act on what they hear. saysafe handles two jobs a screen used to do:

- **Who's listening.** A reply with a login code, a balance, a diagnosis or an address isn't read out when other people may hear. It's redacted or sent to the phone, and the rest is still said.
- **Who's speaking.** Risky actions need proof scaled to the risk (the owner's voice, a one-time word, or a phone tap), and the approval is bound to the exact action. *The public API for this is in progress.*

saysafe never touches the mic or the speaker. Your app passes text in and gets back what to say.

## Install

Not on PyPI yet. Until then, from a clone:

```bash
pip install .                # core: Python 3.10+, no torch
pip install ".[pipecat]"     # Pipecat step (Pipecat needs Python 3.11+)
pip install ".[voice]"       # voice ID: torch, speechbrain
```

## Check a reply

```python
from saysafe import PrivacyGuard, Room

guard = PrivacyGuard()

d = guard.check("Your Chase code is 482913.", room=Room.OTHERS_PRESENT)
d.say            # "I sent it to your phone."
d.send_to_phone  # "Your Chase code is 482913."

guard.check("Your rent of $2,400 is due Friday.").say
# "Your rent is due Friday. Details are on your phone."   (room unknown, so cautious)
```

Speak `d.say`. When `d.send_to_phone` isn't `None`, deliver it privately: a push notification, your companion app.

### Where `room` comes from

Your device usually knows more than the audio does. Map the signals you have:

| Signal | Pass |
|---|---|
| Earbuds or headphones connected | `headphones=True` (everything is said) |
| Guest mode, passengers in the car, a meeting on the calendar | `room=Room.OTHERS_PRESENT` |
| The user switched on private mode | `room=Room.ALONE` |
| Nothing known | `room=Room.UNKNOWN` (the default) |
| Voice ID: other voices heard recently | `room=tracker.room()` |

`sources={"otp"}` (or `bank`, `health`, `email`, `calendar`) tells it where the data came from, so a reply built from a login-code tool stays secret even if the code is spelled out in words. `discreet=True` keeps anything personal off the speaker.

What goes where (the table lives in `saysafe/data/routing.yaml`):

| | alone | unknown | others present |
|---|---|---|---|
| public: weather, trivia | said | said | said |
| personal: names, calendar, email senders | said | said | redacted + phone |
| sensitive: balances, health, addresses, legal | said | redacted + phone | redacted + phone |
| secret: codes, passwords, card numbers | said | phone | phone |

## Pipecat

```python
from saysafe import Room
from saysafe.integrations.pipecat import PrivacyFilter

privacy = PrivacyFilter(
    send_to_phone=push_to_phone,   # sync or async; raise if it didn't arrive
    tool_sources={"get_login_code": {"otp"}, "get_balance": {"bank"}},
)
pipeline = Pipeline([transport.input(), stt, user_aggregator, llm, privacy, tts,
                     transport.output(), assistant_aggregator])

# whenever your app reports device state
privacy.room = Room.OTHERS_PRESENT
privacy.headphones = True
```

Sentences stream straight through while the room may hear them, so ordinary replies aren't slower. A sentence that may not, or that contains a number, is held, and the rest of the reply is decided in one go. Alone or in headphones nothing is held. If `send_to_phone` raises, it says it couldn't reach the phone instead of reading the reply out.

A real run of [examples/pipecat_privacy.py](examples/pipecat_privacy.py) (Groq, fake tool data; the LLM's wording varies between runs):

```
Room: OTHERS_PRESENT
  user:    What's my Chase login code?
  phone:   Your Chase login code is 482913.
  speaker: I sent it to your phone.
  user:    How much is in my checking account?
  phone:   Your checking account balance is $2,412.55.
  speaker: Your checking account balance is on your phone.
  user:    How far is the Moon?
  speaker: The Moon is about 384,400 kilometers (≈238,900 miles) from Earth on average.
```

## Optional LLM layers

Rules catch the patterns they know (`saysafe/data/sensitivity.yaml`). Plug in any LLM for more:

```python
from saysafe.privacy.detect import CLASSIFIER_PROMPT, Verdict
from saysafe.privacy.rewrite import smoother_prompt

guard = PrivacyGuard(classifier=my_classifier, smoother=my_smoother)
# classifier(text) -> Verdict | None; smoother(text, forbidden) -> str | None
```

If the classifier fails and no rule fired, the reply counts as personal, never public.

## Limits

- English only, and the rules catch what they know. The optional classifier catches more.
- Audio can't hear a silent person, so an unknown room is treated cautiously. Device signals beat guessing.
- Voice ID is a convenience, not security: recordings and voice clones can fool it. The one-time word and the phone tap carry the security for risky actions.
- It doesn't make a product PCI or HIPAA compliant.

## Status

Alpha. `PrivacyGuard` and the Pipecat step are ready to try. The approvals API (`ApprovalGuard`) and benchmarks on public data are in progress.

## Results

<!-- results:start -->
Not measured yet. Run `make eval`.
<!-- results:end -->

## Development

```bash
uv sync
uv run pytest -m "not models"
uv run band-demo chat --room others   # the wearable simulator in examples/band_demo
```

Apache-2.0. Independent project by Ratnam Ojha.
