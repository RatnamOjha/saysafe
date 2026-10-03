# saysafe

Privacy and approval guardrails for voice agents. Rules only: no models, no network.

Voice agents speak out loud and act on what they hear. saysafe takes over two jobs a screen used to do:

- **Who's listening.** A reply with a login code, a balance, a diagnosis or an address isn't read out when other people may hear. It's redacted or sent to the phone, and the rest is still said.
- **Who's speaking.** Risky actions need proof scaled to the risk (the owner's voice, a one-time word, or a phone tap), and the approval is bound to the exact action, so the agent can't change the amount afterwards.

saysafe never touches the mic or the speaker. Your app passes text (and, optionally, voice scores) in and gets back what to say.

## Install

```bash
pip install saysafe                # core: Python 3.10+, four dependencies, no torch
pip install "saysafe[pipecat]"     # Pipecat step (Pipecat needs Python 3.11+)
pip install "saysafe[voice]"       # voice ID: torch (CPU), speechbrain
```

## 1. Keep private replies off the speaker

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

**Where `room` comes from.** Your device usually knows more than the audio does:

| Signal | Pass |
|---|---|
| Earbuds or headphones connected | `headphones=True` (everything is said) |
| Guest mode, passengers in the car, a meeting on the calendar | `room=Room.OTHERS_PRESENT` |
| The user switched on private mode | `room=Room.ALONE` |
| Nothing known | `room=Room.UNKNOWN` (the default) |
| Voice ID: other voices heard recently | `room=tracker.room()` (see 3) |

`sources={"otp"}` (or `bank`, `health`, `email`, `calendar`) says where the data came from, so a reply built from a login-code tool stays secret even if the code is spelled out in words. `discreet=True` keeps anything personal off the speaker. `PrivacyGuard(screen="watch")` changes what the spoken lines call the private device.

| | alone | unknown | others present |
|---|---|---|---|
| public: weather, trivia | said | said | said |
| personal: names, calendar, email senders | said | said | redacted + phone |
| sensitive: balances, health, addresses, legal | said | redacted + phone | redacted + phone |
| secret: codes, passwords, card numbers | said | phone | phone |

## 2. Approve actions by voice

```python
import secrets
from saysafe import Action, ApprovalGuard, Room

approvals = ApprovalGuard(secret=secrets.token_bytes(32))  # keep a fixed one server-side
action = Action(type="send_money", counterparty="Jake", amount=50)

step = approvals.start(action, room=Room.ALONE)
step.say      # "Jake, fifty dollars. Say 'copper' to confirm."   -> your TTS

# your STT hears the reply; your voice ID (or saysafe.voice) scores it
step = approvals.reply(step, transcript="copper", voice_score=0.71)
step.status   # "approved", "rejected", or "awaiting_phone"

if step.status == "awaiting_phone":
    push_to_phone(step.id, step.phone_request)    # Approve / Deny -> approvals.phone_approve(step.id)

# in the executor, right before running it (single use; fails if any field changed):
approvals.verify(step.token, action)
```

How much proof each action needs comes from a policy. The [default](https://github.com/RatnamOjha/saysafe/blob/main/src/saysafe/data/policy.yaml) is a starting point. Write your own as a dict or YAML file:

```python
approvals = ApprovalGuard(secret, policy={"rules": [
    {"id": "door", "when": {"type": "unlock_door"}, "tier": "voice_challenge", "reason": "Opens the house"},
    {"id": "big", "when": {"amount_over": 200}, "tier": "phone_tap", "reason": "Over $200"},
    {"id": "default", "when": {}, "tier": "phone_tap", "reason": "Everything else"},
]})
```

| Tier | What the owner does |
|---|---|
| `none` | nothing; approved at once |
| `voice` | says yes, and the voice must match |
| `voice_challenge` | says a one-time word (random, 30 s, single use), and the voice must match |
| `phone_tap` | taps Approve on the phone |

A "no" rejects. A voice that doesn't match, an unclear reply, or a wrong word steps up to the phone. Actions that came from an email or web page (`source="from_content"`) need one tier more. A money read-back isn't said out loud when others may hear; it goes to the phone. If your app knows who's speaking another way (a signed-in phone line), pass `require_voice_score=False`.

## 3. Voice ID (optional)

```python
from saysafe import calibrate
from saysafe.voice import ProfileStore, VoiceID, keyring_key

vid = VoiceID()                                    # ECAPA on CPU; downloads ~80 MB once
profile = vid.enroll("owner", clips)               # 5-8 clips of 16 kHz mono float32
ProfileStore("profiles", keyring_key()).save(profile)   # encrypted at rest

score = vid.score(reply_audio, profile)            # cosine, or None with too little speech
step = approvals.reply(step, transcript, voice_score=score)

tracker = vid.tracker(profile)                     # who's listening
tracker.mic_on()
tracker.observe_segment(speech_segment)            # every speech segment your VAD finds
guard.check(reply, room=tracker.room())
```

The packaged thresholds are placeholders. Record the owner and other people on the mic you ship, then `calibrate(owner_scores, other_scores)` gives you thresholds to pass as `ApprovalGuard(thresholds=...)`.

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

privacy.room = Room.OTHERS_PRESENT   # whenever your app reports device state
privacy.headphones = True
```

Sentences stream straight through while the room may hear them, so ordinary replies aren't slower. A sentence that may not, or that contains a number, is held, and the rest of the reply is decided in one go. Alone or in headphones nothing is held. If `send_to_phone` raises, it says it couldn't reach the phone instead of reading the reply out.

A real run of [examples/pipecat_privacy.py](https://github.com/RatnamOjha/saysafe/blob/main/examples/pipecat_privacy.py) (Groq, fake tool data; the LLM's wording varies between runs):

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

Rules catch the patterns they know ([sensitivity.yaml](https://github.com/RatnamOjha/saysafe/blob/main/src/saysafe/data/sensitivity.yaml)). Plug in any LLM for more:

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
- Voice ID is a convenience, not security: recordings and live voice clones can fool it. The one-time word stops recordings, and the phone tap is what to use for anything you can't afford to lose.
- The phone tap is only as good as your phone channel: authenticate it.
- Amounts are read back in dollars only, for now.
- It doesn't make a product PCI or HIPAA compliant.

The full list, with what each defense leaves to you, is in [THREAT_MODEL.md](https://github.com/RatnamOjha/saysafe/blob/main/THREAT_MODEL.md).

## Status

Alpha. 0.1 is the first release; the API may still change before 1.0. The private-reply benchmark is waiting on reviewed labels; the rest are below.

## Results

<!-- results:start -->

| Result | Value |
|---|---|
| Privacy check, per reply (p50 / p95) | 0.053 / 0.208 ms on Apple M3 |
| Voice ID equal error rate, 3 s / 0.5 s of speech | 0.4% / 5.9% |
| Strangers approved by voice, end to end (packaged thresholds, 1 s reply) | 0 of 760 (0%) |
| Owner approved without a phone tap (same) | 151 of 160 (94.4%) |
| Private-reply accuracy | not measured yet (0 of 140 labels reviewed) |

Voice numbers are on LibriSpeech (clean read speech), close to the best case. Details and caveats: [bench/results/BENCHMARKS.md](https://github.com/RatnamOjha/saysafe/blob/main/bench/results/BENCHMARKS.md). Rerun with `make bench`.

<!-- results:end -->

## Development

```bash
uv sync
uv run pytest -m "not models"
uv run band-demo chat --room others   # the wearable simulator in examples/band_demo
```

See [CONTRIBUTING.md](https://github.com/RatnamOjha/saysafe/blob/main/CONTRIBUTING.md). Security reports: [SECURITY.md](https://github.com/RatnamOjha/saysafe/blob/main/SECURITY.md).

Apache-2.0. Independent project by Ratnam Ojha.
