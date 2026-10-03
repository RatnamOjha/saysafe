# Private replies

`PrivacyGuard.check` decides what a reply may say out loud, given who may hear it.

```python
from saysafe import PrivacyGuard, Room

guard = PrivacyGuard()
d = guard.check("Your rent of $2,400 is due Friday.", room=Room.OTHERS_PRESENT)
d.say            # "Your rent is due Friday. Details are on your phone."
d.send_to_phone  # "Your rent of $2,400 is due Friday."
d.level          # "sensitive"
d.reasons        # ("Sensitive reply", "Others may hear", ...)
```

Speak `d.say`. When `d.send_to_phone` isn't `None`, deliver it privately.

## How it decides

1. **Detect.** Named rules find codes, passwords, card and account numbers, SSNs, the user's money, addresses, health and legal matters, and email senders. Each match is a span with a category, and the reply gets the highest level among them: `public` < `personal` < `sensitive` < `secret`.
2. **Route.** The level and the room pick a cell:

    | | alone | unknown | others present |
    |---|---|---|---|
    | public | said | said | said |
    | personal | said | said | redacted + phone |
    | sensitive | said | redacted + phone | redacted + phone |
    | secret | said | phone | phone |

3. **Rewrite.** For "redacted + phone", each span becomes a placeholder ("your code", "the amount") or is dropped, and the result is checked again. If anything flagged survived, or the sentence broke, it falls back to "Got it. I sent the details to your phone."

## Telling it who may hear

`room` is the most important input. Device signals beat audio, because audio can't hear a silent person:

| Signal | Pass |
|---|---|
| Earbuds or headphones connected | `headphones=True` (everything is said) |
| Guest mode, passengers in the car, a meeting on the calendar | `room=Room.OTHERS_PRESENT` |
| The user switched on private mode | `room=Room.ALONE` |
| Nothing known | `room=Room.UNKNOWN` (the default: cautious) |
| Voice ID: other voices heard in the last 45 s | `room=tracker.room()` ([Voice ID](voice.md)) |

`discreet=True` (a user setting) or `whisper=True` (the user whispered) keeps anything personal or above off the speaker, whoever is around.

## Telling it where the data came from

`sources` raises the floor when you know the reply was built from private data:

```python
guard.check("Here it is: four eight two nine one three.", sources={"otp"}, room=Room.UNKNOWN)
# -> phone_only, even though no rule saw digits
```

| Source | At least |
|---|---|
| `otp` | secret |
| `bank`, `health` | sensitive |
| `email`, `calendar` | personal |

## Streaming replies

LLM replies arrive in pieces. Check each sentence as it completes, passing what was already said as `context`. Context can make a later sentence sensitive ("Your code is ready." then "It's 482913."), and never makes one less so. [`says_as_is`](reference.md) and `speaks_everything` answer the cheap questions without rewriting. The [Pipecat step](pipecat.md) does all of this for you.

## Optional LLM layers

```python
from saysafe.privacy.detect import CLASSIFIER_PROMPT, Verdict
from saysafe.privacy.rewrite import smoother_prompt

def classifier(text: str) -> Verdict | None:
    ...  # call your LLM with CLASSIFIER_PROMPT; return None on failure or timeout

def smoother(text: str, forbidden: list[str]) -> str | None:
    ...  # call your LLM with smoother_prompt(forbidden); return None on failure

guard = PrivacyGuard(classifier=classifier, smoother=smoother)
```

The classifier can only raise a level. If it fails and no rule fired, the reply counts as personal, never public. Smoothed text is re-checked like any rewrite.

## Wording

`PrivacyGuard(screen="watch")` makes the lines say "I sent it to your watch." Use whatever the user would call the private device.
