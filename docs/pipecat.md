# Pipecat

`pip install "saysafe[pipecat]"` (Pipecat needs Python 3.11+). `PrivacyFilter` is one pipeline step, right after the LLM:

```python
from saysafe import PrivacyGuard, Room
from saysafe.integrations.pipecat import PrivacyFilter

async def push_to_phone(text: str) -> None:
    ...  # your push notification or companion app; raise if it didn't arrive

privacy = PrivacyFilter(
    send_to_phone=push_to_phone,
    tool_sources={"get_login_code": {"otp"}, "get_balance": {"bank"}},
    guard=PrivacyGuard(),          # optional: a guard with an LLM classifier, screen="watch"...
)
pipeline = Pipeline([
    transport.input(), stt, user_aggregator, llm,
    privacy,
    tts, transport.output(), assistant_aggregator,
])
```

## Telling it who may hear

Set the attributes whenever your app reports device state, or pass callables that are read for every sentence:

```python
privacy.room = Room.OTHERS_PRESENT
privacy.headphones = True

privacy = PrivacyFilter(send_to_phone=push_to_phone, room=tracker.room)  # voice ID
```

## What it does with a reply

- Sentences stream straight through while this room may hear them, so ordinary replies aren't slower.
- A sentence that may not, or that has a number in it, is held: the words that make a number secret can come after it ("It's 482913. That's your code."). From then on, the rest of the reply is decided in one go: spoken as is, spoken redacted with the full reply sent to the phone, or sent to the phone with one short line spoken. So there's never more than one "I sent it to your phone" per reply.
- Alone or in headphones, nothing is checked or held.
- Results of the functions named in `tool_sources` tag the next reply (e.g. `otp` makes it secret).
- `TTSSpeakFrame`s are checked too.
- An interruption drops whatever was held.
- If `send_to_phone` raises, it says "I couldn't send that to your phone, and I can't say it out loud here." instead of the reply.

The LLM context records what was actually said ("I sent it to your phone."). Function results, codes included, stay in the context as usual, so "what was that code again?" still works once the user is alone.

The `on_decision` event fires for every reply checked as a whole:

```python
@privacy.event_handler("on_decision")
async def on_decision(processor, decision):
    log.info("privacy %s %s", decision.channel, decision.categories)
```

## Try it

[`examples/pipecat_privacy.py`](https://github.com/RatnamOjha/saysafe/blob/main/examples/pipecat_privacy.py) runs a text-only pipeline with a real LLM (Groq's free tier) and fake tools, nobody else around and then with others in the room:

```
Room: OTHERS_PRESENT
  user:    What's my Chase login code?
  phone:   Your Chase login code is 482913.
  speaker: I sent it to your phone.
  user:    How much is in my checking account?
  phone:   Your checking account balance is $2,412.55.
  speaker: Your checking account balance is on your phone.
```
