# earshot

## What this is
A layer for a screenless AI wearable (target: Persona Band) that answers two questions a screen used to answer:
1. Who's speaking? Only the owner's voice can approve risky actions (payments, orders, cancellations).
2. Who's listening? Sensitive replies (codes, balances, health, addresses) don't get read out loud when other people are around.

It plugs into two points of an assistant's pipeline:
- before_execute(action): after the agent proposes an action, before it runs.
- before_speak(reply): after the agent writes a reply, before TTS plays it.

Independent concept project by Ratnam Ojha. Not affiliated with Persona. Never use Persona's logo, colors, fonts, or wording that implies it's official.

## The band (from Persona's public product page)
No wake word, no screen, one LED ring (listening / working / done). Two mics that pick up a whisper across a room, a speaker, audio handoff to headphones. Approvals by tap or voice. Pairs with a phone app.
We don't have the hardware. We simulate it with a laptop mic and speaker plus a web page with a ring.

## Principles
- Fail closed on money and secrets, fail open on harmless stuff.
- Deterministic core: decisions come from scores and policy tables. LLMs are optional helpers with timeouts and a non-LLM fallback. Everything runs with no API key.
- An approval is bound to the exact action that was read back. Change any field after approval and it's void.
- Voiceprints stay local, encrypted at rest. Never log raw audio or embeddings.
- Every decision emits a trace event (scores, thresholds, rule that fired, latency per step) for the demo UI.
- Never fabricate, round up, or hardcode evaluation numbers. Every number in the README comes from a script in eval/ that anyone can rerun.

## Scope: 48 hours
Pick the simplest thing that works and is honest. No frontend framework. No premature abstractions. One interface each for STT, TTS and LLM, that's it.

## Stack
- Python 3.11, uv, FastAPI + websockets, pydantic v2, typer, pytest, ruff
- Audio: 16 kHz mono float32 everywhere. silero-vad, faster-whisper, SpeechBrain ECAPA (speechbrain/spkrec-ecapa-voxceleb), Piper TTS
- LLM: only through earshot/llm.py, an OpenAI-compatible client (openai SDK with a custom base_url). Default provider is Groq. Base URL, key and model names come from env.
- Demo UI: one static HTML/CSS/JS page served by FastAPI
- Eval: numpy, scikit-learn, matplotlib

## Layout
earshot/
  llm.py
  audio/       io, capture, vad, stt, tts
  identity/    embed, enroll, verify, profile_store
  approvals/   actions, policy, readback, challenge, response, hook, tokens, audit
  privacy/     detect, audience, whisper, route, rewrite, hook
  agent/       fake_user.json, mock_agent, executor, channels, pipeline, events
  server/      app.py, static/
config/        *.yaml, challenge_words.txt
eval/          scripts, reports/ (eval/data/ is gitignored)
scripts/
docs/
media/         gitignored
tests/

## Working rules
- Small commits with clear messages.
- Every module ships with tests. `uv run pytest -m "not models"` passes before you say you're done.
- Tests that need downloaded models are marked @pytest.mark.models.
- Tunable numbers live in config/*.yaml.
- If something needs a mic, speakers or my phone, say so and give me the exact command to run.
- End every task with: what changed, how to run it, test output, open questions. Keep it short.
