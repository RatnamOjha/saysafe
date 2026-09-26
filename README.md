# earshot

Voice-locked approvals and private replies for a screenless AI wearable.

A screen quietly does two jobs: it shows who approved something, and it keeps what's shown private. A screenless band with always-on mics loses both. earshot adds them back at two points in an assistant's pipeline:

- `before_execute(action)`: only the owner's voice can approve risky actions. Risky ones need a one-time word, so a recording doesn't work. If it isn't sure, it asks for a tap on the phone.
- `before_speak(reply)`: sensitive replies (codes, balances, health, addresses) go to the phone when other people are nearby. With headphones in, it reads everything.

**Status:** work in progress. Code, tests, an eval anyone can rerun, and a demo are coming.

Independent project by Ratnam Ojha. Not affiliated with Persona.
