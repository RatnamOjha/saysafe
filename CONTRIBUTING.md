# Contributing

Thanks for helping. Bug reports with a failing example are the most valuable contribution. A reply that leaked, or an approval that shouldn't have passed, is exactly what we want to hear about.

## Setup

```bash
git clone https://github.com/RatnamOjha/saysafe && cd saysafe
uv sync                         # library + dev tools + the band demo
uv run pytest -m "not models"   # fast tests, no model downloads
uv run pytest -m models         # tests that load the ECAPA / VAD / Whisper models
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

## Principles

- Fail closed on money and secrets, fail open on harmless things.
- The core has no models and no network. LLMs plug in through callbacks (`classifier=`, `smoother=`) and are never required. Keep the core installable without torch; CI checks a core-only install on Python 3.10.
- No global state in the library: everything is an instance or a pure function. Settings are arguments with packaged defaults.
- Never log audio, embeddings or transcripts.
- Never tune rules on the data you report numbers on. Every published number comes from a script anyone can rerun, with its sample size.

## Changing detection rules

Detection rules live in `src/saysafe/privacy/detect.py` (patterns) and `src/saysafe/data/sensitivity.yaml` (levels and keyword lists). With every rule change:

1. Add cases to `tests/data/sensitivity_cases.yaml`: what should now be caught, and a public reply that must *not* be (over-redaction matters too).
2. Use made-up data only: no real names with real numbers, no real codes.

## Pull requests

- Small and focused, with tests. `uv run pytest -m "not models"`, ruff and mypy pass.
- Add a line to `CHANGELOG.md` under "Unreleased".
- By contributing, you agree your work is licensed under Apache-2.0.
