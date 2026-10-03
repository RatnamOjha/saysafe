# Changelog

All notable changes to saysafe. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/) (before 1.0, minor versions may break the API).

## Unreleased

## [0.1.0] - 2026-10-03

First release.

### Added
- `PrivacyGuard.check(text, sources=, room=, headphones=, discreet=, context=)`: what to say out loud and what to send to the phone, from rules only (no models, no network). Optional LLM `classifier=` and `smoother=`.
- `Room`: who may hear, set from device state or from `AudienceTracker.room()`.
- `ApprovalGuard`, a step-based approvals API: `start`, `reply`, `phone_approve`, `phone_deny`, `cancel`, `verify`. Risk tiers come from a policy (packaged default, or your own dict or YAML file). Approvals are HMAC-signed, single-use tokens bound to the exact action.
- Pluggable storage: `NonceStore` (memory or SQLite) for used tokens, and `StepStore` for steps. Plus `on_event` for tracing, and `AuditLog`, which records no audio, transcripts or embeddings.
- `saysafe.voice` (the `voice` extra): `VoiceID` to enroll, score, verify and track the audience with ECAPA on CPU; `ProfileStore` for encrypted voiceprints.
- `calibrate(owner_scores, other_scores)`: voice thresholds from your own data.
- `saysafe.integrations.pipecat.PrivacyFilter` (the `pipecat` extra): a pipeline step between the LLM and TTS.
- `screen=`: what spoken lines call the private device ("phone", "watch", "app").
- `make bench`: latency, voice ID on LibriSpeech test-clean (per speech length, and end to end through `ApprovalGuard`), and private replies on reviewed labels.
