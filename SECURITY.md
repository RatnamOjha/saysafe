# Security

saysafe sits between a voice agent and things that matter (money, doors, private data), so security reports are very welcome.

## Reporting a vulnerability

Please report privately through GitHub: the **Security** tab of this repository, then **Report a vulnerability**. Don't open a public issue for anything exploitable.

Include what you did, what happened, and what you expected. A failing test case is the most useful form. You'll get a reply within a week. Once a fix is released, you'll be credited in the changelog unless you'd rather not be.

## What counts

Examples of things to report:

- An approval token that verifies for an action other than the one approved, after it expired, or twice.
- A way to get `ApprovalGuard` to approve without the proof its policy requires.
- A reply that `PrivacyGuard` speaks in full to a room it shouldn't, when the same kind of data is caught elsewhere (a bypass, not a missing keyword).
- Audit logs or events that contain audio, embeddings or transcripts.
- Voice profiles readable without the key.

Known limits are listed in [THREAT_MODEL.md](THREAT_MODEL.md). For example, a live voice clone can say a one-time word. Reports that sharpen those limits are still useful; please file them as regular issues.

## Supported versions

Until 1.0, only the latest release gets security fixes.
