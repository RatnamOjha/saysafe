# Threat model

What saysafe defends against, how, and what it leaves to you. Read this before you put it in front of anything that moves money or opens doors.

## What it protects

1. **Private replies.** Login codes, balances, health, addresses, legal matters and the like shouldn't be read out loud when someone else may hear.
2. **Consent for actions.** An action runs only with proof from the owner, scaled to the risk, and only exactly as the owner approved it.

## Threats and defenses

| Threat | Defense | What's left |
|---|---|---|
| Someone nearby hears a private reply | `PrivacyGuard` routes by sensitivity and room: redacts, or sends to the phone. Unknown rooms are treated cautiously. | The room signal can be wrong. Audio can't hear a silent person, so prefer device signals. Rules miss what they don't know (English only); add the optional classifier for more. |
| Someone else (or the TV) gives a command or says "yes" | A spoken yes approves only if the voice score is in the accept band. Otherwise it steps up to a phone tap. | Voice ID makes mistakes. Calibrate thresholds on your mic and users (`saysafe.calibrate`). |
| A recording of the owner saying "yes" | Money and bumped actions need a one-time word: random, bound to one action, expires in 30 s, used once. A plain yes only works for the voice tier, which the default policy keeps for purchases under $75 and emails to known contacts. | A replayed yes still passes the voice tier if the voice score passes. Move actions to `voice_challenge` or `phone_tap` in your policy if that matters. |
| A live voice clone says the one-time word | **Not defended.** A real-time clone can hear the word and say it. | Put anything you can't afford to lose on `phone_tap`. Replay and clone detection are on the roadmap. |
| Prompt injection: an email or web page asks the agent to act | `Action(source="from_content")` raises the tier by one, and the read-back says "This came from an email." | You must set `source` honestly. If the agent labels an injected action as `user_voice`, there's no bump. |
| The LLM changes the payee or amount after the read-back | The approval token is bound to a hash of the whole action. Change any field and `verify` raises `Refused("action_changed")`. | Your executor must call `verify(token, action)` with the action it's about to run, right before running it. |
| An approval token is reused or forged | HMAC-SHA256 with your secret; a single-use nonce burned atomically; 60 s expiry. | Keep the secret server-side, at least 32 random bytes. Use `SQLiteNonceStore` (or your own store) so replays stay blocked across restarts within the token's lifetime. |
| A phone tap that wasn't the owner | Each phone request resolves once, and expires after 120 s. | **You must authenticate the tap.** saysafe can't tell who pressed Approve. The demo's ntfy buttons call an unauthenticated URL; a real app should require a signed-in session or a device key. |
| Voiceprints are stolen | `ProfileStore` encrypts profiles at rest (Fernet), with the key passed in or kept in the OS keychain. | Embeddings are biometric data even though they can't be turned back into audio. Treat them like passwords. |
| Logs leak what was said | `AuditLog` writes only allow-listed fields: no audio, transcripts or embeddings. Event details name the matched word, never the transcript. | Your own logging of `on_event` data and transcripts is up to you. |

## Assumptions

- The host passes accurate `room`, `headphones` and `sources`, and sets `Action.source` honestly.
- The executor verifies the token immediately before running the action, with the exact action it runs.
- The approval secret and the profile key stay on your side, never in the client.
- Clocks are roughly right; expiry uses the host's clock.

## Out of scope

- Securing your transport, servers and phone channel.
- Jailbreaks that make the agent say or do harmful things unrelated to these two checks.
- Bystanders who can see the user's phone screen.
