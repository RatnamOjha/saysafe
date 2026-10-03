# Voice ID

Optional: `pip install "saysafe[voice]"`. ECAPA speaker embeddings ([speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)) and Silero VAD, on CPU. The model downloads once (~80 MB) to `~/.cache/saysafe/ecapa`; after that nothing leaves the machine. Audio is 16 kHz mono `float32` numpy everywhere.

## Enroll

```python
from saysafe.voice import ProfileStore, VoiceID, keyring_key

vid = VoiceID()
profile = vid.enroll("owner", clips, mic_name="band v2")
```

Give 5-8 clips of about 3 s each, recorded on the mic you ship. Include the short words people will approve with ("yes ... yes ... yes"): one "yes" is too short to embed on its own. Clips with under 0.8 s of speech are skipped, and so are clips that don't sound like the rest. `EnrollmentError` says what to fix. `saysafe.voice.PROMPTS` is a tested set of prompts.

Profiles are biometric data. Store them encrypted:

```python
store = ProfileStore("profiles", keyring_key())  # key in the OS keychain
store.save(profile)
profile = store.load("owner")
```

## Score

```python
score = vid.score(reply_audio, profile)  # cosine to the owner, or None with too little speech
step = approvals.reply(step, transcript, voice_score=score)
```

Replies can be short ("yes" is ~0.4 s), so scoring needs only 0.35 s of speech by default; shorter clips return `None` and the approval steps up to the phone. `vid.verify(audio, profile, thresholds)` also returns the band and the amount of speech. Call `vid.warm()` at startup so the first score isn't slow.

## Who's listening

```python
tracker = vid.tracker(profile)
tracker.mic_on()
for segment in vad_segments:          # every speech segment, commands or not
    tracker.observe_segment(segment)  # scored, labelled owner / other / unclear, then dropped
guard.check(reply, room=tracker.room())
```

Another voice in the last 45 s means `OTHERS_PRESENT`; under 45 s of listening means `UNKNOWN`; only the owner for 45 s means `ALONE`. Only `(time, label, score)` is kept, for 60 s, and no audio. `saysafe.voice.StreamingVAD` turns a mic stream into segments.

## Thresholds

The packaged thresholds (accept at 0.45, reject under 0.25) are placeholders from a handful of clips. Calibrate on your own data:

```python
from saysafe import ApprovalGuard, calibrate

thresholds = calibrate(owner_scores, other_scores)  # held-out owner clips vs other people
approvals = ApprovalGuard(secret, thresholds=thresholds)
```

`t_accept` accepts at most 1% of the other voices; `t_reject` rejects at most 1% of the owner's clips. Between them is the "uncertain" band, which steps up to the phone. Use a few hundred scores of each, recorded on the mic you ship, and never the clips you enrolled with.

## Limits

A voice score is evidence, not proof. Recordings of the owner score like the owner, and so can a live voice clone. The one-time word stops recordings; nothing in saysafe stops a live clone yet. Use `phone_tap` for anything you can't afford to lose. See the [threat model](threat-model.md).
