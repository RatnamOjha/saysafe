import numpy as np
import pytest
from cryptography.fernet import Fernet

from earshot.identity import enroll
from earshot.identity.embed import Embedding, TooShort, cosine, normalize
from earshot.identity.profile_store import Profile, ProfileError, ProfileStore
from earshot.identity.verify import Thresholds, band, verify

T = Thresholds(t_accept=0.45, t_reject=0.25, source="test")


def _vec(seed: int) -> np.ndarray:
    return normalize(np.random.default_rng(seed).standard_normal(192))


def _profile(vec=None) -> Profile:
    vec = _vec(0) if vec is None else vec
    return Profile(
        name="alex",
        mean_embedding=vec.tolist(),
        embeddings=[vec.tolist()],
        median_rms_dbfs=-25.0,
        mic_name="test mic",
    )


# profile store


def test_profile_roundtrip(tmp_path):
    store = ProfileStore(tmp_path, Fernet.generate_key())
    store.save(_profile())
    loaded = store.load("alex")
    assert np.allclose(loaded.mean, _profile().mean)
    raw = (tmp_path / "alex.profile").read_bytes()
    assert b"mean_embedding" not in raw  # encrypted at rest
    assert (tmp_path / "alex.profile").stat().st_mode & 0o077 == 0


def test_wrong_key_gives_clear_error(tmp_path):
    ProfileStore(tmp_path, Fernet.generate_key()).save(_profile())
    with pytest.raises(ProfileError, match="wrong EARSHOT_PROFILE_KEY"):
        ProfileStore(tmp_path, Fernet.generate_key()).load("alex")


def test_invalid_key_and_bad_name(tmp_path):
    with pytest.raises(ProfileError):
        ProfileStore(tmp_path, b"not-a-key")
    store = ProfileStore(tmp_path, Fernet.generate_key())
    with pytest.raises(ProfileError):
        store.load("../etc/passwd")
    with pytest.raises(ProfileError, match="No profile"):
        store.load("nobody")


def test_repr_never_shows_embeddings():
    text = repr(_profile()) + str(_profile())
    assert "mean_embedding" not in text and "embeddings" not in text
    assert "speech_seconds" in repr(Embedding(_vec(1), 2.0, 5.0))


# verify bands


@pytest.mark.parametrize(
    "score, expected",
    [
        (0.45, "accept"),
        (0.4499, "uncertain"),
        (0.25, "uncertain"),
        (0.2499, "reject"),
        (0.9, "accept"),
        (-0.1, "reject"),
    ],
)
def test_band_at_exact_thresholds(score, expected):
    assert band(score, T) == expected


def test_verify_owner_and_stranger():
    owner = _vec(0)
    same = lambda audio: Embedding(owner, 2.0, 1.0)  # noqa: E731
    other = lambda audio: Embedding(_vec(99), 2.0, 1.0)  # noqa: E731
    assert verify(np.zeros(1), _profile(owner), same, T).band == "accept"
    assert verify(np.zeros(1), _profile(owner), other, T).band == "reject"


def test_too_short_is_uncertain_without_score():
    def short(audio):
        raise TooShort(0.3, 0.8)

    r = verify(np.zeros(1), _profile(), short, T)
    assert r.too_short and r.band == "uncertain" and r.speech_seconds == 0.3


# enrollment


def test_outliers_flag_the_odd_clip():
    base = _vec(0)
    near = [normalize(base + 0.1 * _vec(i)) for i in range(1, 8)]
    assert enroll.outliers([*near, _vec(42)], 0.5) == [7]


def test_run_enrollment_rerecords_short_and_outlier_clips():
    base = _vec(0)
    calls = {"n": 0}

    def embedder(audio):
        calls["n"] += 1
        if calls["n"] == 2:
            raise TooShort(0.4, 0.8)  # clip 2, first try
        if calls["n"] == 4:
            return Embedding(_vec(42), 2.0, 1.0)  # clip 3 is someone else
        return Embedding(normalize(base + 0.1 * _vec(calls["n"])), 2.0, 1.0)

    record = lambda s: np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1  # noqa: E731
    shown = []
    p = enroll.run_enrollment("alex", record, "test mic", shown.append, embedder)
    assert len(p.embeddings) == len(enroll.PROMPTS)
    assert calls["n"] == len(enroll.PROMPTS) + 2  # one retry for short, one for outlier
    assert cosine(p.mean, base) > 0.9
    assert any("Too short" in s for s in shown) and any("Re-recording" in s for s in shown)


# real model


@pytest.mark.models
def test_embed_real_speech(piper_speech):
    from earshot.identity.embed import embed

    e = embed(piper_speech)
    assert e.vector.shape == (192,) and abs(np.linalg.norm(e.vector) - 1) < 1e-4
    assert cosine(e.vector, embed(piper_speech[: len(piper_speech) // 2 + 16000]).vector) > 0.7
    with pytest.raises(TooShort):
        embed(piper_speech[:8000])
