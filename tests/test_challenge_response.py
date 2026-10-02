import itertools

import jellyfish
import pytest
from rapidfuzz.distance import Levenshtein

from saysafe.approvals.challenge import Challenge, ChallengeIssuer, words
from saysafe.approvals.response import match_response

NOW = 1000.0


def ch(word="maple", issued=NOW - 5, ttl=30, used=False) -> Challenge:
    return Challenge(word, "a1", issued, issued + ttl, used)


# word list


def test_word_list_is_big_and_distinct():
    w = words()
    assert len(w) >= 200 and len(set(w)) == len(w)
    assert all(x.isalpha() and x.islower() for x in w)
    codes = [jellyfish.metaphone(x) for x in w]
    assert len(set(codes)) == len(codes), "two words sound alike"
    close = [(a, b) for a, b in itertools.combinations(w, 2) if Levenshtein.distance(a, b) <= 2]
    assert close == []


@pytest.mark.parametrize(
    "common", "yes yeah yep sure confirm please okay no nope stop cancel wait maybe the and "
              "correct right sounds good go ahead do it".split(),
)  # fmt: skip
def test_common_reply_words_never_look_like_challenge_words(common):
    assert match_response(common, "voice_challenge", ch("zebra"), NOW).kind not in (
        "challenge_ok", "challenge_wrong",
    )  # fmt: skip


# issuer


def test_issue_binds_expires_and_avoids_repeats():
    t = {"now": NOW}
    issuer = ChallengeIssuer(clock=lambda: t["now"])
    c = issuer.issue("act-1")
    assert c.action_id == "act-1" and c.expires_at - c.issued_at == 30
    assert issuer.by_action["act-1"] is c
    assert c.status(NOW + 29.9) == "valid" and c.status(NOW + 30) == "expired"
    issued = [issuer.issue(f"a{i}").word for i in range(50)]
    assert len(set(issued)) == 50  # no repeats within the recent window
    issuer.consume(c)
    assert c.status(NOW) == "used"


# responses: 30+ strings


@pytest.mark.parametrize(
    "said, tier, expected",
    [
        # plain affirmatives
        ("yes", "voice", "affirm"),
        ("Yes.", "voice", "affirm"),
        ("yeah", "voice", "affirm"),
        ("yep", "voice", "affirm"),
        ("sure", "voice", "affirm"),
        ("confirm", "voice", "affirm"),
        ("do it", "voice", "affirm"),
        ("go ahead", "voice", "affirm"),
        ("sounds good", "voice", "affirm"),
        ("yes please", "voice", "affirm"),
        ("uh, yes", "voice", "affirm"),
        ("um yeah do it", "voice", "affirm"),
        ("okay so, go ahead", "voice", "affirm"),
        ("Okay, yes.", "voice", "affirm"),
        # negation wins
        ("no", "voice", "negate"),
        ("yes, no wait", "voice", "negate"),
        ("yeah no", "voice", "negate"),
        ("yeah don't", "voice", "negate"),
        ("yeah, do not", "voice", "negate"),
        ("cancel", "voice", "negate"),
        ("stop", "voice", "negate"),
        ("hold on", "voice", "negate"),
        ("wait wait", "voice", "negate"),
        ("never mind", "voice", "negate"),
        ("nope", "voice", "negate"),
        ("maple, no", "voice_challenge", "negate"),
        # unclear
        ("", "voice", "unclear"),
        ("uh", "voice", "unclear"),
        ("what's the weather", "voice", "unclear"),
        ("okay", "voice", "unclear"),
        # challenge tier
        ("maple", "voice_challenge", "challenge_ok"),
        ("Maple.", "voice_challenge", "challenge_ok"),
        ("uh, maple", "voice_challenge", "challenge_ok"),
        ("maple please", "voice_challenge", "challenge_ok"),
        ("mapel", "voice_challenge", "challenge_ok"),       # one edit
        ("maypole", "voice_challenge", "challenge_ok"),     # same metaphone as maple
        ("copper", "voice_challenge", "challenge_wrong"),   # an old / different word
        ("yes copper", "voice_challenge", "challenge_wrong"),
        ("yes", "voice_challenge", "affirm"),               # a replayed yes is not enough
        ("banana", "voice_challenge", "unclear"),
    ],
)  # fmt: skip
def test_match_response(said, tier, expected):
    assert match_response(said, tier, ch() if tier == "voice_challenge" else None, NOW).kind == (
        expected
    )


def test_expired_challenge_is_wrong():
    m = match_response("maple", "voice_challenge", ch(issued=NOW - 31), NOW)
    assert m.kind == "challenge_wrong" and "expired" in m.detail


def test_used_challenge_is_wrong():
    m = match_response("maple", "voice_challenge", ch(used=True), NOW)
    assert m.kind == "challenge_wrong" and "used" in m.detail


def test_split_compound_word():
    assert match_response("sun flower", "voice_challenge", ch("sunflower"), NOW).kind == (
        "challenge_ok"
    )
