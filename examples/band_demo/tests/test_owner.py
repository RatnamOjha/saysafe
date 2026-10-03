import keyring
from band_demo.owner import load_owner_scorer
from keyring.backends.fail import Keyring as NoKeyring


def test_no_keychain_means_no_owner(monkeypatch):
    """Fails closed: with no usable profile there's no owner, so voice approvals step up."""
    monkeypatch.delenv("EARSHOT_PROFILE_KEY", raising=False)
    monkeypatch.setattr(keyring, "get_keyring", lambda: NoKeyring())
    monkeypatch.setattr(keyring.core, "_keyring_backend", NoKeyring())
    assert load_owner_scorer() is None
