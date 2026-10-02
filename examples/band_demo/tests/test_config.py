from band_demo import config


def test_empty_env_counts_as_unset(monkeypatch):
    monkeypatch.setenv("EARSHOT_TEST_VAR", "")
    assert config.env("EARSHOT_TEST_VAR", "fallback") == "fallback"


def test_demo_yaml_loads():
    assert {"stt", "tts"} <= set(config.demo_config())
