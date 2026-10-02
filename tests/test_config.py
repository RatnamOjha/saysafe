from saysafe import config


def test_empty_env_counts_as_unset(monkeypatch):
    monkeypatch.setenv("EARSHOT_TEST_VAR", "")
    assert config.env("EARSHOT_TEST_VAR", "fallback") == "fallback"


def test_every_config_yaml_loads():
    for path in config.CONFIG_DIR.glob("*.yaml"):
        assert isinstance(config.load_yaml(path.stem), dict)
