from importlib import resources

from saysafe import config


def test_every_packaged_yaml_loads():
    names = [
        p.name[:-5] for p in resources.files("saysafe.data").iterdir() if p.name.endswith(".yaml")
    ]
    assert {"policy", "sensitivity", "routing", "thresholds", "voice"} <= set(names)
    for name in names:
        assert isinstance(config.load_yaml(name), dict)


def test_default_config_is_an_editable_copy():
    mine = config.default_config("policy")
    mine["tokens"]["ttl_s"] = 1
    assert config.load_yaml("policy")["tokens"]["ttl_s"] != 1


def test_challenge_words_ship_with_the_package():
    assert len(config.data_text("challenge_words.txt").split()) > 200
