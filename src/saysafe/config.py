"""Default settings ship inside the package (saysafe/data/*.yaml).

saysafe never reads environment variables or files outside the package. To change a
setting, pass your own value to the class or function that uses it.
"""

import copy
from functools import lru_cache
from importlib import resources

import yaml


@lru_cache
def load_yaml(name: str) -> dict:
    """The packaged defaults for `name` (policy, sensitivity, routing, thresholds, voice).
    Cached and shared: treat it as read-only. Use default_config() for a copy to edit."""
    text = resources.files("saysafe.data").joinpath(f"{name}.yaml").read_text()
    return yaml.safe_load(text) or {}


def default_config(name: str) -> dict:
    """An editable copy of the packaged defaults, e.g. to start your own policy from."""
    return copy.deepcopy(load_yaml(name))


def data_text(filename: str) -> str:
    """A packaged data file as text (e.g. challenge_words.txt)."""
    return resources.files("saysafe.data").joinpath(filename).read_text()
