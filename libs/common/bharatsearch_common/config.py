"""Configuration: environment variables plus config/default.yaml.

Env values override YAML values. Section 0, rule 3: no hidden constants.
"""

from __future__ import annotations

import os
from typing import Any

import yaml

_DEFAULT_PATH = "config/default.yaml"


def load_config(path: str = _DEFAULT_PATH) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get(key: str, default: Any = None) -> Any:
    env_value = os.environ.get(key.upper())
    if env_value is not None:
        return env_value
    return load_config().get(key, default)
