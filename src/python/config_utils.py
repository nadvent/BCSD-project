from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def find_default_config_path() -> Path | None:
    """Finds the default config path relative to the current working directory or package root."""
    candidate_paths = [
        Path("config/default.yaml"),
        Path(__file__).resolve().parent.parent.parent / "config" / "default.yaml",
    ]
    for path in candidate_paths:
        if path.exists():
            return path
    return None


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Loads a YAML configuration file.

    If config_path is None, attempts to locate the default 'config/default.yaml'.
    Returns an empty dictionary if no configuration file is found.
    """
    path: Path | None = None
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    else:
        path = find_default_config_path()

    if path is None or not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        return {}

    return data
