"""Persistent page state helpers for workbench UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fraclab_sdk.config import SDKConfig

_STATE_DIRNAME = "workbench"
_STATE_FILENAME = "ui_state.json"
_GLOBAL_STATE_KEY = "_global"


def _state_path(config: SDKConfig | None = None) -> Path:
    """Return the persistent UI state file path."""
    cfg = config or SDKConfig()
    return cfg.sdk_home / _STATE_DIRNAME / _STATE_FILENAME


def _load_all_state(config: SDKConfig | None = None) -> dict[str, Any]:
    """Load all persisted page state, ignoring malformed files."""
    path = _state_path(config)
    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    return raw if isinstance(raw, dict) else {}


def _write_all_state(state: dict[str, Any], config: SDKConfig | None = None) -> None:
    """Atomically write the full workbench UI state file."""
    path = _state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def read_page_state(page_key: str, config: SDKConfig | None = None) -> dict[str, Any]:
    """Read the persisted state for one workbench page."""
    state = _load_all_state(config)
    page_state = state.get(page_key)
    return page_state if isinstance(page_state, dict) else {}


def write_page_state(page_key: str, page_state: dict[str, Any], config: SDKConfig | None = None) -> None:
    """Persist the state for one workbench page."""
    state = _load_all_state(config)
    state[page_key] = page_state
    _write_all_state(state, config)


def read_global_setting(setting_key: str, config: SDKConfig | None = None) -> Any | None:
    """Read one persisted global workbench setting."""
    state = _load_all_state(config)
    global_state = state.get(_GLOBAL_STATE_KEY)
    if not isinstance(global_state, dict):
        return None
    return global_state.get(setting_key)


def write_global_setting(setting_key: str, value: Any, config: SDKConfig | None = None) -> None:
    """Persist one global workbench setting."""
    state = _load_all_state(config)
    global_state = dict(state.get(_GLOBAL_STATE_KEY) or {})
    global_state[setting_key] = value
    state[_GLOBAL_STATE_KEY] = global_state
    _write_all_state(state, config)


def choose_valid_option(options: list[Any], saved_value: Any, fallback: Any | None = None) -> Any | None:
    """Return a saved option when still valid, otherwise use a fallback or first option."""
    if saved_value in options:
        return saved_value
    if fallback in options:
        return fallback
    return options[0] if options else None


__all__ = [
    "choose_valid_option",
    "read_global_setting",
    "read_page_state",
    "write_global_setting",
    "write_page_state",
]
