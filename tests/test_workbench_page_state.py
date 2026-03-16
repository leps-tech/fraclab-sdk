from pathlib import Path

from fraclab_sdk.config import SDKConfig
from fraclab_sdk.workbench.page_state import (
    choose_valid_option,
    read_global_setting,
    read_page_state,
    write_global_setting,
    write_page_state,
)


def test_page_state_round_trip(tmp_path: Path) -> None:
    config = SDKConfig(tmp_path)

    write_page_state("browse", {"snapshot_id": "snap-1", "dataset_key": "ds-a"}, config)

    assert read_page_state("browse", config) == {"snapshot_id": "snap-1", "dataset_key": "ds-a"}


def test_page_state_ignores_invalid_json(tmp_path: Path) -> None:
    config = SDKConfig(tmp_path)
    state_path = config.sdk_home / "workbench" / "ui_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("{not-json}", encoding="utf-8")

    assert read_page_state("selection", config) == {}


def test_global_setting_round_trip(tmp_path: Path) -> None:
    config = SDKConfig(tmp_path)

    write_global_setting("language", "zh-CN", config)

    assert read_global_setting("language", config) == "zh-CN"


def test_choose_valid_option_uses_saved_value_when_available() -> None:
    assert choose_valid_option(["a", "b"], "b", "a") == "b"
    assert choose_valid_option(["a", "b"], "missing", "a") == "a"
    assert choose_valid_option(["a", "b"], "missing", "also-missing") == "a"
