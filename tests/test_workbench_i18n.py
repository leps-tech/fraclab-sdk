from fraclab_sdk.workbench import i18n


def test_tx_preserves_placeholders_when_no_kwargs(monkeypatch) -> None:
    monkeypatch.setattr(i18n, "get_language", lambda: "en")

    assert (
        i18n.tx(
            "Required by schema: all {count} item(s) must be selected.",
            "Schema 要求：必须选择全部 {count} 个条目。",
        )
        == "Required by schema: all {count} item(s) must be selected."
    )


def test_tx_preserves_missing_placeholders_when_partially_formatted(monkeypatch) -> None:
    monkeypatch.setattr(i18n, "get_language", lambda: "zh-CN")

    assert (
        i18n.tx(
            "Current: {count}, max {max}.",
            "当前：{count}，最大 {max}。",
            count=3,
        )
        == "当前：3，最大 {max}。"
    )


def test_get_language_prefers_toolbar_selection(monkeypatch) -> None:
    monkeypatch.setattr(i18n.st, "session_state", {i18n._LANGUAGE_SELECT_KEY: "zh-CN"})  # type: ignore[attr-defined]
    writes: list[str] = []
    monkeypatch.setattr(i18n, "write_global_setting", lambda key, value, config=None: writes.append(value))

    assert i18n.get_language() == "zh-CN"
    assert i18n.st.session_state[i18n._LANGUAGE_SESSION_KEY] == "zh-CN"  # type: ignore[attr-defined]
    assert writes == ["zh-CN"]


def test_default_language_is_chinese_without_override(monkeypatch) -> None:
    monkeypatch.delenv("FRACLAB_WORKBENCH_LANG", raising=False)

    assert i18n._default_language() == "zh-CN"


def test_get_language_reads_persisted_language_on_fresh_session(monkeypatch) -> None:
    monkeypatch.setattr(i18n.st, "session_state", {})  # type: ignore[attr-defined]
    monkeypatch.setattr(i18n, "read_global_setting", lambda key, config=None: "en")
    writes: list[str] = []
    monkeypatch.setattr(i18n, "write_global_setting", lambda key, value, config=None: writes.append(value))

    assert i18n.get_language() == "en"
    assert i18n.st.session_state[i18n._LANGUAGE_SELECT_KEY] == "en"  # type: ignore[attr-defined]
    assert writes == []


def test_language_option_labels_are_fixed() -> None:
    assert i18n._language_option_label("en") == "English"  # type: ignore[attr-defined]
    assert i18n._language_option_label("zh-CN") == "中文"  # type: ignore[attr-defined]


def test_page_titles_use_refined_chinese_labels(monkeypatch) -> None:
    monkeypatch.setattr(i18n, "get_language", lambda: "zh-CN")

    assert i18n.page_title("schema_edit") == "输入参数编辑"
    assert i18n.page_title("output_edit") == "输出结果定义"
    assert i18n.page_title("selection") == "运行配置"
