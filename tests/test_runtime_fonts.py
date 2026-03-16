from __future__ import annotations

import matplotlib

from fraclab_sdk.runtime.fonts import (
    configure_matplotlib_runtime_fonts,
    get_available_runtime_cjk_font_families,
)


def test_get_available_runtime_cjk_font_families_filters_supported_names(monkeypatch):
    monkeypatch.setattr(
        "fraclab_sdk.runtime.fonts._get_visible_matplotlib_font_names",
        lambda: {"WenQuanYi Micro Hei", "DejaVu Sans"},
    )

    assert get_available_runtime_cjk_font_families() == ("WenQuanYi Micro Hei",)


def test_configure_matplotlib_runtime_fonts_prepends_runtime_fonts(monkeypatch):
    monkeypatch.setattr(
        "fraclab_sdk.runtime.fonts.get_available_runtime_cjk_font_families",
        lambda: ("WenQuanYi Micro Hei", "Noto Serif CJK JP"),
    )

    with matplotlib.rc_context():
        matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
        matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
        matplotlib.rcParams["font.serif"] = ["DejaVu Serif"]
        matplotlib.rcParams["axes.unicode_minus"] = True

        configured = configure_matplotlib_runtime_fonts()

        assert configured == ("WenQuanYi Micro Hei", "Noto Serif CJK JP")
        assert matplotlib.rcParams["font.family"] == ["sans-serif", "DejaVu Sans"]
        assert matplotlib.rcParams["font.sans-serif"][:2] == [
            "WenQuanYi Micro Hei",
            "Noto Serif CJK JP",
        ]
        assert matplotlib.rcParams["font.serif"][0] == "Noto Serif CJK JP"
        assert matplotlib.rcParams["axes.unicode_minus"] is False
