"""Runtime font defaults aligned with the sandbox execution environment."""

from __future__ import annotations

from collections.abc import Iterable

RUNTIME_CJK_FONT_FAMILIES = (
    "WenQuanYi Micro Hei",
    "Noto Sans CJK JP",
    "Noto Serif CJK JP",
)

RUNTIME_CJK_FONT_PACKAGES = (
    "fonts-wqy-microhei",
    "fonts-noto-cjk",
)


def _dedupe_font_names(values: Iterable[object]) -> list[str]:
    """Return non-empty font names in first-seen order."""
    deduped: list[str] = []
    seen: set[str] = set()

    for value in values:
        if value is None:
            continue
        font_name = str(value).strip()
        if not font_name or font_name in seen:
            continue
        seen.add(font_name)
        deduped.append(font_name)

    return deduped


def _coerce_rc_font_list(value: object) -> list[str]:
    """Normalize a matplotlib rcParams font entry to a string list."""
    if value is None:
        return []
    if isinstance(value, str):
        return _dedupe_font_names([value])
    return _dedupe_font_names(value if isinstance(value, Iterable) else [value])


def _get_visible_matplotlib_font_names() -> set[str]:
    """Return font family names visible to matplotlib in the current process."""
    try:
        from matplotlib import font_manager
    except ImportError:
        return set()

    return {
        font.name
        for font in font_manager.fontManager.ttflist
        if getattr(font, "name", None)
    }


def get_available_runtime_cjk_font_families() -> tuple[str, ...]:
    """Return sandbox-aligned CJK font families currently visible to matplotlib."""
    visible_fonts = _get_visible_matplotlib_font_names()
    return tuple(name for name in RUNTIME_CJK_FONT_FAMILIES if name in visible_fonts)


def configure_matplotlib_runtime_fonts() -> tuple[str, ...]:
    """Prepend sandbox-aligned CJK fonts to matplotlib defaults.

    The runner calls this before loading user code so algorithms that use
    matplotlib get Chinese-capable defaults without each algorithm repeating
    the same rcParams boilerplate.
    """
    try:
        from matplotlib import rcParams
    except ImportError:
        return ()

    available_fonts = get_available_runtime_cjk_font_families()
    sans_serif_fonts = _dedupe_font_names(
        [*available_fonts, *_coerce_rc_font_list(rcParams.get("font.sans-serif"))]
    )
    serif_fonts = _dedupe_font_names(
        [
            *(name for name in available_fonts if "Serif" in name),
            *_coerce_rc_font_list(rcParams.get("font.serif")),
        ]
    )
    font_family = _coerce_rc_font_list(rcParams.get("font.family"))

    if not font_family:
        font_family = ["sans-serif"]
    elif "sans-serif" not in font_family:
        font_family = ["sans-serif", *font_family]

    rcParams["font.family"] = font_family
    rcParams["font.sans-serif"] = sans_serif_fonts
    if serif_fonts:
        rcParams["font.serif"] = serif_fonts
    rcParams["axes.unicode_minus"] = False

    return available_fonts


__all__ = [
    "RUNTIME_CJK_FONT_FAMILIES",
    "RUNTIME_CJK_FONT_PACKAGES",
    "configure_matplotlib_runtime_fonts",
    "get_available_runtime_cjk_font_families",
]
