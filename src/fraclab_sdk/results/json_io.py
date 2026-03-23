"""JSON file helpers for result artifacts."""

import json
from pathlib import Path
from typing import Any

_JSON_TEXT_FALLBACK_ENCODINGS = (
    "utf-8-sig",
    "gb18030",
    "cp1252",
)


def load_json_file(path: Path) -> Any:
    """Load JSON from disk with robust decoding fallbacks.

    The standard library can parse UTF-8/16/32 directly from bytes. Some result
    artifacts are produced with legacy encodings, so we retry a small set of
    explicit fallbacks before surfacing the decode error.
    """
    raw = path.read_bytes()

    try:
        return json.loads(raw)
    except UnicodeDecodeError as first_error:
        last_error = first_error
    except json.JSONDecodeError:
        raise

    for encoding in _JSON_TEXT_FALLBACK_ENCODINGS:
        try:
            return json.loads(raw.decode(encoding))
        except UnicodeDecodeError as exc:
            last_error = exc
        except json.JSONDecodeError:
            raise

    raise last_error
