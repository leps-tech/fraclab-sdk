"""CLI entrypoint for launching the Streamlit workbench."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence


def _require_optional_deps() -> None:
    """Fail fast with a clear message when optional deps are missing."""
    missing = []
    try:
        import streamlit  # noqa: F401
    except ImportError:
        missing.append("streamlit>=1.30")
    try:
        import pandas  # noqa: F401
    except ImportError:
        missing.append("pandas")

    if missing:
        deps = ", ".join(missing)
        raise SystemExit(
            f"Workbench dependencies missing ({deps}). "
            "Install with `pip install fraclab-sdk[workbench]`."
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Launch the Streamlit app using the bundled Home.py."""
    _require_optional_deps()
    from streamlit.web import cli as stcli

    workbench_dir = Path(__file__).resolve().parent
    home_path = workbench_dir / "Home.py"
    if not home_path.exists():
        raise SystemExit(f"Workbench entry script not found: {home_path}")

    extra_args = list(argv) if argv is not None else sys.argv[1:]
    sys.argv = ["streamlit", "run", str(home_path), *extra_args]
    os.chdir(workbench_dir)
    stcli.main()


if __name__ == "__main__":
    main()
