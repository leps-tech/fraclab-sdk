"""Algorithm workspace scaffold utilities."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from fraclab_sdk.version import __version__ as SDK_VERSION

BASE_SCHEMA_UTILS = '''"""Schema base utilities for json_schema_extra helpers + shared UI models.

Keep this module stable & generic:
- No algorithm-specific dataset keys
- No algorithm-specific constraints
- Only reusable helpers/types
"""

from __future__ import annotations

from typing import Any, Annotated, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.json_schema import WithJsonSchema


# -----------------------------
# show_when helpers
# -----------------------------
def show_when_condition(field: str, op: str = "equals", value: Any = True) -> dict[str, Any]:
    return {"field": field, "op": op, "value": value}


def show_when_and(*conditions: dict[str, Any]) -> dict[str, Any]:
    return {"and": list(conditions)}


def show_when_or(*conditions: dict[str, Any]) -> dict[str, Any]:
    return {"or": list(conditions)}


# -----------------------------
# json_schema_extra helper
# -----------------------------
def schema_extra(
    *,
    group: str | None = None,
    order: int | None = None,
    unit: str | None = None,
    step: float | None = None,
    ui_type: str | None = None,
    collapsible: bool | None = None,
    show_when: dict[str, Any] | None = None,
    enum_labels: dict[str, str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if group is not None:
        result["group"] = group
    if order is not None:
        result["order"] = order
    if unit is not None:
        result["unit"] = unit
    if step is not None:
        result["step"] = step
    if ui_type is not None:
        result["ui_type"] = ui_type
    if collapsible is not None:
        result["collapsible"] = collapsible
    if show_when is not None:
        result["show_when"] = show_when
    if enum_labels is not None:
        result["enum_labels"] = enum_labels
    result.update(kwargs)
    return result


# -----------------------------
# Generic: optional bool with titles for anyOf branches
# (prevents FIELD_MISSING_TITLE for Optional[bool])
# -----------------------------
def opt_bool(title: str) -> Any:
    return Annotated[
        Optional[bool],
        WithJsonSchema(
            {
                "anyOf": [
                    {"type": "boolean", "title": title},
                    {"type": "null", "title": title},
                ],
                "default": None,
                "title": title,
            }
        ),
    ]


# -----------------------------
# Shared: time window models
# -----------------------------
class TimeWindow(BaseModel):
    """A single picked window on a curve."""
    itemKey: Optional[str] = Field(default=None, title="Item Key")
    min: float = Field(title="Start")
    max: float = Field(title="End")

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_order(self) -> "TimeWindow":
        if self.max < self.min:
            raise ValueError("TimeWindow.max must be >= min")
        if self.itemKey is not None and not str(self.itemKey).strip():
            raise ValueError("TimeWindow.itemKey cannot be empty when provided")
        return self


def time_window_list(
    *,
    min_items: int = 1,
    max_items: int | None = None,
    title: str = "Windows",
    description: str = "List of time windows.",
) -> Any:
    """Return Annotated[Optional[list[TimeWindow]], Field(min_length/max_length...)].

    Use in schema files to define datasetKey-specific templates, e.g. 1..3 or exactly 2.
    The type is Optional so the field can default to None (no windows selected yet).
    """
    kwargs: dict[str, Any] = {"min_length": min_items, "title": title, "description": description}
    if max_items is not None:
        kwargs["max_length"] = max_items
    return Annotated[Optional[list[TimeWindow]], Field(**kwargs)]
'''


def ensure_schema_base(schema_dir: Path) -> Path:
    """Ensure schema/base.py exists for algorithm schema helpers."""
    schema_dir.mkdir(parents=True, exist_ok=True)
    init_path = schema_dir / "__init__.py"
    if not init_path.exists():
        init_path.write_text("", encoding="utf-8")

    base_path = schema_dir / "base.py"
    base_path.write_text(BASE_SCHEMA_UTILS, encoding="utf-8")
    return base_path


def create_algorithm_scaffold(
    algo_id: str,
    code_version: str,
    contract_version: str,
    name: str,
    summary: str,
    authors: list[dict[str, str]],
    notes: str | None = None,
    tags: list[str] | None = None,
    *,
    workspace_root: Path,
) -> Path:
    """Create a new algorithm workspace with minimal files."""
    ws_dir = workspace_root / algo_id / code_version
    if ws_dir.exists():
        raise FileExistsError(f"Algorithm workspace already exists: {ws_dir}")

    parent_dir = ws_dir.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(
        tempfile.mkdtemp(
            prefix=f".tmp-{algo_id}-{code_version}-",
            dir=str(parent_dir),
        )
    )

    authors_list = [
        {
            "name": (a.get("name") or "").strip(),
            "email": (a.get("email") or "").strip(),
            "organization": (a.get("organization") or "").strip(),
        }
        for a in authors
    ]
    authors_list = [a for a in authors_list if any(v for v in a.values())] or [{"name": "unknown"}]

    summary_val = summary.strip() or f"Algorithm {algo_id}"
    manifest = {
        "manifestVersion": "1",
        "algorithmId": algo_id,
        "name": name or algo_id,
        "summary": summary_val,
        "authors": authors_list,
        "contractVersion": contract_version,
        "codeVersion": code_version,
        "notes": notes or None,
        "tags": tags or None,
        "files": {
            "paramsSchemaPath": "dist/params.schema.json",
            "dsPath": "dist/ds.json",
            "outputContractPath": "dist/output_contract.json",
        },
        "requires": {"sdk": SDK_VERSION},
        "repository": None,
        "homepage": None,
        "license": None,
    }

    try:
        (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        dist_dir = tmp_dir / "dist"
        dist_dir.mkdir(parents=True, exist_ok=True)
        (dist_dir / "ds.json").write_text(json.dumps({"datasets": []}, indent=2), encoding="utf-8")
        (dist_dir / "params.schema.json").write_text(
            json.dumps({"type": "object", "title": "Parameters", "properties": {}}, indent=2),
            encoding="utf-8",
        )
        (dist_dir / "output_contract.json").write_text(
            json.dumps({"datasets": [], "invariants": [], "relations": []}, indent=2),
            encoding="utf-8",
        )

        main_stub = '''"""Algorithm entrypoint."""

from __future__ import annotations

def run(ctx) -> None:
    """Implement algorithm logic here."""
    # TODO: replace with real logic
    ctx.logger.info("algorithm scaffold run")
'''
        (tmp_dir / "main.py").write_text(main_stub, encoding="utf-8")

        ensure_schema_base(tmp_dir / "schema")

        tmp_dir.replace(ws_dir)
        return ws_dir
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
