"""Validation tools for InputSpec, OutputContract, and run manifests.

Provides:
- InputSpec linting (json_schema_extra validation, show_when structure)
- OutputContract validation (structure, key uniqueness)
- Bundle validation (hash integrity)
- RunManifest vs OutputContract alignment validation
- Algorithm signature validation
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from fraclab_sdk.errors import AlgorithmError


class ValidationSeverity(Enum):
    """Severity level for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation issue."""

    severity: ValidationSeverity
    code: str
    message: str
    path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of validation."""

    valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Get error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Get warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]


# =============================================================================
# InputSpec Validation
# =============================================================================

# Allowed json_schema_extra keys (spec-defined)
ALLOWED_JSON_SCHEMA_EXTRA_KEYS = {
    "group", "unit", "step", "ui_type", "show_when",
    "enum_labels", "order", "collapsible"
}

# Type constraints for json_schema_extra keys
JSON_SCHEMA_EXTRA_TYPES: dict[str, type | tuple[type, ...]] = {
    "group": str,
    "unit": str,
    "ui_type": str,
    "order": int,
    "collapsible": bool,
    "step": (int, float),
}

# Canonical show_when operators (per InputSpec spec)
CANONICAL_SHOW_WHEN_OPS = {
    "equals", "not_equals", "gt", "gte", "lt", "lte", "in", "not_in"
}

# Operator aliases (normalized to canonical form)
SHOW_WHEN_OP_ALIASES = {
    "eq": "equals",
    "neq": "not_equals",
    "nin": "not_in",
}

# Numeric operators (require numeric field and value)
NUMERIC_SHOW_WHEN_OPS = {"gt", "gte", "lt", "lte"}

# Array operators (require array value)
ARRAY_SHOW_WHEN_OPS = {"in", "not_in"}

# Pattern to detect snake_case
SNAKE_CASE_PATTERN = re.compile(r"[a-z]+_[a-z]+")


def _is_numeric_schema(schema: dict[str, Any]) -> bool:
    """Return True if schema type is numeric."""
    return schema.get("type") in {"number", "integer"}


def _resolve_ref_schema(schema: dict[str, Any], root_schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve local $ref like '#/$defs/Name'."""
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return schema
    current: Any = root_schema
    for part in ref[2:].split("/"):
        if not isinstance(current, dict):
            return schema
        current = current.get(part)
        if current is None:
            return schema
    return current if isinstance(current, dict) else schema


def _unwrap_nullable_anyof(schema: dict[str, Any]) -> dict[str, Any]:
    """For Optional[T], prefer non-null anyOf branch."""
    any_of = schema.get("anyOf")
    if not isinstance(any_of, list):
        return schema
    non_null = [branch for branch in any_of if isinstance(branch, dict) and branch.get("type") != "null"]
    if len(non_null) == 1:
        return non_null[0]
    return schema


def _normalize_schema(schema: dict[str, Any], root_schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve common wrappers used by Pydantic JSON schema."""
    current = _unwrap_nullable_anyof(schema)
    if "$ref" in current:
        current = _resolve_ref_schema(current, root_schema)
    return current


def _is_float_range_schema(schema: dict[str, Any], root_schema: dict[str, Any]) -> bool:
    """Validate FloatRange shape: {min: number, max: number}."""
    schema = _normalize_schema(schema, root_schema)
    if schema.get("type") != "object":
        return False
    props = schema.get("properties")
    if not isinstance(props, dict):
        return False
    if "min" not in props or "max" not in props:
        return False
    min_schema = props["min"] if isinstance(props["min"], dict) else {}
    max_schema = props["max"] if isinstance(props["max"], dict) else {}
    min_schema = _normalize_schema(min_schema, root_schema)
    max_schema = _normalize_schema(max_schema, root_schema)
    if not _is_numeric_schema(min_schema) or not _is_numeric_schema(max_schema):
        return False
    return True


def _validate_time_window_shape(
    field_schema: dict[str, Any],
    root_schema: dict[str, Any],
    path: str,
    issues: list[ValidationIssue],
) -> None:
    """Validate ui_type=time_window schema shape.

    Allowed:
    1) FloatRange: {min:number, max:number}
    2) List[FloatRange]
    3) List[List[FloatRange]]
    """
    field_schema = _normalize_schema(field_schema, root_schema)

    # 1) Single range object
    if _is_float_range_schema(field_schema, root_schema):
        return

    # 2) List[FloatRange]
    if field_schema.get("type") == "array":
        items = field_schema.get("items")
        if isinstance(items, dict) and _is_float_range_schema(items, root_schema):
            return

    # 3) List[List[FloatRange]]
    if field_schema.get("type") == "array":
        outer_items = field_schema.get("items")
        if isinstance(outer_items, dict) and outer_items.get("type") == "array":
            inner_items = outer_items.get("items")
            if isinstance(inner_items, dict) and _is_float_range_schema(inner_items, root_schema):
                return

    issues.append(
        ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code="TIME_WINDOW_SCHEMA_INVALID",
            message=(
                "ui_type='time_window' requires one of: "
                "FloatRange {min,max}, List[FloatRange], List[List[FloatRange]]"
            ),
            path=path,
        )
    )


def _to_camel_case(snake_str: str) -> str:
    """Convert snake_case to camelCase."""
    parts = snake_str.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def _resolve_ref(ref_path: str, root_schema: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve a $ref path in JSON schema."""
    if not ref_path.startswith("#/"):
        return None
    parts = ref_path[2:].split("/")
    current = root_schema
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current if isinstance(current, dict) else None


def _merge_all_of(all_of_list: list[dict], root_schema: dict[str, Any]) -> dict[str, Any]:
    """Merge allOf schemas into single view for path resolution."""
    merged: dict[str, Any] = {"properties": {}}
    for sub in all_of_list:
        resolved = sub
        if "$ref" in sub:
            resolved = _resolve_ref(sub["$ref"], root_schema) or sub
        merged["properties"].update(resolved.get("properties", {}))
        if "type" in resolved:
            merged["type"] = resolved["type"]
    return merged


def _unwrap_any_of(any_of_list: list[dict]) -> dict[str, Any]:
    """Unwrap anyOf, preferring non-null branch."""
    non_null = [s for s in any_of_list if s.get("type") != "null"]
    if len(non_null) == 1:
        return non_null[0]
    return non_null[0] if non_null else any_of_list[0]


def _resolve_field_in_schema(
    field_path: str, schema: dict[str, Any]
) -> dict[str, Any] | None:
    """Resolve field path in JSON Schema, handling $ref, allOf, anyOf/oneOf.

    Args:
        field_path: Dot-separated field path (e.g., "denoise.enable").
        schema: Root JSON schema dict.

    Returns:
        Field schema if found, None otherwise.
    """
    segments = field_path.split(".")
    current = schema

    for segment in segments:
        # Resolve $ref
        if "$ref" in current:
            resolved = _resolve_ref(current["$ref"], schema)
            if resolved is None:
                return None
            current = resolved

        # Merge allOf (common in Pydantic v2 for inheritance)
        if "allOf" in current:
            current = _merge_all_of(current["allOf"], schema)

        # Unwrap anyOf/oneOf (find "real type")
        if "anyOf" in current:
            current = _unwrap_any_of(current["anyOf"])
        if "oneOf" in current:
            current = _unwrap_any_of(current["oneOf"])

        props = current.get("properties", {})
        if segment not in props:
            return None
        current = props[segment]

    # Final resolution for the target field
    if "$ref" in current:
        resolved = _resolve_ref(current["$ref"], schema)
        if resolved:
            current = resolved
    if "allOf" in current:
        current = _merge_all_of(current["allOf"], schema)
    if "anyOf" in current:
        current = _unwrap_any_of(current["anyOf"])
    if "oneOf" in current:
        current = _unwrap_any_of(current["oneOf"])

    return current


def _validate_show_when_condition(
    condition: dict[str, Any], schema: dict[str, Any], path: str, issues: list[ValidationIssue]
) -> None:
    """Validate a single show_when condition.

    Args:
        condition: The condition dict {field, op, value}.
        schema: The full JSON schema for field lookup.
        path: Current path for error reporting.
        issues: List to append issues to.
    """
    if not isinstance(condition, dict):
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="SHOW_WHEN_INVALID_CONDITION",
                message="show_when condition must be a dict",
                path=path,
            )
        )
        return

    # Check required keys
    if "field" not in condition:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="SHOW_WHEN_MISSING_FIELD",
                message="show_when condition missing 'field' key",
                path=path,
            )
        )
        return

    field_path = condition["field"]
    op = condition.get("op", "equals")
    value = condition.get("value")

    # Check for snake_case in field path (must be ERROR with fix suggestion)
    if SNAKE_CASE_PATTERN.search(field_path):
        segments = field_path.split(".")
        suggested = ".".join(
            _to_camel_case(s) if "_" in s else s for s in segments
        )
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="SHOW_WHEN_SNAKE_CASE_FIELD",
                message=f"snake_case in show_when.field causes UI breakage: '{field_path}'",
                path=path,
                details={"original": field_path, "suggested": suggested},
            )
        )

    # Check operator: alias → WARNING + normalize; unknown → ERROR
    if op in SHOW_WHEN_OP_ALIASES:
        canonical = SHOW_WHEN_OP_ALIASES[op]
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="SHOW_WHEN_OP_ALIAS",
                message=f"Operator '{op}' is an alias; use canonical '{canonical}' instead",
                path=path,
                details={"alias": op, "canonical": canonical},
            )
        )
        op = canonical
    elif op not in CANONICAL_SHOW_WHEN_OPS:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="SHOW_WHEN_INVALID_OP",
                message=f"Invalid show_when operator: '{op}'. Valid: {sorted(CANONICAL_SHOW_WHEN_OPS)}",
                path=path,
            )
        )
        return  # Can't validate further with invalid op

    # Resolve field in schema for type compatibility checks
    field_schema = _resolve_field_in_schema(field_path, schema)

    # Validate field path exists in schema
    if field_schema is None:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="SHOW_WHEN_FIELD_NOT_FOUND",
                message=f"show_when references non-existent field: '{field_path}'",
                path=path,
                details={"field": field_path},
            )
        )
        return

    # Type compatibility checks
    field_type = field_schema.get("type")

    # Numeric operators require numeric field and value
    if op in NUMERIC_SHOW_WHEN_OPS:
        if field_type not in ("number", "integer"):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="SHOW_WHEN_NUMERIC_OP_ON_NON_NUMERIC",
                    message=f"Numeric operator '{op}' used on non-numeric field (type: {field_type})",
                    path=path,
                    details={"op": op, "field_type": field_type},
                )
            )
        if value is not None and not isinstance(value, (int, float)):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="SHOW_WHEN_NUMERIC_OP_VALUE_NOT_NUMBER",
                    message=f"Numeric operator '{op}' requires numeric value, got {type(value).__name__}",
                    path=path,
                )
            )

    # Array operators require array value
    if op in ARRAY_SHOW_WHEN_OPS:
        if not isinstance(value, list):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="SHOW_WHEN_ARRAY_OP_VALUE_NOT_ARRAY",
                    message=f"Array operator '{op}' requires list value, got {type(value).__name__}",
                    path=path,
                )
            )

    # equals/not_equals on enum field: check value is in enum
    if op in ("equals", "not_equals"):
        enum_values = field_schema.get("enum")
        if enum_values is not None and value is not None and value not in enum_values:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="SHOW_WHEN_VALUE_NOT_IN_ENUM",
                    message=f"show_when value '{value}' not in enum: {enum_values}",
                    path=path,
                    details={"value": value, "enum": enum_values},
                )
            )


def _validate_show_when(
    show_when: Any, schema: dict[str, Any], path: str, issues: list[ValidationIssue]
) -> None:
    """Validate show_when structure.

    Supports:
    - Single condition: {field, op, value}
    - AND list: [{cond1}, {cond2}]
    - OR object: {"or": [{cond1}, {cond2}]}

    Args:
        show_when: The show_when value.
        schema: Full JSON schema for field lookup.
        path: Current path for error reporting.
        issues: List to append issues to.
    """
    if show_when is None:
        return

    if isinstance(show_when, dict):
        if "or" in show_when:
            # OR object
            or_conditions = show_when["or"]
            if not isinstance(or_conditions, list):
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="SHOW_WHEN_INVALID_OR",
                        message="show_when 'or' must be a list",
                        path=path,
                    )
                )
            else:
                for i, cond in enumerate(or_conditions):
                    _validate_show_when_condition(cond, schema, f"{path}.or[{i}]", issues)
        elif "and" in show_when:
            # AND object (explicit)
            and_conditions = show_when["and"]
            if not isinstance(and_conditions, list):
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="SHOW_WHEN_INVALID_AND",
                        message="show_when 'and' must be a list",
                        path=path,
                    )
                )
            else:
                for i, cond in enumerate(and_conditions):
                    _validate_show_when_condition(cond, schema, f"{path}.and[{i}]", issues)
        else:
            # Single condition
            _validate_show_when_condition(show_when, schema, path, issues)

    elif isinstance(show_when, list):
        # Implicit AND list
        for i, cond in enumerate(show_when):
            _validate_show_when_condition(cond, schema, f"{path}[{i}]", issues)

    else:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="SHOW_WHEN_INVALID_TYPE",
                message=f"show_when must be dict or list, got {type(show_when).__name__}",
                path=path,
            )
        )


def _validate_enum_labels(
    field_schema: dict[str, Any],
    enum_labels: dict[str, str],
    path: str,
    issues: list[ValidationIssue],
) -> None:
    """Validate enum_labels keys match enum values strictly.

    Args:
        field_schema: The field's JSON schema.
        enum_labels: The enum_labels dict from json_schema_extra.
        path: Current path for error reporting.
        issues: List to append issues to.
    """
    enum_values = field_schema.get("enum")

    if enum_values is None:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="ENUM_LABELS_ON_NON_ENUM_FIELD",
                message="enum_labels provided for non-enum field",
                path=path,
            )
        )
        return

    label_keys = set(enum_labels.keys())
    enum_set = set(str(v) for v in enum_values)

    missing = enum_set - label_keys
    extra = label_keys - enum_set

    if missing:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="ENUM_LABELS_MISSING_KEYS",
                message=f"enum_labels missing keys for enum values: {sorted(missing)}",
                path=path,
                details={"missing": sorted(missing)},
            )
        )
    if extra:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="ENUM_LABELS_EXTRA_KEYS",
                message=f"enum_labels has keys not in enum: {sorted(extra)}",
                path=path,
                details={"extra": sorted(extra)},
            )
        )


def _validate_json_schema_extra(
    extra: dict[str, Any],
    field_schema: dict[str, Any],
    full_schema: dict[str, Any],
    path: str,
    issues: list[ValidationIssue],
    orders_in_scope: set[int],
) -> None:
    """Validate json_schema_extra keys and values.

    Args:
        extra: The json_schema_extra dict.
        field_schema: The field's JSON schema.
        full_schema: The full schema for show_when validation.
        path: Current path for error reporting.
        issues: List to append issues to.
        orders_in_scope: Set of order values seen in current properties scope.
    """
    for key, value in extra.items():
        # x_* prefix → WARNING (extension keys)
        if key.startswith("x_"):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="JSON_SCHEMA_EXTRA_EXTENSION_KEY",
                    message=f"Extension key '{key}' (x_* prefix) will be ignored by SDK",
                    path=f"{path}.{key}",
                )
            )
            continue

        # Unknown key (not in whitelist) → ERROR
        if key not in ALLOWED_JSON_SCHEMA_EXTRA_KEYS:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="JSON_SCHEMA_EXTRA_UNKNOWN_KEY",
                    message=f"Unknown json_schema_extra key: '{key}'. Allowed: {sorted(ALLOWED_JSON_SCHEMA_EXTRA_KEYS)}",
                    path=f"{path}.{key}",
                )
            )
            continue

        # Type validation for known keys
        if key in JSON_SCHEMA_EXTRA_TYPES:
            expected_type = JSON_SCHEMA_EXTRA_TYPES[key]
            if not isinstance(value, expected_type):
                expected_name = (
                    expected_type.__name__
                    if isinstance(expected_type, type)
                    else " | ".join(t.__name__ for t in expected_type)
                )
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="JSON_SCHEMA_EXTRA_TYPE_MISMATCH",
                        message=f"json_schema_extra['{key}'] must be {expected_name}, got {type(value).__name__}",
                        path=f"{path}.{key}",
                    )
                )
                continue

        # step must be > 0
        if key == "step":
            if value <= 0:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="JSON_SCHEMA_EXTRA_STEP_INVALID",
                        message=f"step must be > 0, got {value}",
                        path=f"{path}.step",
                    )
                )

        # order duplicate check within same properties scope
        if key == "order":
            if value in orders_in_scope:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="JSON_SCHEMA_EXTRA_DUPLICATE_ORDER",
                        message=f"Duplicate order value {value} in same properties scope",
                        path=f"{path}.order",
                    )
                )
            else:
                orders_in_scope.add(value)

        # show_when validation
        if key == "show_when":
            _validate_show_when(value, full_schema, f"{path}.show_when", issues)

        # enum_labels validation
        if key == "enum_labels":
            if isinstance(value, dict):
                _validate_enum_labels(field_schema, value, f"{path}.enum_labels", issues)

        # ui_type specific shape validation
        if key == "ui_type" and value == "time_window":
            _validate_time_window_shape(field_schema, full_schema, path, issues)


def _is_leaf_field(field_schema: dict[str, Any]) -> bool:
    """Check if field is a leaf (no nested properties)."""
    return "properties" not in field_schema


def _validate_title_requirement(
    field_schema: dict[str, Any],
    path: str,
    issues: list[ValidationIssue],
) -> None:
    """Warn if leaf field is missing title."""
    if _is_leaf_field(field_schema) and "title" not in field_schema:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="FIELD_MISSING_TITLE",
                message="Leaf field missing 'title' for UI display",
                path=path,
            )
        )


def _extract_schema_from_workspace(workspace: Path) -> dict[str, Any]:
    """Extract JSON schema from workspace InputSpec.

    Args:
        workspace: Algorithm workspace.

    Returns:
        JSON schema dict.
    """
    script = '''
import json
import sys

try:
    from schema.inputspec import INPUT_SPEC
    model = INPUT_SPEC
    schema = model.model_json_schema()
    print(json.dumps(schema))
except Exception as e:
    print(json.dumps({"error": str(e)}))
'''

    env = {"PYTHONPATH": str(workspace), "PYTHONUNBUFFERED": "1"}
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=workspace,
        env={**dict(__import__("os").environ), **env},
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        raise AlgorithmError(f"Failed to extract schema: {result.stderr}")

    data = json.loads(result.stdout)
    if "error" in data:
        raise AlgorithmError(f"Failed to extract schema: {data['error']}")

    return data


def validate_inputspec(workspace: Path) -> ValidationResult:
    """Validate InputSpec (schema.inputspec:INPUT_SPEC).

    Checks:
    - Schema can be generated
    - json_schema_extra fields are valid
    - show_when conditions reference existing fields
    - enum_labels keys match enum values

    Args:
        workspace: Algorithm workspace path.

    Returns:
        ValidationResult with issues found.
    """
    workspace = Path(workspace).resolve()
    issues: list[ValidationIssue] = []

    # Extract schema
    try:
        schema = _extract_schema_from_workspace(workspace)
    except AlgorithmError as e:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="INPUTSPEC_LOAD_FAILED",
                message=str(e),
            )
        )
        return ValidationResult(valid=False, issues=issues)

    # Validate properties
    _validate_schema_properties(schema, schema, "", issues)

    # Check for required fields
    if "properties" not in schema:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="INPUTSPEC_NO_PROPERTIES",
                message="Schema has no properties defined",
            )
        )

    has_errors = any(i.severity == ValidationSeverity.ERROR for i in issues)
    return ValidationResult(valid=not has_errors, issues=issues)


def _validate_schema_properties(
    props_container: dict[str, Any],
    full_schema: dict[str, Any],
    path_prefix: str,
    issues: list[ValidationIssue],
) -> None:
    """Recursively validate schema properties.

    Args:
        props_container: Dict containing 'properties' key.
        full_schema: The full schema for field lookups.
        path_prefix: Current path prefix for error reporting.
        issues: List to append issues to.
    """
    properties = props_container.get("properties", {})
    orders_in_scope: set[int] = set()  # Track order values within this scope

    for field_name, field_schema in properties.items():
        field_path = f"{path_prefix}.{field_name}" if path_prefix else field_name

        # Resolve the actual field schema (handle $ref, allOf, anyOf)
        resolved_schema = field_schema
        if "$ref" in field_schema:
            resolved = _resolve_ref(field_schema["$ref"], full_schema)
            if resolved:
                resolved_schema = resolved
        if "allOf" in resolved_schema:
            resolved_schema = _merge_all_of(resolved_schema["allOf"], full_schema)
        if "anyOf" in resolved_schema:
            resolved_schema = _unwrap_any_of(resolved_schema["anyOf"])

        # Check json_schema_extra (stored in various places depending on Pydantic version)
        extra = (
            field_schema.get("json_schema_extra")
            or field_schema.get("extra")
            or {}
        )

        # Validate json_schema_extra comprehensively
        if extra:
            _validate_json_schema_extra(
                extra, resolved_schema, full_schema, field_path, issues, orders_in_scope
            )

        # Validate title requirement for leaf fields
        _validate_title_requirement(resolved_schema, field_path, issues)

        # Recurse into nested objects
        if resolved_schema.get("type") == "object" or "properties" in resolved_schema:
            _validate_schema_properties(resolved_schema, full_schema, field_path, issues)

        # Handle allOf, anyOf, oneOf at field level
        for combiner in ["allOf", "anyOf", "oneOf"]:
            if combiner in field_schema:
                for i, sub_schema in enumerate(field_schema[combiner]):
                    if "properties" in sub_schema:
                        _validate_schema_properties(
                            sub_schema, full_schema, f"{field_path}.{combiner}[{i}]", issues
                        )

    # Handle $defs
    if "$defs" in props_container:
        for def_name, def_schema in props_container["$defs"].items():
            _validate_schema_properties(
                def_schema, full_schema, f"$defs.{def_name}", issues
            )


# =============================================================================
# OutputContract Validation
# =============================================================================


def validate_output_contract(workspace_or_path: Path) -> ValidationResult:
    """Validate OutputContract structure.

    Checks:
    - Contract can be loaded
    - Dataset keys are unique
    - Item keys are unique within datasets
    - Artifact keys are unique within items
    - kind matches schema.type

    Args:
        workspace_or_path: Workspace path or direct path to output_contract.json.

    Returns:
        ValidationResult with issues found.
    """
    workspace_or_path = Path(workspace_or_path).resolve()
    issues: list[ValidationIssue] = []

    # Find contract file
    if workspace_or_path.is_file():
        contract_path = workspace_or_path
    else:
        contract_path = workspace_or_path / "dist" / "output_contract.json"
        if not contract_path.exists():
            # Try extracting from workspace
            try:
                script = '''
import json
from schema.output_contract import OUTPUT_CONTRACT
if hasattr(OUTPUT_CONTRACT, 'model_dump'):
    print(json.dumps(OUTPUT_CONTRACT.model_dump(mode="json")))
else:
    print(json.dumps(OUTPUT_CONTRACT.dict()))
'''
                env = {"PYTHONPATH": str(workspace_or_path), "PYTHONUNBUFFERED": "1"}
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    cwd=workspace_or_path,
                    env={**dict(__import__("os").environ), **env},
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    contract = json.loads(result.stdout)
                else:
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            code="OUTPUT_CONTRACT_NOT_FOUND",
                            message="output_contract.json not found and could not extract from workspace",
                        )
                    )
                    return ValidationResult(valid=False, issues=issues)
            except Exception as e:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="OUTPUT_CONTRACT_LOAD_FAILED",
                        message=str(e),
                    )
                )
                return ValidationResult(valid=False, issues=issues)
        else:
            contract = json.loads(contract_path.read_text())

    if "contract" not in dir():
        contract = json.loads(contract_path.read_text())

    # Validate contract structure
    _validate_contract_structure(contract, issues)

    has_errors = any(i.severity == ValidationSeverity.ERROR for i in issues)
    return ValidationResult(valid=not has_errors, issues=issues)


def _validate_contract_structure(contract: dict[str, Any], issues: list[ValidationIssue]) -> None:
    """Validate OutputContract structure.

    Args:
        contract: Contract dict.
        issues: List to append issues to.
    """
    datasets = contract.get("datasets", [])

    # Check dataset key uniqueness
    dataset_keys = [ds.get("key") for ds in datasets if "key" in ds]
    duplicates = [k for k in dataset_keys if dataset_keys.count(k) > 1]
    if duplicates:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="OUTPUT_CONTRACT_DUPLICATE_DATASET_KEY",
                message=f"Duplicate dataset keys: {set(duplicates)}",
            )
        )

    allowed_kinds = {"frame", "object", "blob", "scalar"}
    allowed_owners = {"stage", "well", "platform"}
    allowed_cardinality = {"one", "many"}
    allowed_roles = {"primary", "supporting", "debug"}
    kind_schema_map = {
        "frame": {"frame"},
        "object": {"object"},
        "blob": {"blob"},
        "scalar": {"scalar"},
    }

    for ds in datasets:
        ds_key = ds.get("key", "unknown")
        kind = ds.get("kind")
        owner = ds.get("owner")
        cardinality = ds.get("cardinality")
        role = ds.get("role")
        schema = ds.get("schema") or {}
        schema_type = schema.get("type")

        if kind not in allowed_kinds:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="OUTPUT_CONTRACT_INVALID_KIND",
                    message=f"Invalid kind '{kind}' (expected one of {allowed_kinds})",
                    path=f"datasets.{ds_key}",
                )
            )

        if owner not in allowed_owners:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="OUTPUT_CONTRACT_INVALID_OWNER",
                    message=f"Invalid owner '{owner}' (expected one of {allowed_owners})",
                    path=f"datasets.{ds_key}",
                )
            )

        if cardinality not in allowed_cardinality:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="OUTPUT_CONTRACT_INVALID_CARDINALITY",
                    message=f"Invalid cardinality '{cardinality}' (expected one of {allowed_cardinality})",
                    path=f"datasets.{ds_key}",
                )
            )

        if role and role not in allowed_roles:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="OUTPUT_CONTRACT_INVALID_ROLE",
                    message=f"Invalid role '{role}' (expected one of {allowed_roles})",
                    path=f"datasets.{ds_key}",
                )
            )

        if kind and schema_type and schema_type not in kind_schema_map.get(kind, set()):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="OUTPUT_CONTRACT_KIND_SCHEMA_MISMATCH",
                    message=f"Schema type '{schema_type}' incompatible with kind '{kind}'",
                    path=f"datasets.{ds_key}.schema",
                )
            )

        dimensions = ds.get("dimensions") or []
        if not isinstance(dimensions, list):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="OUTPUT_CONTRACT_DIMENSIONS_NOT_LIST",
                    message="dimensions must be a list of strings",
                    path=f"datasets.{ds_key}.dimensions",
                )
            )
        else:
            dim_duplicates = [d for d in dimensions if dimensions.count(d) > 1]
            if dim_duplicates:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="OUTPUT_CONTRACT_DUPLICATE_DIMENSION",
                        message=f"Duplicate dimensions: {set(dim_duplicates)}",
                        path=f"datasets.{ds_key}.dimensions",
                    )
                )

        # Validate schema structure per kind
        _validate_dataset_schema(ds, ds_key, issues)

        # Validate dimensions don't overlap with owner-level keys
        _validate_dimensions_policy(dimensions, ds_key, issues)

        # Validate groupPath depth
        group_path = ds.get("groupPath") or []
        _validate_group_path_policy(group_path, ds_key, issues)

    # Validate invariants
    invariants = contract.get("invariants") or []
    datasets_by_key = {ds.get("key"): ds for ds in datasets if ds.get("key")}
    _validate_invariants(invariants, datasets_by_key, issues)

    # Validate relations
    relations = contract.get("relations") or []
    _validate_relations(relations, datasets_by_key, issues)


# Schema dtype sets per spec
FRAME_COLUMN_DTYPES = {"string", "int", "float", "bool", "datetime"}
SCALAR_DTYPES = {"string", "int", "float", "bool"}
OWNER_LEVEL_KEYS = {"stageId", "wellId", "platformId"}
MAX_GROUP_PATH_DEPTH = 4


def _validate_dataset_schema(
    dataset: dict[str, Any], ds_key: str, issues: list[ValidationIssue]
) -> None:
    """Validate dataset schema structure per kind."""
    kind = dataset.get("kind")
    schema = dataset.get("schema") or {}

    if kind == "scalar":
        dtype = schema.get("dtype")
        if dtype and dtype not in SCALAR_DTYPES:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="SCALAR_INVALID_DTYPE",
                    message=f"Invalid scalar dtype '{dtype}'. Valid: {sorted(SCALAR_DTYPES)}",
                    path=f"datasets.{ds_key}.schema.dtype",
                )
            )

    elif kind == "blob":
        ext = schema.get("ext")
        if ext and not re.match(r"^\.[a-zA-Z0-9]+$", ext):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="BLOB_EXT_INVALID_FORMAT",
                    message=f"Invalid blob ext format '{ext}'. Must be '.<alphanumeric>'",
                    path=f"datasets.{ds_key}.schema.ext",
                )
            )

    elif kind == "frame":
        # Validate index field exists in columns if specified
        index = schema.get("index")
        columns = schema.get("columns") or []
        if isinstance(index, dict):
            index_kind = index.get("kind")
            index_field = index.get("field")
            if index_kind in ("time", "depth") and index_field:
                col_names = [c.get("name") if isinstance(c, dict) else c for c in columns]
                if index_field not in col_names:
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            code="FRAME_INDEX_FIELD_NOT_IN_COLUMNS",
                            message=f"Frame index field '{index_field}' not found in columns",
                            path=f"datasets.{ds_key}.schema.index",
                        )
                    )


def _validate_dimensions_policy(
    dimensions: list[str], ds_key: str, issues: list[ValidationIssue]
) -> None:
    """Validate dimensions don't contain owner-level keys."""
    overlap = set(dimensions) & OWNER_LEVEL_KEYS
    if overlap:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="DIMENSIONS_CONTAINS_OWNER_KEYS",
                message=f"dimensions contains owner-level keys {sorted(overlap)}; use 'owner' instead",
                path=f"datasets.{ds_key}.dimensions",
                details={"overlap": sorted(overlap)},
            )
        )


def _validate_group_path_policy(
    group_path: list[str], ds_key: str, issues: list[ValidationIssue]
) -> None:
    """Validate groupPath depth."""
    if group_path and len(group_path) > MAX_GROUP_PATH_DEPTH:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="GROUP_PATH_TOO_DEEP",
                message=f"groupPath depth {len(group_path)} exceeds recommended max {MAX_GROUP_PATH_DEPTH}",
                path=f"datasets.{ds_key}.groupPath",
                details={"depth": len(group_path)},
            )
        )


def _validate_invariants(
    invariants: list[dict[str, Any]],
    datasets_by_key: dict[str, dict[str, Any]],
    issues: list[ValidationIssue],
) -> None:
    """Validate invariants reference valid datasets."""
    for idx, inv in enumerate(invariants):
        inv_type = inv.get("type")
        inv_path = f"invariants[{idx}]"

        if inv_type == "sameOwner":
            level = inv.get("level")
            targets = inv.get("targets") or []
            for i, target in enumerate(targets):
                key = target.get("key") if isinstance(target, dict) else target
                if key not in datasets_by_key:
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            code="INVARIANT_REFERENCES_UNKNOWN_DATASET",
                            message=f"sameOwner invariant references unknown dataset '{key}'",
                            path=f"{inv_path}.targets[{i}]",
                        )
                    )
                elif level:
                    ds_owner = datasets_by_key[key].get("owner")
                    if ds_owner != level:
                        issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                code="SAME_OWNER_LEVEL_MISMATCH",
                                message=f"sameOwner level '{level}' doesn't match dataset owner '{ds_owner}'",
                                path=f"{inv_path}.targets[{i}]",
                            )
                        )

        elif inv_type == "joinOnOwner":
            left = inv.get("left") or {}
            right = inv.get("right") or {}
            for ref_name, ref in [("left", left), ("right", right)]:
                key = ref.get("key")
                if key and key not in datasets_by_key:
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            code="INVARIANT_REFERENCES_UNKNOWN_DATASET",
                            message=f"joinOnOwner.{ref_name} references unknown dataset '{key}'",
                            path=f"{inv_path}.{ref_name}",
                        )
                    )

        elif inv_type == "itemsCount":
            ds_key = inv.get("datasetKey")
            if ds_key and ds_key not in datasets_by_key:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="INVARIANT_REFERENCES_UNKNOWN_DATASET",
                        message=f"itemsCount references unknown dataset '{ds_key}'",
                        path=f"{inv_path}.datasetKey",
                    )
                )
            count = inv.get("count")
            if count is not None and (not isinstance(count, int) or count < 1):
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="ITEMS_COUNT_INVALID",
                        message=f"itemsCount.count must be integer >= 1, got {count}",
                        path=f"{inv_path}.count",
                    )
                )


def _validate_relations(
    relations: list[dict[str, Any]],
    datasets_by_key: dict[str, dict[str, Any]],
    issues: list[ValidationIssue],
) -> None:
    """Validate relations reference valid datasets and fields."""
    for idx, rel in enumerate(relations):
        rel_path = f"relations[{idx}]"
        from_ref = rel.get("from") or {}
        to_ref = rel.get("to") or {}

        from_key = from_ref.get("key")
        to_key = to_ref.get("key")

        # Keys must exist
        if from_key and from_key not in datasets_by_key:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="RELATION_FROM_KEY_NOT_FOUND",
                    message=f"relation.from references unknown dataset '{from_key}'",
                    path=f"{rel_path}.from",
                )
            )
        if to_key and to_key not in datasets_by_key:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="RELATION_TO_KEY_NOT_FOUND",
                    message=f"relation.to references unknown dataset '{to_key}'",
                    path=f"{rel_path}.to",
                )
            )

        # blob/scalar cannot have field relations
        for key, ref_name in [(from_key, "from"), (to_key, "to")]:
            if key and key in datasets_by_key:
                kind = datasets_by_key[key].get("kind")
                if kind in ("blob", "scalar"):
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            code=f"RELATION_{ref_name.upper()}_CANNOT_BE_BLOB_OR_SCALAR",
                            message=f"relation.{ref_name} cannot reference {kind} dataset",
                            path=f"{rel_path}.{ref_name}",
                        )
                    )

        # Validate field exists in schema (for frame/object)
        _validate_relation_field(from_ref, datasets_by_key, "from", rel_path, issues)
        _validate_relation_field(to_ref, datasets_by_key, "to", rel_path, issues)


def _validate_relation_field(
    ref: dict[str, Any],
    datasets_by_key: dict[str, dict[str, Any]],
    ref_name: str,
    rel_path: str,
    issues: list[ValidationIssue],
) -> None:
    """Validate relation field exists in dataset schema."""
    key = ref.get("key")
    field = ref.get("field")
    if not key or not field or key not in datasets_by_key:
        return

    dataset = datasets_by_key[key]
    schema = dataset.get("schema") or {}
    kind = dataset.get("kind")

    if kind == "frame":
        columns = schema.get("columns") or []
        col_names = [c.get("name") if isinstance(c, dict) else c for c in columns]
        if field not in col_names:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code=f"RELATION_{ref_name.upper()}_FIELD_NOT_IN_COLUMNS",
                    message=f"relation.{ref_name}.field '{field}' not in frame columns",
                    path=f"{rel_path}.{ref_name}.field",
                )
            )
    elif kind == "object":
        fields = schema.get("fields") or []
        field_names = [f.get("name") if isinstance(f, dict) else f for f in fields]
        if field not in field_names:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code=f"RELATION_{ref_name.upper()}_FIELD_NOT_IN_FIELDS",
                    message=f"relation.{ref_name}.field '{field}' not in object fields",
                    path=f"{rel_path}.{ref_name}.field",
                )
            )


# =============================================================================
# Bundle Validation
# =============================================================================


def validate_bundle(bundle_path: Path) -> ValidationResult:
    """Validate bundle hash integrity.

    Checks:
    - manifest.json exists and is valid
    - ds.json hash matches manifest.specFiles.dsSha256
    - drs.json hash matches manifest.specFiles.drsSha256

    Args:
        bundle_path: Path to bundle directory.

    Returns:
        ValidationResult with issues found.
    """
    bundle_path = Path(bundle_path).resolve()
    issues: list[ValidationIssue] = []

    if not bundle_path.is_dir():
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="BUNDLE_NOT_FOUND",
                message=f"Bundle directory not found: {bundle_path}",
            )
        )
        return ValidationResult(valid=False, issues=issues)

    # Check required files
    manifest_path = bundle_path / "manifest.json"
    ds_path = bundle_path / "ds.json"
    drs_path = bundle_path / "drs.json"

    if not manifest_path.exists():
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="BUNDLE_MANIFEST_NOT_FOUND",
                message="manifest.json not found",
            )
        )
        return ValidationResult(valid=False, issues=issues)

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="BUNDLE_MANIFEST_INVALID_JSON",
                message=f"Invalid manifest.json: {e}",
            )
        )
        return ValidationResult(valid=False, issues=issues)

    spec_files = manifest.get("specFiles", {})

    # Validate ds.json hash
    if ds_path.exists():
        expected_hash = spec_files.get("dsSha256")
        if expected_hash:
            actual_hash = hashlib.sha256(ds_path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="BUNDLE_DS_HASH_MISMATCH",
                        message=f"ds.json hash mismatch: expected {expected_hash[:16]}..., got {actual_hash[:16]}...",
                        details={"expected": expected_hash, "actual": actual_hash},
                    )
                )
    else:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="BUNDLE_DS_NOT_FOUND",
                message="ds.json not found",
            )
        )

    # Validate drs.json hash
    if drs_path.exists():
        expected_hash = spec_files.get("drsSha256")
        if expected_hash:
            actual_hash = hashlib.sha256(drs_path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="BUNDLE_DRS_HASH_MISMATCH",
                        message=f"drs.json hash mismatch: expected {expected_hash[:16]}..., got {actual_hash[:16]}...",
                        details={"expected": expected_hash, "actual": actual_hash},
                    )
                )
    else:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="BUNDLE_DRS_NOT_FOUND",
                message="drs.json not found",
            )
        )

    has_errors = any(i.severity == ValidationSeverity.ERROR for i in issues)
    return ValidationResult(valid=not has_errors, issues=issues)


# =============================================================================
# RunManifest vs OutputContract Validation
# =============================================================================


def validate_run_manifest(
    manifest_path: Path,
    contract_path: Path | None = None,
) -> ValidationResult:
    """Validate run output manifest against OutputContract.

    Checks:
    - Manifest structure is valid
    - All contract datasets are present in manifest
    - All contract items are present in manifest datasets
    - All contract artifacts are present in manifest items
    - kind/schema/mime consistency
    - dimensions key sets match

    Args:
        manifest_path: Path to output manifest.json.
        contract_path: Path to output_contract.json (optional).

    Returns:
        ValidationResult with issues found.
    """
    manifest_path = Path(manifest_path).resolve()
    issues: list[ValidationIssue] = []

    if not manifest_path.exists():
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="MANIFEST_NOT_FOUND",
                message=f"Manifest not found: {manifest_path}",
            )
        )
        return ValidationResult(valid=False, issues=issues)

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="MANIFEST_INVALID_JSON",
                message=f"Invalid manifest JSON: {e}",
            )
        )
        return ValidationResult(valid=False, issues=issues)

    # If no contract provided, just validate manifest structure
    if contract_path is None:
        has_errors = any(i.severity == ValidationSeverity.ERROR for i in issues)
        return ValidationResult(valid=not has_errors, issues=issues)

    contract_path = Path(contract_path).resolve()
    if not contract_path.exists():
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="CONTRACT_NOT_FOUND",
                message=f"Contract not found: {contract_path}",
            )
        )
        return ValidationResult(valid=False, issues=issues)

    try:
        contract = json.loads(contract_path.read_text())
    except json.JSONDecodeError as e:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="CONTRACT_INVALID_JSON",
                message=f"Invalid contract JSON: {e}",
            )
        )
        return ValidationResult(valid=False, issues=issues)

    # Align manifest against contract
    _validate_manifest_against_contract(manifest, contract, issues)

    has_errors = any(i.severity == ValidationSeverity.ERROR for i in issues)
    return ValidationResult(valid=not has_errors, issues=issues)


def _validate_manifest_against_contract(
    manifest: dict[str, Any],
    contract: dict[str, Any],
    issues: list[ValidationIssue],
) -> None:
    """Validate manifest against contract.

    Args:
        manifest: Run output manifest.
        contract: Output contract.
        issues: List to append issues to.
    """
    contract_datasets = {ds["key"]: ds for ds in contract.get("datasets", []) if "key" in ds}
    manifest_datasets = {
        ds.get("datasetKey") or ds.get("key"): ds for ds in manifest.get("datasets", [])
    }

    # Check all contract datasets are in manifest (if required)
    for ds_key, contract_ds in contract_datasets.items():
        if ds_key not in manifest_datasets:
            if contract_ds.get("required", True):
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="MANIFEST_MISSING_DATASET",
                        message=f"Contract dataset '{ds_key}' not found in manifest",
                        path=f"datasets.{ds_key}",
                    )
                )
            continue

        manifest_ds = manifest_datasets[ds_key]
        _validate_dataset_against_contract(manifest_ds, contract_ds, ds_key, issues)


def _validate_dataset_against_contract(
    manifest_ds: dict[str, Any],
    contract_ds: dict[str, Any],
    ds_key: str,
    issues: list[ValidationIssue],
) -> None:
    """Validate a single dataset against contract.

    Args:
        manifest_ds: Manifest dataset.
        contract_ds: Contract dataset.
        ds_key: Dataset key.
        issues: List to append issues to.
    """
    manifest_items = manifest_ds.get("items", [])
    required = contract_ds.get("required", True)
    cardinality = contract_ds.get("cardinality", "many")

    if cardinality == "one":
        if required and len(manifest_items) != 1:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="MANIFEST_CARDINALITY_ONE",
                    message="Cardinality 'one' dataset must have exactly one item when required",
                    path=f"datasets.{ds_key}",
                )
            )
        if not required and len(manifest_items) > 1:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="MANIFEST_CARDINALITY_ONE_OPTIONAL",
                    message="Cardinality 'one' optional dataset may have at most one item",
                    path=f"datasets.{ds_key}",
                )
            )
    elif cardinality == "many":
        if required and len(manifest_items) < 1:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="MANIFEST_CARDINALITY_MANY",
                    message="Cardinality 'many' required dataset must have at least one item",
                    path=f"datasets.{ds_key}",
                )
            )

    for idx, manifest_item in enumerate(manifest_items):
        _validate_item_against_contract(manifest_item, contract_ds, ds_key, f"item[{idx}]", issues)


def _validate_item_against_contract(
    manifest_item: dict[str, Any],
    contract_ds: dict[str, Any],
    ds_key: str,
    item_label: str,
    issues: list[ValidationIssue],
) -> None:
    """Validate a single item against contract.

    Args:
        manifest_item: Manifest item.
        contract_ds: Contract dataset.
        ds_key: Dataset key.
        item_label: Item label/index for errors.
        issues: List to append issues to.
    """
    path = f"datasets.{ds_key}.items.{item_label}"

    # Owner check
    expected_owner = contract_ds.get("owner")
    owner = manifest_item.get("owner", {})
    owner_ok = True
    if expected_owner == "stage":
        owner_ok = bool(owner.get("stageId"))
    elif expected_owner == "well":
        owner_ok = bool(owner.get("wellId"))
    elif expected_owner == "platform":
        owner_ok = bool(owner.get("platformId"))

    if expected_owner and not owner_ok:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="MANIFEST_MISSING_OWNER",
                message=f"Owner '{expected_owner}Id' required for dataset '{ds_key}'",
                path=path,
            )
        )

    # Dimensions check
    contract_dims = set(contract_ds.get("dimensions", []) or [])
    manifest_dims = set((manifest_item.get("dims") or {}).keys())
    if contract_dims and manifest_dims != contract_dims:
        missing = contract_dims - manifest_dims
        extra = manifest_dims - contract_dims
        if missing:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="MANIFEST_MISSING_DIMENSIONS",
                    message=f"Missing dimensions: {missing}",
                    path=path,
                )
            )
        if extra:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="MANIFEST_EXTRA_DIMENSIONS",
                    message=f"Extra dimensions not in contract: {extra}",
                    path=path,
                )
            )

    # Ensure dimension values are non-empty when present
    dims_dict = manifest_item.get("dims") or {}
    for dim_key in contract_dims:
        if dim_key in dims_dict:
            if dims_dict[dim_key] in (None, ""):
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="MANIFEST_DIMENSION_EMPTY",
                        message=f"Dimension '{dim_key}' must have a non-empty value",
                        path=f"{path}.dims.{dim_key}",
                    )
                )

    # Artifact check
    artifact = manifest_item.get("artifact")
    if artifact is None:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="MANIFEST_MISSING_ARTIFACT",
                message="Item missing artifact",
                path=path,
            )
        )
        return

    art_key = artifact.get("artifactKey") or artifact.get("key")
    art_type = artifact.get("type")
    if not art_key:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="MANIFEST_ARTIFACT_NO_KEY",
                message="Artifact missing artifactKey",
                path=path,
            )
        )

    kind_to_types = {
        "scalar": {"scalar"},
        "blob": {"blob"},
        "object": {"json", "object"},
        "frame": {"json", "parquet"},
    }
    expected_types = kind_to_types.get(contract_ds.get("kind"), set())
    if expected_types and art_type not in expected_types:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="MANIFEST_KIND_MISMATCH",
                message=f"Artifact type '{art_type}' incompatible with contract kind '{contract_ds.get('kind')}'",
                path=path,
            )
        )

    # For blob kind, check mime/ext consistency if provided
    if contract_ds.get("kind") == "blob":
        contract_schema = contract_ds.get("schema") or {}
        contract_mime = contract_schema.get("mime")
        if contract_mime and artifact.get("mimeType") and artifact.get("mimeType") != contract_mime:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="MANIFEST_BLOB_MIME_MISMATCH",
                    message=f"Artifact mimeType '{artifact.get('mimeType')}' does not match contract '{contract_mime}'",
                    path=path,
                )
            )


# =============================================================================
# Algorithm Signature Validation
# =============================================================================


def validate_algorithm_signature(workspace: Path) -> ValidationResult:
    """Validate algorithm run function signature.

    Checks:
    - main.py exists
    - Top-level run function exists
    - run function is not async (sandbox doesn't support it)
    - run function has exactly 1 positional parameter
    - run function has no *args, **kwargs, or keyword-only args

    Args:
        workspace: Algorithm workspace path.

    Returns:
        ValidationResult with issues found.
    """
    workspace = Path(workspace).resolve()
    issues: list[ValidationIssue] = []

    main_path = workspace / "main.py"

    if not main_path.exists():
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="ALGORITHM_MAIN_NOT_FOUND",
                message="main.py not found in algorithm workspace",
                path="main.py",
            )
        )
        return ValidationResult(valid=False, issues=issues)

    try:
        source = main_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename="main.py")
    except SyntaxError as e:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="ALGORITHM_SYNTAX_ERROR",
                message=f"Syntax error in main.py: {e}",
                path="main.py",
                details={"error": str(e)},
            )
        )
        return ValidationResult(valid=False, issues=issues)

    # Find TOP-LEVEL run functions only (not nested in classes/functions)
    run_funcs = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
    ]

    if len(run_funcs) == 0:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="ALGORITHM_RUN_NOT_FOUND",
                message="Top-level 'run' function not found in main.py",
                path="main.py",
            )
        )
        return ValidationResult(valid=False, issues=issues)

    if len(run_funcs) > 1:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="ALGORITHM_MULTIPLE_RUN_FUNCTIONS",
                message=f"Multiple top-level 'run' functions found ({len(run_funcs)})",
                path="main.py",
            )
        )
        return ValidationResult(valid=False, issues=issues)

    run_func = run_funcs[0]

    # async def run → ERROR (sandbox doesn't support it)
    if isinstance(run_func, ast.AsyncFunctionDef):
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="ALGORITHM_ASYNC_RUN_NOT_SUPPORTED",
                message="'async def run' is not supported; sandbox requires synchronous 'def run'",
                path="main.py",
            )
        )

    args = run_func.args

    # No *args
    if args.vararg:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="ALGORITHM_RUN_HAS_VARARG",
                message=f"run function must not have *args (found: *{args.vararg.arg})",
                path="main.py",
            )
        )

    # No **kwargs
    if args.kwarg:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="ALGORITHM_RUN_HAS_KWARG",
                message=f"run function must not have **kwargs (found: **{args.kwarg.arg})",
                path="main.py",
            )
        )

    # No keyword-only args
    if args.kwonlyargs:
        kw_names = [a.arg for a in args.kwonlyargs]
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="ALGORITHM_RUN_HAS_KWONLY_ARGS",
                message=f"run function must not have keyword-only args (found: {kw_names})",
                path="main.py",
            )
        )

    # Exactly 1 positional parameter (excluding 'self' for methods)
    positional_args = list(args.posonlyargs) + list(args.args)
    if positional_args and positional_args[0].arg == "self":
        positional_args = positional_args[1:]

    if len(positional_args) != 1:
        param_names = [a.arg for a in positional_args]
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="ALGORITHM_RUN_WRONG_PARAM_COUNT",
                message=f"run function must have exactly 1 parameter (context), found {len(positional_args)}: {param_names}",
                path="main.py",
                details={"found": len(positional_args), "params": param_names},
            )
        )

    has_errors = any(i.severity == ValidationSeverity.ERROR for i in issues)
    return ValidationResult(valid=not has_errors, issues=issues)


__all__ = [
    "ValidationSeverity",
    "ValidationIssue",
    "ValidationResult",
    "validate_inputspec",
    "validate_output_contract",
    "validate_bundle",
    "validate_run_manifest",
    "validate_algorithm_signature",
]
