"""Shared JSON-schema helpers for workbench parameter forms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnumChoice:
    """A single enum option for UI rendering."""

    value: Any
    label: str


@dataclass(frozen=True)
class ObjectUnionBranch:
    """A selectable object-union branch."""

    key: str
    label: str
    schema: dict[str, Any]


@dataclass(frozen=True)
class ObjectUnionSchema:
    """Resolved object union schema for discriminator-based rendering."""

    nullable: bool
    discriminator: str | None
    branches: tuple[ObjectUnionBranch, ...]


def extract_ui_type(schema: dict[str, Any]) -> str | None:
    """Extract uiType from schema field metadata."""
    if isinstance(schema.get("uiType"), str):
        return schema.get("uiType")
    extra = schema.get("json_schema_extra")
    if isinstance(extra, dict) and isinstance(extra.get("uiType"), str):
        return extra.get("uiType")
    return None


def schema_meta(schema: dict[str, Any], key: str, default: Any = None) -> Any:
    """Read metadata from top-level first, then json_schema_extra."""
    if key in schema:
        return schema.get(key)
    extra = schema.get("json_schema_extra")
    if isinstance(extra, dict) and key in extra:
        return extra.get(key)
    return default


def resolve_ref_schema(schema: dict[str, Any], root_schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve local JSON-schema $ref like '#/$defs/Name'."""
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


def split_nullable_branches(
    schema: dict[str, Any],
    root_schema: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    """Return non-null branches and whether the schema allows null."""
    if "$ref" in schema:
        schema = resolve_ref_schema(schema, root_schema)
    any_of = schema.get("anyOf")
    if not isinstance(any_of, list):
        return [schema], False
    non_null: list[dict[str, Any]] = []
    nullable = False
    for branch in any_of:
        if not isinstance(branch, dict):
            continue
        if branch.get("type") == "null":
            nullable = True
            continue
        resolved = resolve_ref_schema(branch, root_schema) if "$ref" in branch else branch
        non_null.append(resolved if isinstance(resolved, dict) else branch)
    return non_null or [schema], nullable


def normalize_schema(schema: dict[str, Any], root_schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve common wrappers (anyOf/null + $ref) for type-shape checks."""
    branches, _nullable = split_nullable_branches(schema, root_schema)
    current = branches[0] if len(branches) == 1 else schema
    if "$ref" in current:
        current = resolve_ref_schema(current, root_schema)
    return current


def is_nullable_schema(schema: dict[str, Any], root_schema: dict[str, Any]) -> bool:
    """Whether the field schema explicitly allows null."""
    _branches, nullable = split_nullable_branches(schema, root_schema)
    return nullable


def extract_enum_choices(schema: dict[str, Any], root_schema: dict[str, Any]) -> tuple[tuple[EnumChoice, ...], bool] | None:
    """Extract scalar enum options, preserving enumLabels."""
    branches, nullable = split_nullable_branches(schema, root_schema)
    if len(branches) != 1:
        return None
    branch = normalize_schema(branches[0], root_schema)
    enum_values = branch.get("enum")
    if not isinstance(enum_values, list):
        if "const" not in branch:
            return None
        enum_values = [branch.get("const")]

    labels = schema_meta(schema, "enumLabels", {})
    if not isinstance(labels, dict):
        labels = {}
    choices = tuple(
        EnumChoice(
            value=value,
            label=str(labels.get(str(value)) or value),
        )
        for value in enum_values
    )
    return choices, nullable


def extract_array_enum_choices(
    schema: dict[str, Any],
    root_schema: dict[str, Any],
) -> tuple[tuple[EnumChoice, ...], bool] | None:
    """Extract array item enum options for multiselect rendering."""
    branches, nullable = split_nullable_branches(schema, root_schema)
    if len(branches) != 1:
        return None
    branch = normalize_schema(branches[0], root_schema)
    if branch.get("type") != "array":
        return None
    items = branch.get("items")
    if not isinstance(items, dict):
        return None
    return extract_enum_choices(items, root_schema)


def _branch_identity(
    branch_schema: dict[str, Any],
    discriminator: str | None,
    index: int,
) -> tuple[str, str] | None:
    if discriminator:
        props = branch_schema.get("properties")
        if not isinstance(props, dict):
            return None
        disc_schema = props.get(discriminator)
        if not isinstance(disc_schema, dict):
            return None
        if "const" in disc_schema:
            key = str(disc_schema.get("const"))
        else:
            enum_values = disc_schema.get("enum")
            if not isinstance(enum_values, list) or len(enum_values) != 1:
                return None
            key = str(enum_values[0])
        return key, str(branch_schema.get("title") or key)
    label = str(branch_schema.get("title") or f"Option {index + 1}")
    return label, label


def extract_object_union(schema: dict[str, Any], root_schema: dict[str, Any]) -> ObjectUnionSchema | None:
    """Resolve discriminator-based object unions used by InputSpec."""
    branches, nullable = split_nullable_branches(schema, root_schema)
    if len(branches) != 1:
        return None
    branch_root = normalize_schema(branches[0], root_schema)
    discriminator_cfg = branch_root.get("discriminator")
    discriminator = None
    if isinstance(discriminator_cfg, dict) and isinstance(discriminator_cfg.get("propertyName"), str):
        discriminator = discriminator_cfg["propertyName"]

    raw_branches = branch_root.get("oneOf")
    if not isinstance(raw_branches, list):
        return None

    resolved_branches: list[ObjectUnionBranch] = []
    for idx, raw_branch in enumerate(raw_branches):
        if not isinstance(raw_branch, dict):
            return None
        resolved = normalize_schema(raw_branch, root_schema)
        if resolved.get("type") != "object":
            return None
        identity = _branch_identity(resolved, discriminator, idx)
        if identity is None:
            return None
        key, label = identity
        resolved_branches.append(ObjectUnionBranch(key=key, label=label, schema=resolved))

    if not resolved_branches:
        return None
    return ObjectUnionSchema(
        nullable=nullable,
        discriminator=discriminator,
        branches=tuple(resolved_branches),
    )


def match_object_union_branch(union_schema: ObjectUnionSchema, value: Any) -> str | None:
    """Pick the active union branch from the current field value."""
    if not isinstance(value, dict):
        return None
    if union_schema.discriminator:
        current = value.get(union_schema.discriminator)
        for branch in union_schema.branches:
            if current == branch.key:
                return branch.key
    return union_schema.branches[0].key if union_schema.branches else None


def constant_field_value(schema: dict[str, Any], root_schema: dict[str, Any]) -> Any:
    """Return a schema const value, if present."""
    normalized = normalize_schema(schema, root_schema)
    if "const" in normalized:
        return normalized.get("const")
    enum_values = normalized.get("enum")
    if isinstance(enum_values, list) and len(enum_values) == 1:
        return enum_values[0]
    return None


def sort_schema_properties(properties: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Sort schema properties by order while preserving declaration order."""
    ordered: list[tuple[str, int, int, str, dict[str, Any]]] = []
    unordered: list[tuple[int, str, dict[str, Any]]] = []
    for idx, (key, schema) in enumerate(properties.items()):
        order = schema_meta(schema, "order")
        if isinstance(order, int):
            ordered.append((order, idx, key, str(schema.get("title") or key), schema))
        else:
            unordered.append((idx, key, schema))
    ordered.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return [(key, schema) for _order, _idx, key, _title, schema in ordered] + [
        (key, schema) for _idx, key, schema in unordered
    ]


def _lookup_path(values: dict[str, Any], field_path: str) -> Any:
    current: Any = values
    for part in field_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _evaluate_show_when_condition(condition: dict[str, Any], values: dict[str, Any]) -> bool:
    field_path = condition.get("field")
    op = condition.get("op", "equals")
    expected = condition.get("value")
    if not isinstance(field_path, str) or not field_path:
        return True
    actual = _lookup_path(values, field_path)
    if op == "equals":
        return actual == expected
    if op == "not_equals":
        return actual != expected
    if op == "gt":
        return actual is not None and expected is not None and actual > expected
    if op == "gte":
        return actual is not None and expected is not None and actual >= expected
    if op == "lt":
        return actual is not None and expected is not None and actual < expected
    if op == "lte":
        return actual is not None and expected is not None and actual <= expected
    if op == "in":
        return isinstance(expected, list) and actual in expected
    if op == "not_in":
        return isinstance(expected, list) and actual not in expected
    return True


def evaluate_show_when(show_when: Any, values: dict[str, Any]) -> bool:
    """Evaluate showWhen against the current param values."""
    if show_when is None:
        return True
    if isinstance(show_when, dict):
        if "or" in show_when:
            conditions = show_when.get("or")
            return any(_evaluate_show_when_condition(cond, values) for cond in conditions) if isinstance(conditions, list) else True
        if "and" in show_when:
            conditions = show_when.get("and")
            return all(_evaluate_show_when_condition(cond, values) for cond in conditions) if isinstance(conditions, list) else True
        return _evaluate_show_when_condition(show_when, values)
    if isinstance(show_when, list):
        return all(_evaluate_show_when_condition(cond, values) for cond in show_when if isinstance(cond, dict))
    return True


def field_is_visible(schema: dict[str, Any], values: dict[str, Any]) -> bool:
    """Whether a field should be shown for the current values."""
    show_when = schema_meta(schema, "showWhen")
    return evaluate_show_when(show_when, values)
