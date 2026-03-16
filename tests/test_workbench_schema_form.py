from fraclab_sdk.workbench.schema_form import (
    extract_enum_choices,
    extract_object_union,
    field_is_visible,
    match_object_union_branch,
    sort_schema_properties,
)


def test_sort_schema_properties_respects_order_and_preserves_unordered_tail() -> None:
    properties = {
        "third": {"type": "string"},
        "second": {"type": "string", "order": 20},
        "first": {"type": "string", "order": 10},
        "fourth": {"type": "string"},
    }

    ordered = sort_schema_properties(properties)

    assert [key for key, _schema in ordered] == ["first", "second", "third", "fourth"]


def test_field_is_visible_evaluates_show_when_against_current_values() -> None:
    schema = {
        "type": "number",
        "showWhen": {
            "and": [
                {"field": "enabled", "op": "equals", "value": True},
                {"field": "mode", "op": "in", "value": ["advanced", "expert"]},
            ]
        },
    }

    assert field_is_visible(schema, {"enabled": True, "mode": "advanced"}) is True
    assert field_is_visible(schema, {"enabled": False, "mode": "advanced"}) is False
    assert field_is_visible(schema, {"enabled": True, "mode": "basic"}) is False


def test_extract_enum_choices_uses_enum_labels_and_nullable_wrapper() -> None:
    schema = {
        "anyOf": [
            {
                "type": "string",
                "enum": ["movingAverage", "gaussian"],
            },
            {"type": "null"},
        ],
        "enumLabels": {
            "movingAverage": "Moving Average",
            "gaussian": "Gaussian",
        },
    }

    extracted = extract_enum_choices(schema, schema)

    assert extracted is not None
    choices, nullable = extracted
    assert nullable is True
    assert [(choice.value, choice.label) for choice in choices] == [
        ("movingAverage", "Moving Average"),
        ("gaussian", "Gaussian"),
    ]


def test_extract_object_union_supports_discriminator_and_branch_match() -> None:
    schema = {
        "anyOf": [
            {
                "discriminator": {"propertyName": "method"},
                "oneOf": [
                    {"$ref": "#/$defs/SmoothingMovingAverage"},
                    {"$ref": "#/$defs/SmoothingGaussian"},
                ],
            },
            {"type": "null"},
        ],
        "$defs": {
            "SmoothingMovingAverage": {
                "type": "object",
                "title": "Moving Average",
                "properties": {
                    "method": {"const": "movingAverage", "type": "string"},
                    "params": {"type": "object", "properties": {"windowSize": {"type": "integer"}}},
                },
            },
            "SmoothingGaussian": {
                "type": "object",
                "title": "Gaussian",
                "properties": {
                    "method": {"const": "gaussian", "type": "string"},
                    "params": {"type": "object", "properties": {"sigma": {"type": "number"}}},
                },
            },
        },
    }

    union_schema = extract_object_union(schema, schema)

    assert union_schema is not None
    assert union_schema.nullable is True
    assert union_schema.discriminator == "method"
    assert [branch.key for branch in union_schema.branches] == ["movingAverage", "gaussian"]
    assert match_object_union_branch(union_schema, {"method": "gaussian", "params": {"sigma": 6.0}}) == "gaussian"
