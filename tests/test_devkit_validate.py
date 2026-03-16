from fraclab_sdk.devkit.validate import _validate_schema_properties


def _issue_codes(issues):
    return [issue.code for issue in issues]


def test_top_level_inputspec_fields_still_require_camel_case() -> None:
    schema = {
        "type": "object",
        "properties": {
            "bad_field": {"type": "string", "title": "Bad Field"},
        },
    }
    issues = []

    _validate_schema_properties(schema, schema, "", issues)

    assert _issue_codes(issues) == ["INPUTSPEC_FIELD_NOT_CAMEL_CASE"]
    assert issues[0].path == "bad_field"


def test_nested_parameter_fields_do_not_require_camel_case() -> None:
    schema = {
        "type": "object",
        "properties": {
            "dataSelection": {"$ref": "#/$defs/DataSelection"},
        },
        "$defs": {
            "DataSelection": {
                "type": "object",
                "properties": {
                    "timeWindows_fracRecord_stage_4026": {
                        "type": "array",
                        "title": "Stage Windows",
                        "items": {"type": "number"},
                    }
                },
            }
        },
    }
    issues = []

    _validate_schema_properties(schema, schema, "", issues)

    assert "INPUTSPEC_FIELD_NOT_CAMEL_CASE" not in _issue_codes(issues)


def test_nested_parameter_fields_still_validate_sdk_metadata_keys() -> None:
    schema = {
        "type": "object",
        "properties": {
            "dataSelection": {
                "type": "object",
                "properties": {
                    "timeWindows_fracRecord_stage_4026": {
                        "type": "string",
                        "title": "Stage Selection",
                        "ui_type": "text",
                    }
                },
            }
        },
    }
    issues = []

    _validate_schema_properties(schema, schema, "", issues)

    assert _issue_codes(issues) == ["JSON_SCHEMA_EXTRA_SNAKE_CASE_KEY"]
    assert issues[0].path == "dataSelection.timeWindows_fracRecord_stage_4026.ui_type"


def test_time_window_dataset_bound_field_name_is_allowed() -> None:
    schema = {
        "type": "object",
        "properties": {
            "timeWindows_samples_core_stage_2222": {
                "anyOf": [
                    {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "min": {"type": "number"},
                                "max": {"type": "number"},
                            },
                            "required": ["min", "max"],
                        },
                    },
                    {"type": "null"},
                ],
                "uiType": "time_window",
                "bindDatasetKey": "samples_core_stage_2222",
                "title": "Time Windows",
            }
        },
    }
    issues = []

    _validate_schema_properties(schema, schema, "", issues)

    assert "INPUTSPEC_FIELD_NOT_CAMEL_CASE" not in _issue_codes(issues)


def test_time_window_optional_shape_accepts_nullable_array_schema() -> None:
    schema = {
        "type": "object",
        "properties": {
            "timeWindows_samples_core_stage_2222": {
                "anyOf": [
                    {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "min": {"type": "number"},
                                "max": {"type": "number"},
                            },
                            "required": ["min", "max"],
                        },
                    },
                    {"type": "null"},
                ],
                "default": None,
                "uiType": "time_window",
                "bindDatasetKey": "samples_core_stage_2222",
                "title": "Time Windows",
            }
        },
    }
    issues = []

    _validate_schema_properties(schema, schema, "", issues)

    assert "TIME_WINDOW_NOT_OPTIONAL" not in _issue_codes(issues)
