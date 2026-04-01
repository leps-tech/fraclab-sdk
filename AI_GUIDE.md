# AI Guide For Fraclab SDK

This file is the machine-first authoring contract for this repo.

Read this before generating:

- algorithm workspace files
- schema files
- SDK examples
- docs that describe public behavior

If this file conflicts with a historical example under `algorithms/`, prefer this file plus the source models under `src/fraclab_sdk/`.

## Source Of Truth Priority

Use sources in this order:

1. `src/fraclab_sdk/runtime/runner_main.py`
2. `src/fraclab_sdk/runtime/output.py`
3. `src/fraclab_sdk/runtime/data_client.py`
4. `src/fraclab_sdk/models/output_contract.py`
5. `src/fraclab_sdk/models/drs.py`
6. `src/fraclab_sdk/models/dataspec.py`
7. `src/fraclab_sdk/models/algorithm_manifest.py`
8. `src/fraclab_sdk/devkit/validate.py`
9. `docs/getting-started/first-algorithm.md`
10. `docs/guides/input-spec.md`
11. `docs/guides/output.md`
12. `docs/guides/compile-export-import.md`

Do not infer public rules from old examples before checking the files above.

## Files To Avoid As Canonical Samples

These files contain legacy or overly relaxed patterns. Do not use them as source-of-truth when generating new code:

- `algorithms/schemas/specs/output.py`
- `algorithms/echo_algorithm/local/schema/output_contract.py`
- `algorithms/parquet-smoke-test/0.1.0/main.py`
- `algorithms/pressure-trace-viz/0.1.0/main.py`

Reasons:

- some use snake_case helper names where current validation expects camelCase keys
- some relax constrained fields into plain `str`
- some older examples reach into run-local files directly instead of treating `DataClient` as the runtime read contract
- some import paths do not match the real public package layout

## Repository Examples Worth Reading

If you need in-repo examples that follow the current SDK more closely, start with:

- `algorithms/bh-prop-conc/0.1.0/`
- `algorithms/frac-derived-curves/0.1.0/`
- `algorithms/hf-fracture-curves/0.1.0/`
- `algorithms/hf-fracture-curves-streaming/0.1.0/`

These are examples, not source-of-truth. Public behavior is still defined by `src/fraclab_sdk/` plus the docs listed above.

## Package Dependency Contract For Algorithms

Algorithm code may depend on:

- Python standard library
- `fraclab-sdk`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `fastapi`

Do not treat these as algorithm dependency contract:

- `pydantic`
- `plotly`
- `streamlit`
- `streamlit-plotly-events`
- `pyarrow`
- `typer`
- `rich`

Even if those packages are present in the SDK environment, algorithm code should not rely on them as required runtime dependencies.

## Required Algorithm Workspace Layout

Minimum workspace:

```text
my-algorithm/
  main.py
  manifest.json
  schema/
    base.py
    inputspec.py
    output_contract.py
```

Generated build artifacts:

```text
my-algorithm/
  dist/
    params.schema.json
    output_contract.json
    ds.json
    drs.json
```

## Runtime Contract

Algorithm entrypoint file is `main.py`.

Algorithm entrypoint function is:

```python
def run(ctx) -> None:
    ...
```

Hard rules:

- `run()` must accept exactly one argument: `ctx`
- use `ctx.data_client` to read input data
- use `ctx.output` to write outputs
- `ctx.params` is `dict[str, Any]`
- `ctx.logger` is a standard logger
- `ctx.run_context` is a metadata dict
- do not invent `ctx.artifacts`
- return value is not the output contract

Actual runtime context fields come from `RunContext` in `src/fraclab_sdk/runtime/runner_main.py`.

## Minimal Correct Algorithm

```python
def run(ctx) -> None:
    dc = ctx.data_client
    out = ctx.output

    summary = {}
    for dataset_key in dc.get_dataset_keys():
        summary[dataset_key] = {
            "layout": dc.get_layout(dataset_key),
            "count": dc.get_item_count(dataset_key),
        }

    out.write_json("summary", summary, dataset_key="summary")
    out.write_scalar("status", "ok", dataset_key="status")
```

## Input Reading Rules

- call `ctx.data_client.get_dataset_keys()` to discover available datasets
- call `ctx.data_client.get_layout(dataset_key)` before assuming access pattern
- only use `read_object()` and `iterate_objects()` for `object_ndjson_lines`
- use `get_frame_columns()` to discover logical frame columns for `frame_parquet_item_dirs`
- use `read_frame()`, `iter_frame_chunks()`, `iter_dataset_frame_chunks()`, or `iter_frame_batches()` to read frame datasets
- do not use `get_parquet_dir()` or `get_parquet_files()` in new algorithms; raw parquet path access is no longer part of the public runtime contract
- do not assume every dataset is parquet-backed

Current bundle layout values:

- `object_ndjson_lines`
- `frame_parquet_item_dirs`

## Output Writing Rules

Use only runtime-injected `ctx.output`.

Preferred methods:

- `write_scalar()`
- `write_text()`
- `write_json()`
- `write_bytes()`
- `write_blob()`
- `write_file()`
- `register_file()`
- `write_dataframe_csv()`
- `write_figure_png()`

Hard rules:

- do not instantiate `OutputClient` yourself
- do not write outside `ctx.output.dir`
- do not hand-write `output/manifest.json`
- if a library needs a path, write into `ctx.output.dir` then register the file
- if you provide `dataset_key`, `owner`, `dims`, `meta`, or `item_key`, they must match the declared `OutputContract`

`output/manifest.json` is generated by the runner from:

1. explicit `write_*()` / `register_file()` records
2. untracked files discovered under `ctx.output.dir`

## JSON Naming Boundary

There are two different naming domains.

Algorithm-defined JSON uses camelCase:

- `params.schema.json`
- `output_contract.json`
- run output `output/manifest.json`
- keys accessed in `ctx.params`

Bundle-provided JSON keeps the platform format:

- `ds.json`
- `drs.json`

Do not rewrite bundle JSON into invented camelCase keys.

Correct bundle field examples:

- `key`
- `resource`
- `layout`
- `items`

## Schema Authoring Overview

When generating a new algorithm workspace, there are 4 author-written files that matter most:

1. `schema/base.py`
2. `schema/inputspec.py`
3. `schema/output_contract.py`
4. `manifest.json`

The SDK then compiles them into `dist/*`.

## `schema/base.py` Rules

Preferred base utilities are the ones produced by `src/fraclab_sdk/algorithm/scaffold.py`.

Use these names and shapes:

- `CamelModel`
- `showWhenCondition()`
- `showWhenAnd()`
- `showWhenOr()`
- `schemaExtra()`
- `opt_bool()`
- `TimeWindow`
- `time_window_list()`

Do not use snake_case helper names such as:

- `schema_extra`
- `show_when_condition`
- `show_when_and`
- `show_when_or`

Those helper names can generate `json_schema_extra` keys that fail current validation.

`time_window_list()` from the scaffold returns `Optional[list[TimeWindow]]`.
Do not wrap it in another `Optional[...]`.

## `schema/inputspec.py` Rules

Hard rules:

- file path is `schema/inputspec.py`
- must export `INPUT_SPEC`
- prefer inheriting from `schema/base.py` `CamelModel`
- schema field names must be camelCase
- runtime accesses the generated values through `ctx.params`, which is a plain dict

Minimal correct example:

```python
from .base import CamelModel, Field


class InputParams(CamelModel):
    maxItems: int = Field(default=10, ge=1)
    datasetKey: str = Field(default="wells")


INPUT_SPEC = InputParams
```

Corresponding runtime usage:

```python
def run(ctx) -> None:
    max_items = int(ctx.params.get("maxItems", 10))
    dataset_key = ctx.params.get("datasetKey", "wells")
```

Do not treat `ctx.params` as a Pydantic model instance.

### `json_schema_extra` Rules

Current validation enforces:

- keys must use camelCase
- unknown keys are errors
- snake_case legacy keys are errors

Examples of current camelCase keys used by validation:

- `group`
- `order`
- `unit`
- `step`
- `uiType`
- `collapsible`
- `showWhen`
- `enumLabels`
- `bindDatasetKey`
- `windowSlots`
- `windowSlotFallbackNote`

### Numeric UI Rule

`step` controls Workbench numeric precision.

Examples:

- `step=0.01` -> 2 decimals
- `step=0.001` -> 3 decimals

### Time Window Rule

This is one of the highest-risk areas for AI generation.

The ONLY supported shape is:

```python
Optional[list[TimeWindow]]
```

or the scaffold helper equivalent:

```python
WindowsTemplate = time_window_list(...)
```

Correct example:

```python
from .base import CamelModel, Field, schemaExtra, time_window_list


WindowsTemplate = time_window_list(
    min_items=1,
    max_items=3,
    title="Windows",
    description="Template windows.",
)


class InputParams(CamelModel):
    timeWindows_fracRecord_stage_5712: WindowsTemplate = Field(
        default=None,
        title="Time Windows",
        json_schema_extra=schemaExtra(
            uiType="time_window",
            unit="us",
            bindDatasetKey="fracRecord_stage_5712",
        ),
    )


INPUT_SPEC = InputParams
```

Hard rules for `uiType="time_window"`:

- field must be `Optional[list[TimeWindow]]`
- bare `list[TimeWindow]` is invalid
- nested `list[list[TimeWindow]]` is invalid
- `bindDatasetKey` is required
- `unit` must be explicitly `"us"`
- `bindDatasetKey` must match `dist/drs.json` `datasets[*].key`
- canonical field naming pattern is `timeWindows_<datasetKey>`

## `schema/output_contract.py` Rules

Hard rules:

- file path is `schema/output_contract.py`
- must export `OUTPUT_CONTRACT`
- use the real models from `fraclab_sdk.models.output_contract`
- keep JSON keys camelCase

Preferred imports:

```python
from fraclab_sdk.models.output_contract import (
    BlobOutputSchema,
    FrameOutputSchema,
    ObjectOutputSchema,
    OutputContract,
    OutputDatasetContract,
    ScalarOutputSchema,
)
```

Do not import from fake or legacy paths such as `fraclab_sdk.specs.output`.

### Allowed Literal Values

These values come from `src/fraclab_sdk/models/output_contract.py`.
Never invent alternatives.

`OutputDatasetContract`:

- `kind`: `"frame" | "object" | "blob" | "scalar"`
- `owner`: `"stage" | "well" | "platform"`
- `cardinality`: `"one" | "many"`
- `role`: `"primary" | "supporting" | "debug" | None`

`FrameOutputSchema`:

- `type`: `"frame"`
- `index`: `"time" | "depth" | "none" | None`

Other output schema types:

- `ScalarOutputSchema.type`: `"scalar"`
- `ObjectOutputSchema.type`: `"object"`
- `BlobOutputSchema.type`: `"blob"`

### Minimal Correct Output Contract

```python
from fraclab_sdk.models.output_contract import (
    OutputContract,
    OutputDatasetContract,
    ScalarOutputSchema,
)


OUTPUT_CONTRACT = OutputContract(
    datasets=[
        OutputDatasetContract(
            key="status",
            kind="scalar",
            owner="platform",
            cardinality="one",
            schema=ScalarOutputSchema(type="scalar", dtype="string"),
        )
    ]
)
```

### Output Contract Matching Rules

- `kind="scalar"` must use `schema.type="scalar"`
- `kind="frame"` must use `schema.type="frame"`
- `kind="object"` must use `schema.type="object"` or a compatible object schema dict
- `kind="blob"` must use `schema.type="blob"`

Additional structure rules enforced by validation:

- dataset keys must be unique
- `dimensions` must be a list
- duplicate dimensions are invalid
- `groupPath` depth is validated and deep nesting is discouraged

### DRS vs OutputContract Cardinality Difference

Do not mix these two enums.

`DRS.cardinality` allows:

- `one`
- `many`
- `zeroOrMany`

`OutputContract.cardinality` allows only:

- `one`
- `many`

## `manifest.json` Rules

Algorithm package manifest uses `FracLabAlgorithmManifestV1` from `src/fraclab_sdk/models/algorithm_manifest.py`.

Hard rules:

- `manifestVersion` is `"1"`
- `algorithmId` must be a non-empty identifier
- `name` must be non-empty
- `summary` must be non-empty
- `authors` must contain at least one author
- `contractVersion` must be semver-like, for example `1.2.3`
- `files` should point to built artifacts under `dist/`

Minimal shape:

```json
{
  "manifestVersion": "1",
  "algorithmId": "my-algo",
  "name": "My Algo",
  "summary": "Short summary",
  "authors": [{ "name": "author" }],
  "contractVersion": "1.0.0",
  "codeVersion": "0.1.0",
  "files": {
    "paramsSchemaPath": "dist/params.schema.json",
    "outputContractPath": "dist/output_contract.json",
    "drsPath": "dist/drs.json",
    "dsPath": "dist/ds.json"
  }
}
```

## Compile / Export / Import Rules

Compile:

```bash
fraclab-sdk algo compile ./my-algorithm --bundle /path/to/bundle
```

Export:

```bash
fraclab-sdk algo export ./my-algorithm ./my-algorithm.zip --auto-compile --bundle /path/to/bundle
```

Behavior:

- compile generates `dist/params.schema.json` from `INPUT_SPEC`
- compile generates `dist/output_contract.json` from `OUTPUT_CONTRACT`
- compile copies `ds.json` and `drs.json` from the bundle
- export requires a complete `dist/`
- `--auto-compile` compiles first when needed

Import checks include:

- `manifest.json` exists
- package paths are safe
- key files are present

## Selection And Reindexing Rules

This is another area where AI often makes wrong assumptions.

Selection uses snapshot item indices.

`SelectionModel.build_run_ds()` produces a new `DataSpec` whose items are re-indexed to `0..N-1`.
Each item may carry `sourceItemIndex` for traceability.

Implications:

- do not assume run item index equals snapshot item index
- when explaining selection behavior, distinguish snapshot indices from run-local indices
- if code needs original snapshot index, use the mapping or `sourceItemIndex`

## Human Doc vs Machine Doc Boundary

- full human docs live in `docs/`
- this file is the AI authoring contract
- `llms.txt` is a short machine entrypoint that points back to this file

## Common Invalid Patterns

Do not generate any of these:

- `ctx.artifacts`
- `fraclab_sdk.specs.output`
- snake_case schema field names like `dataset_key`
- snake_case `json_schema_extra` keys like `ui_type`, `show_when`, `bind_dataset_key`
- `owner: str` in place of the constrained owner literals
- `cardinality: str` in place of constrained literals
- time window fields without `bindDatasetKey`
- time window fields without `unit="us"`
- hand-written `output/manifest.json`
- output files written outside `ctx.output.dir`

## Generation Checklist

Before finishing AI-generated code, verify all of the following:

1. `main.py` defines `run(ctx)` and uses `ctx.output`, not `ctx.artifacts`.
2. `schema/inputspec.py` exports `INPUT_SPEC`.
3. `schema/output_contract.py` exports `OUTPUT_CONTRACT`.
4. InputSpec field names are camelCase.
5. `json_schema_extra` keys are camelCase.
6. `ctx.params` access uses dict keys, not attributes.
7. Bundle JSON examples use `key` / `resource` / `layout`, not invented names.
8. `OutputContract` literals come from the repo-defined closed sets only.
9. Time window fields use `Optional[list[TimeWindow]]` and `unit="us"`.
10. `bindDatasetKey` matches a DRS dataset key.
11. Output metadata matches the declared `OutputContract`.
12. `manifest.json` uses `manifestVersion="1"` and semver-like `contractVersion`.

## Short Answer For Code Generation

When asked to create a new Fraclab algorithm, generate:

1. `main.py` with `run(ctx)`
2. `schema/base.py` using scaffold-style helpers with camelCase helper names
3. `schema/inputspec.py` exporting `INPUT_SPEC`
4. `schema/output_contract.py` exporting `OUTPUT_CONTRACT`
5. `manifest.json` pointing to `dist/*`

Then ensure the generated code compiles with:

```bash
fraclab-sdk algo compile ./my-algorithm --bundle /path/to/bundle
```
