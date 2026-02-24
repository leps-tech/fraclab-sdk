# Changelog

## 0.1.4 (Unreleased)

### Algorithm Scaffold & CLI
- Introduce shared scaffold module `src/fraclab_sdk/algorithm/scaffold.py`; Workbench and CLI now use the same scaffold/base-template source.
- New CLI command: `fraclab-sdk algo init` to create local algorithm workspace scaffolds without relying on Workbench.
- Scaffold/base template upgraded with shared schema helpers and UI models:
  - `opt_bool(...)`
  - `TimeWindow`
  - `time_window_list(...)`
- Existing repository examples and local `~/.fraclab/algorithms/*/schema/base.py` were aligned to the new base template.

### Workbench Run: Time Window (New Schema Semantics)
- Run page time-window rendering now hard-targets `List[TimeWindow]` / `Optional[List[TimeWindow]]` (clean break from old nested shapes).
- Multiple time-window schema fields are merged into one unified picker at the bottom of the parameters panel.
- Unified picker switches dataset internally; constraints are applied per dataset via each field’s `bind_dataset_key`.
- Matching is based on selected run datasets (`runs/<run_id>/input/ds.json`), not on params `datasetKey`.
- Added per-dataset slot guidance near the chart:
  - reads `window_slots` and `window_slot_fallback_note`
  - shows next slot note based on current selected window count.

## 0.1.3

### Workbench & Packaging
- Export page now previews selected bundle `ds.json` and `drs.json` in separate fixed-height tabs.
- Build/export flow now injects bundle `ds.json` and `drs.json` into algorithm package output and auto-fills missing `files.dsPath` / `files.drsPath` during export.
- New algorithm scaffold / single-`py` import no longer pre-generates empty `dist/drs.json`; `drs` is filled at export time.
- Browse page removes low-signal dataset summary card and redundant divider under dataset tabs for cleaner navigation.
- Run page number parameter precision now follows schema `step`:
  - with `step`: decimal precision derived from `step`
  - without `step`: integer-style display (no decimal places)

### Manifest Tolerance (Import Path)
- Relaxed algorithm manifest file pointer strictness for import path:
  - `files.outputContractPath`, `files.dsPath`, `files.drsPath` are optional
  - `files.paramsSchemaPath` remains required
- Missing `drsPath` now falls back to empty DRS in algorithm handle, allowing selection model to infer datasets from snapshot.

### Release & CI
- CI publish guard: Git tag version must match both `pyproject.toml` and `src/fraclab_sdk/version.py`, otherwise PyPI publish fails fast.
- Add `scripts/release.sh` for one-command version bump (and optional commit/tag/push) to reduce manual release mistakes.

## 0.1.2

### Validation System Upgrade
- **InputSpec strict validation**:
  - `json_schema_extra` whitelist enforcement (unknown keys → ERROR, `x_*` prefix → WARNING for extensions)
  - `show_when` canonical operators only: `equals`, `not_equals`, `gt`, `gte`, `lt`, `lte`, `in`, `not_in`
  - Operator aliases (`eq`, `neq`, `nin`) emit WARNING and normalize to canonical form
  - snake_case field path detection with camelCase fix suggestions (UI compatibility)
  - JSON Schema path resolution with `$ref`, `allOf`, `anyOf/oneOf` handling (Pydantic v2 patterns)
  - `enum_labels` must match enum values exactly (missing/extra keys → ERROR)
  - Leaf fields missing `title` emit WARNING
- **OutputContract policy checks**:
  - Schema structure validation per kind (`frame`, `object`, `blob`, `scalar`)
  - `dimensions` cannot overlap with owner-level keys (`stageId`, `wellId`, `platformId`)
  - `groupPath` depth limit warning (max 4 levels)
  - Invariants validation: dataset references must exist, `sameOwner` level must match
  - Relations validation: dataset/field references must exist, blob/scalar cannot have field relations
- **Algorithm signature validation**:
  - `main.py` must exist with top-level `run` function
  - `async def run` not supported (sandbox limitation)
  - Exactly 1 positional parameter required (no `*args`, `**kwargs`, keyword-only args)
- **Export page integration**:
  - Cached validation with mtime-based invalidation (avoids re-running on every rerender)
  - Status badges for InputSpec, OutputContract, Algorithm
  - Validation details expander with fix suggestions for snake_case issues
  - Export blocked when validation errors exist

### Other Changes
- Snapshot selector display is now unified in Browse/Selection/Export pages as "snapshot_id - imported_at" for easier management.
- Snapshot timestamps in selectors are normalized to second precision (`YYYY-MM-DD HH:MM:SS`) without ISO `T` separator.
- Browse page dataset explorer switched from single-select dropdown to dataset tabs so all dataset options are visible by default.
- Browse page dataset tabs get stronger visual affordance for click-switching; NDJSON line expanders default to expanded.
- Export page: Replace "File Status" with "Validation Status" as the primary integrity check; File Inspector in expander (expanded by default); Revalidate button inline with status badges.
- Schema/Output Edit pages: Show warnings even when validation passes; fix incorrect path passed to `validate_output_contract`.
- Single .py file import: Now creates full template structure (`schema/`, `dist/`) matching "Create New Algorithm".
- Algorithm scaffold: Now creates `dist/output_contract.json` template.
- Snapshot/Algorithm import now auto-focuses and expands the newly imported entry in the management list.
- Flatten exported algorithm zip structure (no top-level version folder in Workbench export).
- Manifest defaults: add `requires.sdk`, `repository`, `homepage`, `license` to scaffolds and bundled examples.
- Workbench Run/Results pages prompt users to use the InputSpec/OutputSpec editor Validate actions when UI or artifacts are missing.
- Workbench pages default sidebar to expanded to avoid being trapped when the Streamlit toolbar is hidden.
- Keep Streamlit toolbar visible (deploy button hidden) so the sidebar toggle is always accessible.
- Force sidebar to stay visible via CSS so navigation is always reachable.
- Hide the sidebar collapse control to keep navigation fixed open.

## 0.1.1
- Package the Streamlit Workbench inside `fraclab_sdk` with the `fraclab-workbench` entrypoint; optional `workbench` extra pulls UI deps.
- Restore core scientific deps to the base install (numpy/pandas/scipy/matplotlib/fastapi/rich) while keeping Workbench UI deps optional.
- Add manifest defaults for `requires.sdk`, `repository`, `homepage`, `license`, and embed the SDK version constant for templating.
- Update bundled example manifests to include the new metadata fields.

## 0.1.0
- Initial public SDK release with snapshots, algorithm library, run manager, CLI, and devkit tooling.
