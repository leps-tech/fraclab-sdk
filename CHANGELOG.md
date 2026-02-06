# Changelog

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
- Export page: Replace "File Status" with "Validation Status" as the primary integrity check; File Inspector in expander (expanded by default); Revalidate button inline with status badges.
- Schema/Output Edit pages: Show warnings even when validation passes; fix incorrect path passed to `validate_output_contract`.
- Single .py file import: Now creates full template structure (`schema/`, `dist/`) matching "Create New Algorithm".
- Algorithm scaffold: Now creates `dist/output_contract.json` template.
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
