# Vision Lifecycle contracts

`v1/` contains the versioned JSON Schema contracts shared by API, CLI and the onboarding skill. The same definitions are available at `GET /api/v1/schemas` and with `visionops schemas --output contracts.json`.

`result-manifest.schema.json` describes the outer `{ "manifest": ... }` import request. The manifest itself keeps the required `schema_version`, `external_run_id`, `kind`, and `name` fields while allowing typed `config`, `metrics`, `details`, and `environment` values.

Add a new directory for incompatible contract changes. Do not silently change a published schema; update the adapter and fixture tests together.
