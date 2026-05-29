# MCAI Schema

`mcai_schema.yaml` is the single source of truth for the observation/action binary
layout shared by the Java gym (training) and the Fabric mod (inference).

- Layout: ordered, packed (no padding), little-endian. Python parses it into a NumPy
  structured dtype in `trainer/mcai_train/schema/spec.py`; that dtype is the wire format
  and the shared-memory layout.
- Golden contract: `fixtures/golden_*.json` (values) + `fixtures/golden_*.bin` (bytes).
  Every language implementation MUST reproduce the `.bin` bytes for the `.json` values.
  Regenerate the `.bin` files ONLY when intentionally bumping `schema_version`.
- Registry: `registry.sample.json` is a small fixture; the full block/item id table is
  dumped from `BuiltInRegistries` by the gym (Plan 2).

`block_visibility` (`xray` vs `legit`) is applied server-side at emit time; `legit`
replaces enclosed blocks with `params.hidden_block_id`.
