# schema

`mcai_schema.yaml` is the single source of truth for the obs/action binary layout
shared by the gym, the trainer and the fabric mod.

- layout is ordered, packed, little endian, python parses it into a numpy
  structured dtype in `trainer/mcgym/schema/spec.py`, that dtype is the wire
  format and the shm layout
- golden contract, `fixtures/golden_*.json` are the values, `fixtures/golden_*.bin`
  the bytes, every language implementation must reproduce the bin bytes for the
  json values, regenerate only when bumping `schema_version`
- `registry.json` maps resource names to integer ids, mapping by name keeps the
  gym, trainer and client aligned regardless of registry order
