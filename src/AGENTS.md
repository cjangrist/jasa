# AGENTS.md — `src/`

The repository uses a `src` layout. `src/jasa/` is the only first-party package
and its directory guide is the implementation entry point.

## Packaging boundary

- Hatchling builds `src/jasa` into the wheel.
- `pyproject.toml` derives the version from `src/jasa/__init__.py`.
- Runtime imports must use the `jasa` package name; tests use importlib mode so
  an accidental repository-root import does not mask packaging mistakes.
- Package data currently includes `grounding/system_prompt.txt` because it is
  inside the package tree.
- The omnifetch implementation is not vendored here; it is a full-SHA Git
  dependency resolved into the environment.

## Navigation

Read `jasa/AGENTS.md` for startup, composition, public surfaces, subsystem maps,
and change routing. Build/install failures usually start in `pyproject.toml`,
`uv.lock`, `Dockerfile`, or `tests/test_package.py`, not in this directory.
