# AGENTS.md — `.github/workflows/`

Five workflows split fast correctness, full unit coverage, container runtime,
image publishing, and releases. Keep that separation: a failure should identify
the broken layer without reading a monolithic job.

## Workflow matrix

| File                  | Trigger                    | Outputs / guarantees                                                        |
| --------------------- | -------------------------- | --------------------------------------------------------------------------- |
| `quality.yml`         | PR, push to `main`         | Frozen lock, Ruff format/lint, strict mypy, wheel/sdist artifact.           |
| `unit-tests.yml`      | PR, push to `main`         | Pytest XML and coverage XML; downstream job requires every line and branch. |
| `docker-tests.yml`    | PR, push to `main`         | Builds the real image, probes `/health`, confirms UID 10001.                |
| `container-image.yml` | push to `main` or `v*.*.*` | Per-arch digest builds and combined GHCR manifest.                          |
| `release.yml`         | `v*.*.*` tag               | Version/tag equality, distribution build, GitHub Release.                   |

## Container publishing details

`build-container` runs separate native AMD64 and ARM64 jobs and uploads digest
markers. `publish-container` downloads both and creates one manifest. Main
publishes `latest` and `sha-<full commit>`; release tags publish full/minor/major
semver aliases and stable `latest`. SHA tags are enabled only on `refs/heads/main`
so a later tag build cannot mutate an immutable main-build tag.

## Safe edits

- Preserve full-SHA action pins.
- Preserve `--frozen` dependency installation in CI.
- Preserve the explicit coverage artifact handoff; pytest's terminal threshold
  alone is not the branch-coverage gate.
- Preserve multi-architecture native runners and digest-based manifest creation.
- Keep concurrency keyed by workflow and ref with stale-run cancellation.
- Do not broaden permissions to solve an unrelated failure.

## Tests

```bash
actionlint
conda run -n base uv run pre-commit run --all-files
JASA_RUN_DOCKER_TESTS=1 conda run -n base uv run pytest \
  -m docker_integration --no-cov
```

For publishing edits, also inspect the generated tags in the completed Actions
run and `docker buildx imagetools inspect` the published manifest.
