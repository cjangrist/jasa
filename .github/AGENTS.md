# AGENTS.md — `.github/`

GitHub automation lives here. All active automation is under `workflows/`; see
`workflows/AGENTS.md` before changing a trigger, permission, action pin, cache,
artifact, release, or package-publishing behavior.

## Scope

| Path                            | Purpose                                              |
| ------------------------------- | ---------------------------------------------------- |
| `workflows/quality.yml`         | Lock, format, lint, strict types, build artifact.    |
| `workflows/unit-tests.yml`      | Unit suite, XML artifacts, exact 100% coverage gate. |
| `workflows/docker-tests.yml`    | Real Docker build and health/non-root probe.         |
| `workflows/container-image.yml` | AMD64/ARM64 GHCR build and manifest publish.         |
| `workflows/release.yml`         | Tag/version validation and GitHub Release assets.    |

## Invariants

- Actions are pinned to full commit SHAs; the comment records the friendly
  version.
- Default permissions are least privilege. Package publishing gets
  `packages: write`; releases get `contents: write`; tests remain read-only.
- Pull requests run quality, unit, and Docker jobs. Main also runs them and
  publishes the container.
- Stable Git tags are `vMAJOR.MINOR.PATCH`; `release.yml` enforces exact package
  version equality. The independent image workflow derives semver aliases and
  also publishes `latest`/full-SHA tags for `main`.
- Never place credentials or populated environment files in workflow YAML.

## Verification

Run `actionlint`, inspect the complete diff, and exercise the corresponding
local command from the root `AGENTS.md`. After opening a PR, verify every
expected check appears and finishes successfully.
