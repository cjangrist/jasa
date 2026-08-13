# AGENTS.md — `tests/fixtures/`

Fixtures preserve behavior copied or intentionally ported from upstream source.
The current fixture family is `golden/`; see its local guide for each contract.

## Rules

- Fixtures are inputs/expected outputs, not generated test reports.
- Keep them deterministic, secret-free, and reviewable as text.
- A fixture change is a behavior change. Update the implementation and test in
  the same commit and explain why the expected contract moved.
- Do not rewrite unrelated cases when adding one scenario.
- JSON must remain valid and end with a newline.

Run the owning parity tests after any edit, then the full suite because ranking,
snippets, and URLs compose into `run_search()`.
