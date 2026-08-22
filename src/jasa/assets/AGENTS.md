# AGENTS.md — `src/jasa/assets/`

Packaged brand images. These are data, not code: `../assets.py` reads them and
builds both the `serverInfo.icons` declaration and the bytes the HTTP routes
serve, so the declared icon and the served icon can never disagree.

## Files

| File             | Contents                                                     |
| ---------------- | ------------------------------------------------------------ |
| `icon-48.png`    | 48×48 square; the size inlined as a `data:` URI by default.  |
| `icon-128.png`   | 128×128 square.                                              |
| `icon-256.png`   | 256×256 square; what `/icon.png` serves without a `?size=`.  |
| `favicon.ico`    | Multi-resolution ICO, 16 through 256.                        |

## Invariants

- Every size named in `assets.py` must exist here. `tests/test_assets.py`
  asserts each declared size is a real PNG, because a declared size a client
  asks for and does not receive is worse than one never offered.
- Keep the art square, full-bleed, and opaque. Clients composite an icon onto
  their own background, and a transparent PNG renders as a white tile rather
  than as the shape it was drawn as.
- Regenerate every size from one source image in the same pass, so they cannot
  drift into showing different artwork at different sizes:

  ```bash
  convert icon-source.png -resize 256x256 src/jasa/assets/icon-256.png
  convert icon-source.png -resize 128x128 src/jasa/assets/icon-128.png
  convert icon-source.png -resize 48x48   src/jasa/assets/icon-48.png
  convert icon-source.png \
    -define icon:auto-resize=16,32,48,64,128,256 src/jasa/assets/favicon.ico
  ```

- Adding a size means updating `_DECLARED_SIZES` in `../assets.py`, the route
  handler's size map in `../server.py`, and the parametrized tests together.
- Weigh the inlined size against the wire. The `data:` URI rides on every
  `initialize`, so the smallest square is the one inlined; the larger ones are
  offered as links only when `JASA_PUBLIC_URL` names an origin to serve them
  from.
- Hatchling ships every git-tracked non-Python file under `src/jasa/`, so a new
  asset needs `git add` and nothing in `pyproject.toml`. Verify with
  `uv build --wheel` and `unzip -l` rather than assuming.
