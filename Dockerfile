# Multi-stage build. The build stage resolves the omnifetch git source pin
# (network access to github.com is required at build time); the runtime stage is
# a minimal slim image running as a non-root user.
FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim@sha256:dc6831ca75771711b69e2fcaf47f2b4938bcfd7721daf254c1131791249d000d AS build

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# uv needs the system git binary to resolve the omnifetch git source pin.
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra telemetry --no-install-project

COPY . .
RUN uv sync --frozen --no-dev --extra telemetry --no-editable

FROM python:3.13-slim-trixie@sha256:eb43ff125d8d58d7449dcba7d336c23bcac412f526d861db493b9994d8010280 AS runtime

RUN useradd --create-home --uid 10001 app

WORKDIR /app

COPY --from=build --chown=app:app /app /app

# OMNIFETCH_REST_WEB_FETCH=false is the REQUIRED composed-mode default: jasa owns
# the REST fetch surface at POST /fetch; the child's mirror would otherwise land
# unauthenticated on jasa's app.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JASA_TRANSPORT=http \
    JASA_HOST=0.0.0.0 \
    JASA_PORT=8000 \
    JASA_CACHE_BACKEND=disk \
    JASA_DISK_CACHE_PATH=/home/app/.cache/jasa \
    OMNIFETCH_REST_WEB_FETCH=false

USER app

EXPOSE 8000

HEALTHCHECK --interval=5s --timeout=3s --start-period=10s --retries=12 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

ENTRYPOINT ["jasa"]
CMD ["--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
