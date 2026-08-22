"""Ordered, swappable grounding LLM waterfall loaded from packaged YAML.

Grounding pays for a page fetch before it can call an LLM, so a single
rate-limited or unavailable endpoint used to discard work that was already
billed. The waterfall spends that fetch once and walks an ordered chain of
OpenAI-compatible chat-completions endpoints until one returns usable text.

Tiers are declared in ``waterfall.yaml``, or in the file named by
``JASA_GROUNDING_WATERFALL_PATH``, so an operator swaps a provider by editing
configuration rather than code. An omitted per-tier field inherits the matching
``JASA_GROUNDING_LLM_*`` setting, which keeps the first tier under environment
control.

A tier never carries a credential. It names the environment variable holding
its key, and resolution produces the credentialed chain and its keys as two
separate values, so cache-identity and fingerprint code cannot reach a secret.
The file is read once at composition, but credentials are resolved per request,
so a key exported after boot joins the chain on the next search. A malformed or
unreadable file fails startup rather than silently disabling grounding.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import cast, Literal
from urllib.parse import SplitResult, urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from jasa.config import GroundingSettings
from omnifetch.fetch.shared.util import validate_api_key

WATERFALL_SCHEMA_VERSION: Literal[1] = 1
MIN_TIER_TIMEOUT_MS = 1000
_MAX_NAME_CHARS = 64
_MAX_ENV_NAME_CHARS = 128
_MAX_URL_CHARS = 2000
_HTTP_SCHEMES = frozenset({"http", "https"})
_URL_TAIL_DELIMITERS = ("?", "#")
_INHERITED_ORIGIN = "inherited from JASA_GROUNDING_LLM_BASE_URL"
_FILE_ORIGIN = "set in the waterfall file"
_PACKAGED_WATERFALL = Path(__file__).resolve().parent / "waterfall.yaml"
_STRICT_DOCUMENT_CONFIG = ConfigDict(extra="forbid", strict=True, frozen=True)


@dataclass(frozen=True, slots=True)
class GroundingTier:
    """One ordered, credential-free endpoint in the grounding waterfall."""

    name: str
    base_url: str
    model: str
    timeout_ms: int
    api_key_env: str


GroundingChain = tuple[GroundingTier, ...]


@dataclass(frozen=True, slots=True)
class ResolvedGroundingWaterfall:
    """The credentialed chain and the keys its tiers need, kept apart."""

    chain: GroundingChain
    api_keys: Mapping[str, str] = field(repr=False)


class _WaterfallTierDocument(BaseModel):
    """One strictly validated tier entry as written in the YAML file."""

    model_config = _STRICT_DOCUMENT_CONFIG

    name: str = Field(min_length=1, max_length=_MAX_NAME_CHARS)
    api_key_env: str = Field(min_length=1, max_length=_MAX_ENV_NAME_CHARS)
    base_url: str | None = Field(
        default=None, min_length=1, max_length=_MAX_URL_CHARS
    )
    model: str | None = Field(
        default=None, min_length=1, max_length=_MAX_NAME_CHARS
    )
    timeout_ms: int | None = Field(default=None, ge=MIN_TIER_TIMEOUT_MS)


class _WaterfallDocument(BaseModel):
    """The strict versioned envelope of the whole waterfall file."""

    model_config = _STRICT_DOCUMENT_CONFIG

    version: Literal[1]
    tiers: list[_WaterfallTierDocument] = Field(min_length=1)


def waterfall_path(config: GroundingSettings) -> Path:
    """Return the configured waterfall file, or the packaged default."""
    configured = config.waterfall_path.strip()
    if configured:
        return Path(configured)
    return _PACKAGED_WATERFALL


def _read_waterfall_document(path: Path) -> _WaterfallDocument:
    """Read and strictly validate one waterfall file, or fail startup."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(
            f"grounding waterfall {path} could not be read: "
            f"{type(error).__name__}"
        ) from error
    try:
        return _WaterfallDocument.model_validate(raw)
    except ValidationError as error:
        raise ValueError(
            f"grounding waterfall {path} is not a valid v"
            f"{WATERFALL_SCHEMA_VERSION} document: {error.error_count()} "
            "problems"
        ) from error


def _has_reachable_authority(parsed: SplitResult) -> bool:
    """Report whether the authority names a host and a usable port.

    ``port`` is a property that raises rather than returning a value for a
    non-numeric or out-of-range port, so it is read here instead of at request
    time, and ``hostname`` is checked in place of the raw ``netloc`` because an
    authority such as ``:443`` is non-empty while naming no host at all.

    Port zero parses cleanly but is a bind-time wildcard rather than something
    a client can connect to, so it is rejected alongside the ports that raise.
    An absent port stays valid and defers to the scheme.
    """
    try:
        port = parsed.port
    except ValueError:
        return False
    return bool(parsed.hostname) and port != 0


def _validated_base_url(
    tier_name: str, base_url: str, *, inherited: bool
) -> str:
    """Reject at startup an endpoint no request could ever reach.

    The effective value is checked rather than the written one, because an
    omitted ``base_url`` inherits ``JASA_GROUNDING_LLM_BASE_URL`` and a
    misconfigured setting must fail just as loudly as a misconfigured file.

    A query or fragment is rejected as well. The request target is built by
    appending ``/chat/completions``, so either component would swallow that
    suffix instead of letting it extend the path. The delimiters are matched
    literally rather than through the parsed components, because a trailing
    ``?`` or ``#`` parses to an empty component while still diverting
    everything appended after it.

    Userinfo is rejected too: a credential belongs in ``api_key_env``, where
    resolution keeps it out of the cache identity and the fingerprint, not
    inline in a URL that those code paths hash.

    No rejection message repeats the URL. The values most likely to be rejected
    are the ones carrying userinfo or a query string, so echoing them would
    write the very credential this function exists to refuse into whatever
    reads a failed startup. Each message names the tier and where the value
    came from instead: the effective URL merges the file entry with an
    inherited setting, so without the origin a typo in the environment and a
    typo in the file are indistinguishable. A variable's name is not a secret.
    """
    origin = _INHERITED_ORIGIN if inherited else _FILE_ORIGIN
    parsed = urlsplit(base_url)
    if parsed.scheme not in _HTTP_SCHEMES or not _has_reachable_authority(
        parsed
    ):
        raise ValueError(
            f"grounding waterfall tier {tier_name!r} needs an absolute "
            f"http(s) base_url ({origin})"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(
            f"grounding waterfall tier {tier_name!r} must carry its credential "
            f"in api_key_env, not in base_url ({origin})"
        )
    if any(delimiter in base_url for delimiter in _URL_TAIL_DELIMITERS):
        raise ValueError(
            f"grounding waterfall tier {tier_name!r} needs a base_url with no "
            f"query or fragment ({origin})"
        )
    return base_url.rstrip("/")


def _build_tier(
    document: _WaterfallTierDocument, config: GroundingSettings
) -> GroundingTier:
    """Apply the JASA_GROUNDING_LLM_* inheritance to one tier entry."""
    return GroundingTier(
        name=document.name,
        base_url=_validated_base_url(
            document.name,
            document.base_url or config.llm_base_url,
            inherited=document.base_url is None,
        ),
        model=document.model or config.llm_model,
        timeout_ms=document.timeout_ms or config.llm_timeout_ms,
        api_key_env=document.api_key_env,
    )


def load_grounding_waterfall(config: GroundingSettings) -> GroundingChain:
    """Load the ordered chain declared for this configuration."""
    document = _read_waterfall_document(waterfall_path(config))
    return tuple(_build_tier(tier, config) for tier in document.tiers)


def _normalized_credential(raw: str) -> str:
    """Return one credential in the same shape every other reader expects.

    Normalization is delegated to the helper the search providers already use
    rather than reimplemented, so the two paths cannot drift. They previously
    disagreed about the same environment variable: an ``.env`` entry written
    ``KEY="abc"`` reached the providers as ``abc`` and the grounding waterfall
    as ``"abc"``. Compose passes such a file through verbatim, so the
    disagreement surfaced only under Docker, as a tier that authenticated
    everywhere else and returned 401 here on every call -- burning the first
    tier of the chain on every single grounded URL.

    An empty value is answered directly. ``validate_api_key`` raises for a
    missing credential because a provider asked to run without one has failed,
    whereas an uncredentialed tier is simply not part of this request's chain.
    """
    if not raw.strip():
        return ""
    return cast("str", validate_api_key(raw, "grounding"))


def resolve_grounding_waterfall(
    chain: GroundingChain, environ: Mapping[str, str]
) -> ResolvedGroundingWaterfall:
    """Drop uncredentialed tiers and snapshot the keys the rest need."""
    resolved = {
        tier.api_key_env: _normalized_credential(
            environ.get(tier.api_key_env, "")
        )
        for tier in chain
    }
    credentialed = tuple(tier for tier in chain if resolved[tier.api_key_env])
    api_keys = {
        tier.api_key_env: resolved[tier.api_key_env] for tier in credentialed
    }
    return ResolvedGroundingWaterfall(
        chain=credentialed, api_keys=MappingProxyType(api_keys)
    )


def grounding_chain_semantics(
    chain: GroundingChain,
) -> tuple[tuple[str, str], ...]:
    """Return the ordered ``(base_url, model)`` identity of an exact chain.

    Names and timeouts are excluded: relabelling a tier or retuning its budget
    cannot change the text an accepted snippet contains, so neither may
    invalidate cached output.
    """
    return tuple((tier.base_url, tier.model) for tier in chain)


def grounding_credential_envs(chain: GroundingChain) -> tuple[str, ...]:
    """Return the distinct credential names the chain can be enabled by."""
    return tuple(dict.fromkeys(tier.api_key_env for tier in chain))
