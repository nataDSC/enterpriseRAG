"""Optional observability integrations — all are no-ops when not configured.

Integrations:
  - Sentry          — error/exception tracking
  - OpenTelemetry   — distributed tracing → Grafana Cloud, Jaeger, Tempo, etc.
  - Langfuse        — LLM call tracing, token/cost visibility

All integrations activate only when the required env vars are present AND
the relevant package is installed. Safe to import unconditionally.
"""
from __future__ import annotations

import os
from typing import Any
from urllib.parse import unquote


# ---------------------------------------------------------------------------
# Sentry
# ---------------------------------------------------------------------------

def init_sentry() -> bool:
    """Initialise Sentry SDK. Returns True if successfully activated."""
    dsn = os.environ.get("SENTRY_DSN", "")
    if not dsn:
        return False
    try:
        import sentry_sdk  # noqa: PLC0415

        sentry_sdk.init(
            dsn=dsn,
            traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.2")),
            environment=os.environ.get("APP_ENV", "development"),
            release=os.environ.get("APP_VERSION", "dev"),
        )
        return True
    except Exception:  # missing package or invalid DSN
        return False


def capture_exception(exc: Exception) -> None:
    """Send exception to Sentry if configured, otherwise no-op."""
    try:
        import sentry_sdk  # noqa: PLC0415

        sentry_sdk.capture_exception(exc)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# OpenTelemetry
# ---------------------------------------------------------------------------

class _NoopSpan:
    """Drop-in replacement when OTEL is not available."""
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def set_attribute(self, *a, **kw): pass
    def record_exception(self, *a, **kw): pass
    def set_status(self, *a, **kw): pass


class _NoopTracer:
    def start_as_current_span(self, name: str, **kw):
        from contextlib import contextmanager  # noqa: PLC0415

        @contextmanager
        def _noop():
            yield _NoopSpan()

        return _noop()


def init_otel() -> bool:
    """Initialise OpenTelemetry with OTLP HTTP exporter. Returns True if activated."""
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if not endpoint:
        return False
    try:
        from opentelemetry import trace  # noqa: PLC0415
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: PLC0415
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # noqa: PLC0415

        # Parse optional headers (format: Key1=Val1,Key2=Val2)
        headers: dict[str, str] = {}
        for pair in os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "").split(","):
            if "=" in pair:
                k, _, v = pair.partition("=")
            headers[k.strip()] = unquote(v.strip())

        resource = Resource({SERVICE_NAME: os.environ.get("OTEL_SERVICE_NAME", "enterprise-rag")})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(
            endpoint=endpoint.rstrip("/") + "/v1/traces",
            headers=headers,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        return True
    except Exception:
        return False


def get_tracer():
    """Return an OTEL tracer, or a no-op tracer if OTEL is not available."""
    try:
        from opentelemetry import trace  # noqa: PLC0415

        return trace.get_tracer("enterprise_rag")
    except Exception:
        return _NoopTracer()


# ---------------------------------------------------------------------------
# Langfuse
# ---------------------------------------------------------------------------

def get_langfuse():
    """Return a Langfuse client if configured, otherwise None."""
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    sk = os.environ.get("LANGFUSE_SECRET_KEY", "")
    if not pk or not sk:
        return None
    try:
        from langfuse import Langfuse  # noqa: PLC0415

        return Langfuse(
            public_key=pk,
            secret_key=sk,
            host=(
                os.environ.get("LANGFUSE_HOST")
                or os.environ.get("LANGFUSE_BASE_URL")
                or "https://cloud.langfuse.com"
            ),
        )
    except Exception:
        return None


def langfuse_log_event(
    name: str,
    *,
    input: Any = None,
    output: Any = None,
    metadata: Any = None,
) -> bool:
    """Log a generic Langfuse event across SDK versions.

    Newer SDKs expose `create_event`; older SDKs may use `trace(...).generation(...)`.
    """
    lf = get_langfuse()
    if not lf:
        return False

    # Newer SDK
    if hasattr(lf, "create_event"):
        try:
            lf.create_event(name=name, input=input, output=output, metadata=metadata)
            lf.flush()
            return True
        except Exception:
            return False

    # Older SDK fallback
    if hasattr(lf, "trace"):
        try:
            trace = lf.trace(name=name, input=input)
            generation = trace.generation(name=name, model="system", input=input)
            generation.end(output=output, metadata=metadata)
            lf.flush()
            return True
        except Exception:
            return False

    return False


def langfuse_log_generation(
    name: str,
    *,
    model: str,
    input: Any,
    output: Any,
    usage_input: int = 0,
    usage_output: int = 0,
    metadata: Any = None,
) -> bool:
    """Log an LLM generation to Langfuse across SDK versions."""
    lf = get_langfuse()
    if not lf:
        return False

    # Newer SDK: start_observation(as_type='generation')
    if hasattr(lf, "start_observation"):
        try:
            obs = lf.start_observation(
                name=name,
                as_type="generation",
                model=model,
                input=input,
                metadata=metadata,
            )
            if hasattr(obs, "end"):
                obs.end(output=output, usage_details={"input": usage_input, "output": usage_output})
            lf.flush()
            return True
        except Exception:
            return False

    # Older SDK fallback
    if hasattr(lf, "trace"):
        try:
            trace = lf.trace(name=name, input=input)
            generation = trace.generation(name=name, model=model, input=input)
            generation.end(
                output=output,
                usage={"input": usage_input, "output": usage_output},
                metadata=metadata,
            )
            lf.flush()
            return True
        except Exception:
            return False

    return False


# ---------------------------------------------------------------------------
# One-call bootstrapper (call once at app startup)
# ---------------------------------------------------------------------------

def init_all() -> dict[str, bool]:
    """Initialise all configured integrations. Returns status dict."""
    return {
        "sentry": init_sentry(),
        "otel": init_otel(),
        "langfuse": get_langfuse() is not None,
    }
