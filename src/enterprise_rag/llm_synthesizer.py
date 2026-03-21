from __future__ import annotations

import os
from abc import ABC, abstractmethod

from enterprise_rag.models import SearchResult


class SynthesizerBase(ABC):
    @abstractmethod
    def synthesize(self, query: str, results: list[SearchResult]) -> str: ...

    @property
    @abstractmethod
    def label(self) -> str: ...


class TemplateSynthesizer(SynthesizerBase):
    """Offline answer synthesis — no API key required."""

    label = "Template (offline)"

    def synthesize(self, query: str, results: list[SearchResult]) -> str:
        if not results:
            return "No matching products found for this query."
        top = results[0].item
        others = [r.item for r in results[1:3]]
        other_names = " and ".join(
            f"**{item.name}** [{j + 2}]" for j, item in enumerate(others)
        )
        answer = (
            f"Based on the query **\"{query}\"**, the best match is **{top.name}** [1] "
            f"({top.sku}, confidence {results[0].score:.0%}) — {top.description}"
        )
        if other_names:
            answer += f" You may also consider {other_names}."
        return answer


class OpenAISynthesizer(SynthesizerBase):
    """LLM-backed answer synthesis using OpenAI chat completions."""

    label = "GPT-4o-mini (OpenAI)"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model

    def synthesize(self, query: str, results: list[SearchResult]) -> str:
        if not results:
            return "No matching products found for this query."
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "openai is not installed. Install it with: pip install openai"
            ) from exc

        citations = "\n".join(
            f"[{i + 1}] {r.item.name} ({r.item.sku}, confidence={r.score:.0%}): {r.item.description}"
            for i, r in enumerate(results[:3])
        )
        prompt = (
            "You are a Sales Engineer AI helping a customer find the right product.\n\n"
            f"Customer requirements: {query}\n\n"
            f"Top matching products:\n{citations}\n\n"
            "Write a 2-3 sentence recommendation that maps their requirements to these "
            "products. Reference each product by its number in brackets like [1] and "
            "mention its confidence score."
        )
        client = OpenAI(api_key=self.api_key)

        import time as _time  # noqa: PLC0415
        t0 = _time.perf_counter()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=220,
            temperature=0.3,
        )
        elapsed = _time.perf_counter() - t0
        result_text = response.choices[0].message.content.strip()

        try:
            from enterprise_rag.telemetry import langfuse_log_generation  # noqa: PLC0415

            usage = response.usage
            langfuse_log_generation(
                "llm_synthesis",
                model=self.model,
                input=[{"role": "user", "content": prompt}],
                output=result_text,
                usage_input=usage.prompt_tokens if usage else 0,
                usage_output=usage.completion_tokens if usage else 0,
                metadata={"latency_ms": round(elapsed * 1000, 1), "query": query},
            )
        except Exception:
            pass

        return result_text


def get_synthesizer(api_key: str = "") -> SynthesizerBase:
    """Return the best available synthesizer given the environment."""
    if api_key:
        return OpenAISynthesizer(api_key)
    return TemplateSynthesizer()


def get_synthesizer_from_env() -> SynthesizerBase:
    return get_synthesizer(os.environ.get("OPENAI_API_KEY", ""))
