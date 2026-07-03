from __future__ import annotations

import asyncio
import importlib
import math
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from app.core.config import Settings, get_settings
from app.schemas import (
    RagasEvaluationItem,
    RagasEvaluationResponse,
    RagasEvaluationSample,
    RagasMetricScores,
)


class RagasDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class RagasMetricBundle:
    context_recall: Any
    context_precision: Any
    faithfulness: Any


class RagasEvaluator:
    """Evaluate the real RAG answer and ranked contexts with RAGAS metrics."""

    def __init__(
        self,
        rag_runner: Callable[[RagasEvaluationSample], dict[str, Any]],
        *,
        settings: Settings | None = None,
        metrics: RagasMetricBundle | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.rag_runner = rag_runner
        self.judge_model = self.settings.ragas_judge_model or self.settings.chat_model
        self.metrics = metrics or self._build_metrics()

    def evaluate_sync(self, samples: list[RagasEvaluationSample]) -> RagasEvaluationResponse:
        if len(samples) > self.settings.ragas_max_samples:
            raise ValueError(f"RAGAS evaluation accepts at most {self.settings.ragas_max_samples} samples per request")
        return asyncio.run(self.evaluate(samples))

    async def evaluate(self, samples: list[RagasEvaluationSample]) -> RagasEvaluationResponse:
        results: list[RagasEvaluationItem] = []
        for index, sample in enumerate(samples, start=1):
            results.append(await self._evaluate_sample(sample, index))

        return RagasEvaluationResponse(
            sample_count=len(results),
            successful_samples=sum(not item.metric_errors for item in results),
            judge_model=self.judge_model,
            averages=RagasMetricScores(
                context_recall=self._average(results, "context_recall"),
                context_precision=self._average(results, "context_precision"),
                faithfulness=self._average(results, "faithfulness"),
            ),
            results=results,
        )

    async def _evaluate_sample(self, sample: RagasEvaluationSample, index: int) -> RagasEvaluationItem:
        rag_result = self.rag_runner(sample)
        answer = str(rag_result.get("answer") or "").strip()
        contexts = self._context_texts(rag_result.get("contexts"))
        route = str(rag_result.get("route") or "unknown")

        if not contexts:
            return RagasEvaluationItem(
                sample_id=sample.sample_id or str(index),
                question=sample.question,
                reference=sample.reference,
                answer=answer,
                route=route,
                context_count=0,
                scores=RagasMetricScores(context_recall=0.0, context_precision=0.0, faithfulness=0.0),
                metric_errors={"contexts": "RAG returned no contexts; all evidence-based metrics were set to 0"},
            )

        calls = {
            "context_recall": self.metrics.context_recall.ascore(
                user_input=sample.question,
                reference=sample.reference,
                retrieved_contexts=contexts,
            ),
            "context_precision": self.metrics.context_precision.ascore(
                user_input=sample.question,
                reference=sample.reference,
                retrieved_contexts=contexts,
            ),
        }
        if answer:
            calls["faithfulness"] = self.metrics.faithfulness.ascore(
                user_input=sample.question,
                response=answer,
                retrieved_contexts=contexts,
            )

        values: dict[str, float | None] = {
            "context_recall": None,
            "context_precision": None,
            "faithfulness": 0.0 if not answer else None,
        }
        errors: dict[str, str] = {}
        names = list(calls)
        outcomes = await asyncio.gather(*calls.values(), return_exceptions=True)
        for name, outcome in zip(names, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                errors[name] = str(outcome)
                continue
            values[name] = self._metric_value(outcome)
            if values[name] is None:
                errors[name] = "RAGAS returned a non-finite score"
        if not answer:
            errors["faithfulness"] = "RAG returned an empty answer; faithfulness was set to 0"

        return RagasEvaluationItem(
            sample_id=sample.sample_id or str(index),
            question=sample.question,
            reference=sample.reference,
            answer=answer,
            route=route,
            context_count=len(contexts),
            scores=RagasMetricScores(**values),
            metric_errors=errors,
        )

    def _build_metrics(self) -> RagasMetricBundle:
        try:
            from openai import AsyncOpenAI

            self._install_ragas_vertexai_compat()
            from ragas.llms import llm_factory
            from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness
        except ImportError as exc:
            raise RagasDependencyError(
                "RAGAS evaluation dependencies are not installed. Run: uv sync --extra evaluation"
            ) from exc

        client = AsyncOpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            timeout=self.settings.ragas_timeout_seconds,
        )
        judge_llm = llm_factory(
            self.judge_model,
            provider="openai",
            client=client,
            temperature=0.01,
            max_tokens=2_048,
        )
        return RagasMetricBundle(
            context_recall=ContextRecall(llm=judge_llm),
            context_precision=ContextPrecision(llm=judge_llm),
            faithfulness=Faithfulness(llm=judge_llm),
        )

    @staticmethod
    def _install_ragas_vertexai_compat() -> None:
        """Work around RAGAS 0.4.3 importing a class removed by LangChain 0.4.

        The evaluation extra also pins Instructor 1.13 because RAGAS 0.4.3 uses
        its pre-1.14 structured-output client contract.
        """
        module_name = "langchain_community.chat_models.vertexai"
        if module_name in sys.modules:
            return
        try:
            importlib.import_module(module_name)
            return
        except ModuleNotFoundError:
            pass

        compatibility_module = types.ModuleType(module_name)
        compatibility_module.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[module_name] = compatibility_module

    @staticmethod
    def _context_texts(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        texts: list[str] = []
        for item in value:
            if hasattr(item, "text"):
                text = str(getattr(item, "text") or "").strip()
            elif isinstance(item, dict):
                text = str(item.get("text") or item.get("content") or "").strip()
            else:
                text = str(item or "").strip()
            if text:
                texts.append(text)
        return texts

    @staticmethod
    def _metric_value(result: object) -> float | None:
        raw_value = getattr(result, "value", result)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    @staticmethod
    def _average(results: list[RagasEvaluationItem], metric_name: str) -> float | None:
        values = [getattr(item.scores, metric_name) for item in results]
        valid = [value for value in values if value is not None]
        return round(fmean(valid), 6) if valid else None
