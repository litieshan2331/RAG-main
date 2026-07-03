import asyncio
from types import SimpleNamespace

from app.evaluation.ragas_evaluator import RagasEvaluator, RagasMetricBundle
from app.schemas import RagasEvaluationSample, RetrievedDocument


class _MetricResult:
    def __init__(self, value: float) -> None:
        self.value = value


class _FakeMetric:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[dict] = []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        return _MetricResult(self.value)


def _settings():
    return SimpleNamespace(
        ragas_judge_model="qwen-test-judge",
        chat_model="qwen-chat",
        ragas_max_samples=10,
    )


def test_ragas_evaluator_scores_real_answer_and_ranked_contexts() -> None:
    recall = _FakeMetric(0.75)
    precision = _FakeMetric(0.5)
    faithfulness = _FakeMetric(1.0)
    metrics = RagasMetricBundle(recall, precision, faithfulness)

    def runner(_sample):
        return {
            "answer": "后排包含安全带。",
            "route": "knowledge_base",
            "contexts": [
                RetrievedDocument(text="后排配有安全带。", score=0.9, source="manual"),
                {"text": "无关内容"},
            ],
        }

    evaluator = RagasEvaluator(runner, settings=_settings(), metrics=metrics)
    sample = RagasEvaluationSample(question="后排有什么？", reference="后排配有安全带。")

    report = asyncio.run(evaluator.evaluate([sample]))

    assert report.averages.context_recall == 0.75
    assert report.averages.context_precision == 0.5
    assert report.averages.faithfulness == 1.0
    assert report.results[0].context_count == 2
    assert precision.calls[0]["retrieved_contexts"] == ["后排配有安全带。", "无关内容"]


def test_ragas_evaluator_sets_evidence_metrics_to_zero_without_contexts() -> None:
    metric = _FakeMetric(1.0)
    evaluator = RagasEvaluator(
        lambda _sample: {"answer": "无法回答", "route": "knowledge_base", "contexts": []},
        settings=_settings(),
        metrics=RagasMetricBundle(metric, metric, metric),
    )

    report = asyncio.run(
        evaluator.evaluate([RagasEvaluationSample(question="问题", reference="参考答案")])
    )

    assert report.results[0].scores.context_recall == 0.0
    assert report.results[0].scores.context_precision == 0.0
    assert report.results[0].scores.faithfulness == 0.0
    assert report.results[0].metric_errors
    assert not metric.calls
