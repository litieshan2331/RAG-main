from types import SimpleNamespace

from app.rag.context_compressor import CompressionState, ContextCompressor
from app.rag.react_agent import ReactAgent
from app.rag.tools import RagToolResult


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLlm:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def invoke(self, _prompt: str) -> _Message:
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return _Message(self.responses[index])


class _FakeTools:
    def __init__(self) -> None:
        self.calls = []

    def run(self, strategy, query: str, *, top_k: int, document_id: int | None = None) -> RagToolResult:
        self.calls.append((strategy, query, top_k, document_id))
        return RagToolResult(
            strategy=strategy,
            tool_name=str(strategy),
            content=f"{strategy} result for {query}",
            metadata={"top_k": top_k},
        )


def _settings(max_steps: int = 5):
    return SimpleNamespace(
        react_agent_max_steps=max_steps,
        react_context_max_chars=1000,
        react_observation_max_chars=300,
    )


def test_context_compressor_truncates_long_observation() -> None:
    compressor = ContextCompressor(max_total_chars=240, max_item_chars=160)

    compressed = compressor.truncate_text("x" * 200)

    assert len(compressed) < 200
    assert "compressed" in compressed


def test_context_compressor_summarizes_old_steps_and_reserves_recent_step() -> None:
    summary_llm = _FakeLlm(
        [
            '{"task_overview":"diagnose","completed_actions":"steps 1 and 2","key_findings":"evidence",'
            '"failures_and_constraints":"none","next_step":"continue","context_to_preserve":"vehicle id"}'
        ]
    )
    compressor = ContextCompressor(
        max_total_chars=2000,
        max_item_chars=500,
        max_context_tokens=400,
        trigger_ratio=0.5,
        reserve_ratio=0.25,
        summary_max_tokens=160,
        summary_model=summary_llm,
        token_counter=len,
    )
    state = CompressionState()
    observations = [
        {"step": step, "action": "knowledge_base", "action_input": f"query-{step}", "ok": True, "observation": str(step) * 140}
        for step in range(1, 4)
    ]

    compressed = compressor.compress_observations(observations, state)

    assert "<compressed-context>" in compressed
    assert "Step 3" in compressed
    assert state.summarized_steps == {"1", "2"}
    assert state.compression_count == 1
    assert state.last_report and state.last_report.triggered is True


def test_context_compressor_incrementally_merges_without_readding_old_steps() -> None:
    summary_llm = _FakeLlm(
        [
            '{"task_overview":"first","completed_actions":"1,2","key_findings":"a",'
            '"failures_and_constraints":"none","next_step":"3","context_to_preserve":"x"}',
            '{"task_overview":"merged","completed_actions":"1,2,3","key_findings":"a,b",'
            '"failures_and_constraints":"none","next_step":"4","context_to_preserve":"x"}',
        ]
    )
    compressor = ContextCompressor(
        max_total_chars=2000,
        max_item_chars=500,
        max_context_tokens=400,
        trigger_ratio=0.5,
        reserve_ratio=0.25,
        summary_max_tokens=160,
        summary_model=summary_llm,
        token_counter=len,
    )
    state = CompressionState()
    observations = [
        {"step": step, "action": "web_search", "action_input": f"query-{step}", "ok": True, "observation": str(step) * 140}
        for step in range(1, 4)
    ]
    compressor.compress_observations(observations, state)
    observations.append(
        {"step": 4, "action": "knowledge_base", "action_input": "query-4", "ok": True, "observation": "4" * 140}
    )

    compressor.compress_observations(observations, state)

    assert state.summarized_steps == {"1", "2", "3"}
    assert state.compression_count == 2
    assert "merged" in state.summary


def test_context_compressor_offloads_large_tool_result() -> None:
    compressor = ContextCompressor(
        max_item_chars=180,
        tool_result_max_tokens=100,
        token_counter=len,
    )
    state = CompressionState()

    result = compressor.compress_tool_content(
        tool_name="knowledge_base",
        content="x" * 400,
        metadata={"top_k": 5},
        state=state,
    )

    assert "offload_key=tool-result:knowledge_base:" in result
    assert len(state.offloaded_tool_results) == 1
    assert next(iter(state.offloaded_tool_results.values())) == "x" * 400


def test_context_compressor_falls_back_when_summary_model_fails() -> None:
    class _FailingLlm:
        def invoke(self, _prompt: str):
            raise RuntimeError("summary unavailable")

    compressor = ContextCompressor(
        max_total_chars=2000,
        max_item_chars=500,
        max_context_tokens=300,
        trigger_ratio=0.5,
        reserve_ratio=0.25,
        summary_model=_FailingLlm(),
        token_counter=len,
    )
    state = CompressionState()
    observations = [
        {"step": step, "action": "knowledge_base", "action_input": "query", "ok": step != 1, "error": "failed" if step == 1 else None, "observation": "x" * 120}
        for step in range(1, 4)
    ]

    compressed = compressor.compress_observations(observations, state)

    assert "<compressed-context>" in compressed
    assert "failed" in state.summary


def test_react_agent_calls_tool_then_finishes() -> None:
    llm = _FakeLlm(
        [
            '{"thought":"先查文档","action":"knowledge_base","action_input":"查故障手册","final_answer":""}',
            '{"thought":"证据足够","action":"final_answer","action_input":"","final_answer":"根据文档处理。"}',
        ]
    )
    tools = _FakeTools()
    agent = ReactAgent(tools, llm=llm, settings=_settings())

    result = agent.run("如何处理故障？", top_k=3)

    assert result.strategy == "react_agent"
    assert tools.calls[0][0] == "knowledge_base"
    assert "根据文档处理" in result.content
    assert result.metadata["steps_used"] == 2


def test_react_agent_document_id_restricts_to_knowledge_base() -> None:
    llm = _FakeLlm(['{"thought":"想查外部","action":"web_search","action_input":"查公开信息","final_answer":""}'])
    tools = _FakeTools()
    agent = ReactAgent(tools, llm=llm, settings=_settings(max_steps=1))

    agent.run("当前文档怎么说？", top_k=5, document_id=12)

    assert tools.calls[0][0] == "knowledge_base"
    assert tools.calls[0][3] == 12
