from types import SimpleNamespace

from app.rag.graph import OnlineRagGraph


def _graph_stub(react_enabled: bool = True):
    graph = object.__new__(OnlineRagGraph)
    graph.settings = SimpleNamespace(react_agent_enabled=react_enabled, react_agent_confidence_threshold=0.55)
    return graph


def test_graph_routes_strategy_to_explicit_tool_node() -> None:
    assert OnlineRagGraph._next_tool_node({"fallback_routes": ["knowledge_base"], "attempt_index": 0}) == "hybrid_retrieval"
    assert OnlineRagGraph._next_tool_node({"fallback_routes": ["relational_db"], "attempt_index": 0}) == "text2sql"
    assert OnlineRagGraph._next_tool_node({"fallback_routes": ["graph_db"], "attempt_index": 0}) == "text2cypher"
    assert OnlineRagGraph._next_tool_node({"fallback_routes": ["web_search"], "attempt_index": 0}) == "web_search"
    assert OnlineRagGraph._next_tool_node({"fallback_routes": ["react_agent"], "attempt_index": 0}) == "react_agent"


def test_graph_failed_tool_enters_react_agent_before_static_fallback() -> None:
    graph = _graph_stub()
    state = {
        "fallback_routes": ["graph_db", "knowledge_base", "relational_db"],
        "attempt_index": 1,
        "react_agent_attempted": False,
        "force_knowledge_base": False,
    }

    assert graph._after_tool(state) == "react_agent"


def test_graph_fallback_moves_to_next_explicit_tool_node_when_react_disabled() -> None:
    graph = _graph_stub(react_enabled=False)
    state = {
        "fallback_routes": ["graph_db", "knowledge_base", "relational_db"],
        "attempt_index": 1,
        "react_agent_attempted": False,
        "force_knowledge_base": False,
    }

    assert graph._after_tool(state) == "hybrid_retrieval"


def test_graph_goes_to_answer_after_success_or_exhaustion() -> None:
    graph = _graph_stub()

    assert graph._after_tool({"tool_result": "ok"}) == "answer"
    assert OnlineRagGraph._next_tool_node({"fallback_routes": ["graph_db"], "attempt_index": 1}) == "answer"


def test_graph_routes_realtime_query_to_web_when_router_llm_is_unavailable() -> None:
    decision = OnlineRagGraph._fallback_route_decision("今天是星期几？", force_knowledge_base=False)

    assert decision.strategy == "web_search"
    assert decision.confidence >= 0.55


def test_graph_detects_explicit_web_queries_before_llm_routing() -> None:
    assert OnlineRagGraph._is_explicit_web_query("今天是星期几？") is True
    assert OnlineRagGraph._is_explicit_web_query("What is today's date?") is True
    assert OnlineRagGraph._is_explicit_web_query("如何打开零重力座椅？") is False


def test_graph_returns_tool_context_when_answer_llm_is_unavailable() -> None:
    answer = OnlineRagGraph._fallback_answer("web_search", "Tavily summary: today is Sunday")

    assert "联网搜索结果" in answer
    assert "today is Sunday" in answer
