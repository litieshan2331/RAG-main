from types import SimpleNamespace

from fastapi.testclient import TestClient

import app.api.routes as routes
from app.api.routes import _encode_sse
from app.main import app
from app.rag.conversation import ContextDependencyDecision
from app.rag.graph import OnlineRagGraph
from app.rag.query_router import QueryRouteDecision
from app.schemas import RetrievedDocument


class _FakeCompiledGraph:
    def stream(self, *_args, **_kwargs):
        decision = QueryRouteDecision("实时信息", "web_search", "需要联网", 0.95)
        context = RetrievedDocument(
            text="search result",
            score=0.9,
            source="https://example.com",
            metadata={"title": "Example", "url": "https://example.com"},
        )
        yield {"type": "updates", "data": {"route": {"route_decision": decision, "fallback_routes": ["web_search"]}}}
        yield {"type": "tasks", "data": {"id": "task-1", "name": "web_search", "input": {}}}
        yield {
            "type": "updates",
            "data": {
                "web_search": {
                    "route": "web_search",
                    "contexts": [context],
                    "tool_trace": [{"strategy": "web_search", "ok": True, "metadata": {"result_count": 1}}],
                }
            },
        }
        yield {
            "type": "messages",
            "data": (SimpleNamespace(content="hidden"), {"langgraph_node": "route"}),
        }
        yield {
            "type": "messages",
            "data": (SimpleNamespace(content="hello"), {"langgraph_node": "answer"}),
        }
        yield {
            "type": "values",
            "data": {
                "query": "today",
                "rewritten_query": "today",
                "context_decision": ContextDependencyDecision(False, "independent", 1.0),
                "route": "web_search",
                "route_decision": decision,
                "answer": "hello",
                "contexts": [context],
                "tool_trace": [{"strategy": "web_search", "ok": True}],
            },
        }


def test_graph_stream_emits_tool_tokens_citations_and_completion() -> None:
    graph = object.__new__(OnlineRagGraph)
    graph.graph = _FakeCompiledGraph()

    events = list(graph.stream_answer("today", "user", 3, conversation_id="conversation-1"))
    names = [event["event"] for event in events]
    tokens = [event["data"]["content"] for event in events if event["event"] == "token"]

    assert names == ["started", "route", "tool_started", "tool_finished", "status", "token", "citations", "completed"]
    assert tokens == ["hello"]
    assert events[-1]["data"]["conversation_id"] == "conversation-1"


def test_sse_encoder_preserves_event_name_and_unicode() -> None:
    encoded = _encode_sse("status", {"message": "正在调用混合检索"})

    assert encoded.startswith("event: status\n")
    assert 'data: {"message":"正在调用混合检索"}\n\n' in encoded


def test_stream_endpoint_returns_sse(monkeypatch) -> None:
    class _FakeRagService:
        def __init__(self, _db):
            pass

        def stream_answer(self, *_args, **_kwargs):
            yield {"event": "started", "data": {"message": "thinking"}}
            yield {
                "event": "completed",
                "data": {
                    "query": "hello",
                    "conversation_id": "c1",
                    "rewritten_query": "hello",
                    "context_decision": {},
                    "route": "knowledge_base",
                    "route_decision": {},
                    "answer": "world",
                    "contexts": [],
                    "tool_trace": [],
                },
            }

    monkeypatch.setattr(routes, "RagService", _FakeRagService)
    client = TestClient(app)
    response = client.post("/api/chat/rag/stream", json={"query": "hello"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: started" in response.text
    assert "event: completed" in response.text
