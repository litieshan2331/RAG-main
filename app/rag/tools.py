from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from app.core.config import Settings, get_settings
from app.rag.query_router import Strategy
from app.schemas import RetrievedDocument

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


@dataclass
class RagToolResult:
    strategy: Strategy
    tool_name: str
    content: str
    contexts: list[RetrievedDocument] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeBaseTool:
    name = "knowledge_base"

    def __init__(self, db: "Session") -> None:
        self.db = db

    def run(self, query: str, *, top_k: int, document_id: int | None = None) -> RagToolResult:
        from app.retrieval.hybrid import HybridRetriever

        hits = HybridRetriever(self.db).retrieve(query, top_k=top_k, document_id=document_id)
        contexts = [
            RetrievedDocument(text=hit.text, score=hit.score, source=hit.source, metadata=hit.metadata)
            for hit in hits
        ]
        if not contexts:
            raise RuntimeError("knowledge base returned no contexts")

        content = "\n\n".join(f"[{idx}] {item.text}" for idx, item in enumerate(contexts, start=1))
        return RagToolResult(
            strategy="knowledge_base",
            tool_name=self.name,
            content=content,
            contexts=contexts,
            metadata={"document_id": document_id, "top_k": top_k},
        )


class RelationalDbTool:
    name = "text2sql"

    def __init__(self, db: "Session") -> None:
        from app.text2sql.service import Text2SQLService

        self.service = Text2SQLService(db)

    def run(self, query: str) -> RagToolResult:
        sql, rows = self.service.execute(query)
        if not sql:
            raise RuntimeError("Text2SQL did not generate SQL")
        if get_settings().text2sql_empty_result_triggers_fallback and self.service.is_effectively_empty_result(rows):
            raise RuntimeError("Text2SQL returned no rows")

        return RagToolResult(
            strategy="relational_db",
            tool_name=self.name,
            content=f"SQL: {sql}\nResult: {json.dumps(rows, ensure_ascii=False, default=str)}",
            metadata={"generated_query": sql, "row_count": len(rows)},
        )


class GraphDbTool:
    name = "text2cypher"

    def run(self, query: str) -> RagToolResult:
        from app.text2cypher.service import Text2CypherService

        service = Text2CypherService()
        try:
            cypher, rows = service.execute(query)
        finally:
            service.close()

        if not cypher:
            raise RuntimeError("Text2Cypher did not generate Cypher")
        if get_settings().text2cypher_empty_result_triggers_fallback and not rows:
            raise RuntimeError("Text2Cypher returned no rows")

        return RagToolResult(
            strategy="graph_db",
            tool_name=self.name,
            content=f"Cypher: {cypher}\nResult: {json.dumps(rows, ensure_ascii=False, default=str)}",
            metadata={"generated_query": cypher, "row_count": len(rows)},
        )


class WebSearchTool:
    name = "web_search"

    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client

    def run(self, query: str, *, top_k: int | None = None) -> RagToolResult:
        if not self.settings.web_search_enabled:
            raise RuntimeError("web_search is disabled")
        if not self.settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is not configured")

        max_results = top_k or self.settings.tavily_max_results
        max_results = max(1, min(max_results, self.settings.tavily_max_results))
        payload = {
            "query": query,
            "search_depth": self.settings.tavily_search_depth,
            "max_results": max_results,
            "include_answer": self.settings.tavily_include_answer,
            "include_raw_content": False,
            "include_images": False,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.tavily_api_key}",
            "Content-Type": "application/json",
        }

        if self.client is not None:
            response = self.client.post(self._endpoint(), json=payload, headers=headers)
        else:
            with httpx.Client(timeout=self.settings.tavily_timeout_seconds, trust_env=False) as client:
                response = client.post(self._endpoint(), json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return self._to_result(data)

    def _endpoint(self) -> str:
        endpoint = self.settings.tavily_api_url.rstrip("/")
        if endpoint.endswith("/search"):
            return endpoint
        return f"{endpoint}/search"

    def _to_result(self, data: dict[str, Any]) -> RagToolResult:
        answer = str(data.get("answer") or "").strip()
        raw_results = data.get("results") or []
        contexts: list[RetrievedDocument] = []
        parts: list[str] = []

        if answer:
            parts.append(f"Tavily summary:\n{answer}")

        for idx, item in enumerate(raw_results, start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("url") or f"result-{idx}").strip()
            url = str(item.get("url") or "").strip()
            content = str(item.get("content") or item.get("raw_content") or "").strip()
            score = _optional_float(item.get("score"))
            source = url or "tavily"
            metadata = {
                "tool": self.name,
                "title": title,
                "url": url,
            }
            contexts.append(RetrievedDocument(text=content, score=score, source=source, metadata=metadata))
            parts.append(f"[{idx}] {title}\nURL: {source}\n{content}")

        if not parts:
            raise RuntimeError("web_search returned no usable results")

        return RagToolResult(
            strategy="web_search",
            tool_name=self.name,
            content="\n\n".join(parts),
            contexts=contexts,
            metadata={"result_count": len(contexts)},
        )


class OnlineToolExecutor:
    def __init__(self, db: "Session") -> None:
        self.knowledge_base = KnowledgeBaseTool(db)
        self.relational_db = RelationalDbTool(db)
        self.graph_db = GraphDbTool()
        self.web_search = WebSearchTool()

    def run(
        self,
        strategy: Strategy,
        query: str,
        *,
        top_k: int,
        document_id: int | None = None,
    ) -> RagToolResult:
        if strategy == "knowledge_base":
            return self.knowledge_base.run(query, top_k=top_k, document_id=document_id)
        if strategy == "relational_db":
            return self.relational_db.run(query)
        if strategy == "graph_db":
            return self.graph_db.run(query)
        if strategy == "web_search":
            return self.web_search.run(query, top_k=top_k)
        raise ValueError(f"unknown strategy: {strategy}")


def _optional_float(value: object) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
