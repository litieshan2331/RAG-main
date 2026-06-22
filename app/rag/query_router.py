from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from app.core.json_utils import fix_json


Strategy = Literal["knowledge_base", "relational_db", "graph_db", "web_search", "react_agent"]
STRATEGIES: tuple[Strategy, ...] = (
    "knowledge_base",
    "relational_db",
    "graph_db",
    "web_search",
    "react_agent",
)
SIMPLE_STRATEGIES: tuple[Strategy, ...] = (
    "knowledge_base",
    "relational_db",
    "graph_db",
    "web_search",
)

PROMPT_PATH = Path(__file__).resolve().parents[1] / "resources" / "prompts" / "query_router_prompt.txt"

# A graph query must express graph semantics. Merely mentioning an entity such as
# a part, model, system, or module is not sufficient evidence for Text2Cypher.
GRAPH_SEMANTIC_MARKERS = (
    "关系",
    "关联",
    "联系",
    "连接",
    "路径",
    "链路",
    "最短路",
    "最短路径",
    "依赖",
    "层级",
    "网络",
    "拓扑",
    "上下游",
    "供应链",
    "供应商",
    "影响链",
    "影响路径",
    "组成关系",
    "隶属关系",
    "适配关系",
    "兼容关系",
    "搭载了",
    "搭载哪些",
    "哪些车型搭载",
    "谁的发动机",
    "谁生产",
)


@dataclass(frozen=True)
class QueryRouteDecision:
    intent: str
    strategy: Strategy
    reasoning: str
    confidence: float
    slots: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class QueryRouterService:
    def __init__(self, llm=None) -> None:
        if llm is None:
            from app.core.llm import get_chat_model

            llm = get_chat_model(temperature=0)
        self.llm = llm

    def route(self, query: str, *, force_knowledge_base: bool = False) -> QueryRouteDecision:
        if force_knowledge_base:
            return QueryRouteDecision(
                intent="当前文档问答",
                strategy="knowledge_base",
                reasoning="请求限定了当前文档，只检索对应的内部知识库。",
                confidence=1.0,
                slots={"related": True, "domain_intent": "车辆使用与技术指导"},
            )

        response = self.llm.invoke(self._prompt(query)).content
        decision = self.parse_response(str(response))
        return self._enforce_route_boundaries(query, decision)

    def fallback_order(self, decision: QueryRouteDecision, *, force_knowledge_base: bool = False) -> list[Strategy]:
        if force_knowledge_base:
            return ["knowledge_base"]
        if decision.strategy == "react_agent":
            return ["react_agent"]

        order: list[Strategy] = [decision.strategy]
        for strategy in SIMPLE_STRATEGIES:
            if strategy not in order:
                order.append(strategy)
        return order

    @classmethod
    def parse_response(cls, response: str) -> QueryRouteDecision:
        try:
            payload = json.loads(fix_json(response))
        except Exception:
            return cls.default_decision("路由输出不是合法 JSON，降级到知识库检索。")

        if not isinstance(payload, dict):
            return cls.default_decision("路由输出不是 JSON 对象，降级到知识库检索。")

        return QueryRouteDecision(
            intent=str(payload.get("intent") or "未识别意图").strip(),
            strategy=cls._normalize_strategy(payload.get("strategy")),
            reasoning=str(payload.get("reasoning") or "模型未给出路由依据。").strip(),
            confidence=cls._normalize_confidence(payload.get("confidence")),
            slots=cls._normalize_slots(payload.get("slots")),
        )

    @staticmethod
    def default_decision(reasoning: str) -> QueryRouteDecision:
        return QueryRouteDecision(
            intent="知识库问答",
            strategy="knowledge_base",
            reasoning=reasoning,
            confidence=0.0,
            slots={},
        )

    @classmethod
    def _enforce_route_boundaries(
        cls,
        query: str,
        decision: QueryRouteDecision,
    ) -> QueryRouteDecision:
        if decision.strategy != "graph_db" or cls._has_graph_semantics(query):
            return decision

        return QueryRouteDecision(
            intent=decision.intent,
            strategy="knowledge_base",
            reasoning=(
                "问题是在询问实体本身的清单、定义、位置、功能或用法，未要求实体关系、路径、"
                "依赖或网络分析，应从说明书等非结构化文档检索。"
            ),
            confidence=max(0.9, decision.confidence),
            slots=decision.slots,
        )

    @staticmethod
    def _has_graph_semantics(query: str) -> bool:
        normalized = "".join(str(query).lower().split())
        return any(marker in normalized for marker in GRAPH_SEMANTIC_MARKERS)

    @staticmethod
    def _normalize_slots(value: object) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        return {str(key): slot_value for key, slot_value in value.items()}

    @staticmethod
    def _normalize_strategy(value: object) -> Strategy:
        strategy = str(value or "").strip().lower()
        aliases = {
            "sql": "relational_db",
            "mysql": "relational_db",
            "database": "relational_db",
            "relational": "relational_db",
            "neo4j": "graph_db",
            "cypher": "graph_db",
            "graph": "graph_db",
            "kb": "knowledge_base",
            "knowledge": "knowledge_base",
            "document": "knowledge_base",
            "document_search": "knowledge_base",
            "tavily": "web_search",
            "web": "web_search",
            "search": "web_search",
            "internet": "web_search",
            "external": "web_search",
            "react": "react_agent",
            "agent": "react_agent",
            "multi_tool": "react_agent",
            "complex": "react_agent",
            "complex_reasoning": "react_agent",
        }
        strategy = aliases.get(strategy, strategy)
        if strategy in STRATEGIES:
            return strategy  # type: ignore[return-value]
        return "knowledge_base"

    @staticmethod
    def _normalize_confidence(value: object) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, confidence))

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_prompt_template() -> str:
        return PROMPT_PATH.read_text(encoding="utf-8")

    @classmethod
    def _prompt(cls, query: str) -> str:
        return cls._load_prompt_template().replace("{{USER_QUERY}}", query.strip())
