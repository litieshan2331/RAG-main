from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.ids import snowflake
from app.core.json_utils import fix_json
from app.models.entities import ChatConversation, ChatMessage


CONTEXTUALIZER_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "resources" / "prompts" / "conversation_contextualizer_prompt.txt"
)
MULTI_HOP_MARKERS = (
    "结合",
    "对比",
    "比较",
    "先查询",
    "先查",
    "再查询",
    "再查",
    "然后",
    "分别查询",
    "分别查",
    "并判断",
    "并分析",
    "并给出依据",
    "基于上述",
    "根据结果",
    "综合判断",
)


@dataclass(frozen=True)
class ConversationTurn:
    role: str
    content: str
    transformed_content: str | None = None


@dataclass(frozen=True)
class ContextDependencyDecision:
    depends_on_history: bool
    reasoning: str
    confidence: float
    requires_decomposition: bool = False
    sub_queries: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextualizedQuery:
    standalone_query: str
    decision: ContextDependencyDecision


class ConversationRepository:
    def __init__(self, db: Session, settings: Settings | None = None) -> None:
        self.db = db
        self.settings = settings or get_settings()

    def recent_turns(self, conversation_id: str, user_id: str) -> list[ConversationTurn]:
        if not conversation_id:
            return []
        conversation = self.db.execute(
            select(ChatConversation).where(
                ChatConversation.conversation_id == conversation_id,
                ChatConversation.user_id == user_id,
                ChatConversation.deleted == 0,
            )
        ).scalar_one_or_none()
        if conversation is None:
            return []

        messages = self.db.execute(
            select(ChatMessage)
            .where(ChatMessage.conversation_id == conversation_id, ChatMessage.deleted == 0)
            .order_by(desc(ChatMessage.id))
            .limit(self.settings.conversation_history_max_messages)
        ).scalars().all()
        messages.reverse()
        return [
            ConversationTurn(
                role=message.type,
                content=message.content or "",
                transformed_content=message.transform_content,
            )
            for message in messages
            if message.content
        ]

    def save_turn(
        self,
        *,
        conversation_id: str,
        user_id: str,
        query: str,
        rewritten_query: str,
        answer: str,
        route: str,
        contexts: list[Any],
        context_decision: dict[str, Any] | None,
    ) -> None:
        conversation = self.db.execute(
            select(ChatConversation).where(ChatConversation.conversation_id == conversation_id)
        ).scalar_one_or_none()
        if conversation is None:
            conversation = ChatConversation(
                conversation_id=conversation_id,
                user_id=user_id,
                title=query[:512],
                status="active",
            )
            self.db.add(conversation)

        references = [self._reference_dict(item) for item in contexts]
        self.db.add(
            ChatMessage(
                message_id=snowflake.next_id_str(),
                conversation_id=conversation_id,
                type="user",
                content=query,
                transform_content=rewritten_query,
                metadata_json={"contextDecision": context_decision or {}},
            )
        )
        self.db.add(
            ChatMessage(
                message_id=snowflake.next_id_str(),
                conversation_id=conversation_id,
                type="assistant",
                content=answer,
                rag_references=references,
                metadata_json={"route": route},
            )
        )
        self.db.commit()

    @staticmethod
    def _reference_dict(item: Any) -> dict[str, Any]:
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="json")
        if isinstance(item, dict):
            return item
        return {"text": str(item)}


class ConversationContextService:
    def __init__(self, llm=None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if llm is None:
            from app.core.llm import get_chat_model

            llm = get_chat_model(temperature=0)
        self.llm = llm

    def contextualize(self, query: str, history: list[ConversationTurn]) -> ContextualizedQuery:
        if not history and not self._looks_like_multi_hop(query):
            return ContextualizedQuery(
                standalone_query=query,
                decision=ContextDependencyDecision(False, "没有历史消息，当前问题是独立问题。", 1.0),
            )

        formatted_history = self.format_history(history) if history else "（无历史对话）"
        response = self.llm.invoke(self._contextualizer_prompt(query, formatted_history)).content
        result = self.parse_contextualization(str(response), query)
        decision = result.decision
        if decision.confidence < self.settings.context_dependency_threshold:
            return ContextualizedQuery(
                standalone_query=query,
                decision=ContextDependencyDecision(
                    False,
                    f"依赖判断置信度不足，按独立问题处理：{decision.reasoning}",
                    decision.confidence,
                ),
            )
        if not decision.depends_on_history:
            return ContextualizedQuery(standalone_query=query, decision=decision)
        return result

    def format_history(self, history: list[ConversationTurn]) -> str:
        selected: list[str] = []
        total = 0
        for turn in reversed(history):
            content = turn.transformed_content or turn.content
            line = f"{turn.role}: {content.strip()}"
            if selected and total + len(line) > self.settings.conversation_history_max_chars:
                break
            selected.append(line)
            total += len(line)
        return "\n".join(reversed(selected))

    @staticmethod
    def parse_contextualization(response: str, original_query: str) -> ContextualizedQuery:
        try:
            payload = json.loads(fix_json(response))
        except Exception:
            return ContextualizedQuery(
                standalone_query=original_query,
                decision=ContextDependencyDecision(False, "上下文处理输出不是合法 JSON，按独立问题处理。", 0.0),
            )
        if not isinstance(payload, dict):
            return ContextualizedQuery(
                standalone_query=original_query,
                decision=ContextDependencyDecision(False, "上下文处理输出不是 JSON 对象，按独立问题处理。", 0.0),
            )

        raw_value = payload.get("depends_on_history", False)
        depends_on_history = raw_value if isinstance(raw_value, bool) else str(raw_value).lower() == "true"
        try:
            confidence = max(0.0, min(1.0, float(payload.get("confidence", 0))))
        except (TypeError, ValueError):
            confidence = 0.0
        standalone_query = str(payload.get("standalone_query") or "").strip()
        raw_requires_decomposition = payload.get("requires_decomposition", False)
        requires_decomposition = (
            raw_requires_decomposition
            if isinstance(raw_requires_decomposition, bool)
            else str(raw_requires_decomposition).lower() == "true"
        )
        sub_queries = ConversationContextService._normalize_sub_queries(payload.get("sub_queries"))
        if len(sub_queries) < 2:
            requires_decomposition = False
            sub_queries = []
        if depends_on_history and not standalone_query:
            return ContextualizedQuery(
                standalone_query=original_query,
                decision=ContextDependencyDecision(
                    False,
                    "模型判定问题依赖历史，但未生成独立问题，按原问题处理。",
                    0.0,
                ),
            )
        return ContextualizedQuery(
            standalone_query=standalone_query if depends_on_history else original_query,
            decision=ContextDependencyDecision(
                depends_on_history=depends_on_history,
                reasoning=str(payload.get("reasoning") or "模型未给出原因。").strip(),
                confidence=confidence,
                requires_decomposition=requires_decomposition,
                sub_queries=sub_queries,
            ),
        )

    @staticmethod
    def _normalize_sub_queries(value: object) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []

        normalized: list[dict[str, Any]] = []
        known_ids: set[str] = set()
        for index, item in enumerate(value[:5], start=1):
            if isinstance(item, str):
                query = item.strip()
                raw_id = f"q{index}"
                raw_dependencies: object = []
            elif isinstance(item, dict):
                query = str(item.get("query") or "").strip()
                raw_id = str(item.get("id") or f"q{index}").strip()
                raw_dependencies = item.get("depends_on") or []
            else:
                continue
            if not query:
                continue

            step_id = raw_id if raw_id and raw_id not in known_ids else f"q{index}"
            if step_id in known_ids:
                step_id = f"q{index}_{len(normalized) + 1}"
            dependencies = (
                [str(dependency).strip() for dependency in raw_dependencies]
                if isinstance(raw_dependencies, list)
                else []
            )
            dependencies = [dependency for dependency in dependencies if dependency in known_ids]
            normalized.append({"id": step_id, "query": query, "depends_on": dependencies})
            known_ids.add(step_id)
        return normalized

    @staticmethod
    def _looks_like_multi_hop(query: str) -> bool:
        normalized = "".join(query.lower().split())
        if sum(normalized.count(mark) for mark in ("?", "？")) >= 2:
            return True
        return any(marker in normalized for marker in MULTI_HOP_MARKERS)

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_contextualizer_prompt() -> str:
        return CONTEXTUALIZER_PROMPT_PATH.read_text(encoding="utf-8")

    @classmethod
    def _contextualizer_prompt(cls, query: str, history: str) -> str:
        return (
            cls._load_contextualizer_prompt()
            .replace("{{HISTORY}}", history)
            .replace("{{USER_QUERY}}", query)
        )
