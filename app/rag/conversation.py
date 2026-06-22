from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.ids import snowflake
from app.core.json_utils import fix_json
from app.models.entities import ChatConversation, ChatMessage


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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

    def assess_dependency(self, query: str, history: list[ConversationTurn]) -> ContextDependencyDecision:
        if not history:
            return ContextDependencyDecision(False, "没有历史消息，当前问题是独立问题。", 1.0)

        response = self.llm.invoke(self._dependency_prompt(query, self.format_history(history))).content
        decision = self.parse_dependency(str(response))
        if decision.confidence < self.settings.context_dependency_threshold:
            return ContextDependencyDecision(
                False,
                f"依赖判断置信度不足，按独立问题处理：{decision.reasoning}",
                decision.confidence,
            )
        return decision

    def rewrite_follow_up(self, query: str, history: list[ConversationTurn]) -> str:
        response = self.llm.invoke(self._rewrite_prompt(query, self.format_history(history))).content
        rewritten = str(response).strip()
        return rewritten or query

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
    def parse_dependency(response: str) -> ContextDependencyDecision:
        try:
            payload = json.loads(fix_json(response))
        except Exception:
            return ContextDependencyDecision(False, "依赖判断输出不是合法 JSON，按独立问题处理。", 0.0)
        if not isinstance(payload, dict):
            return ContextDependencyDecision(False, "依赖判断输出不是 JSON 对象，按独立问题处理。", 0.0)

        raw_value = payload.get("depends_on_history", False)
        depends_on_history = raw_value if isinstance(raw_value, bool) else str(raw_value).lower() == "true"
        try:
            confidence = max(0.0, min(1.0, float(payload.get("confidence", 0))))
        except (TypeError, ValueError):
            confidence = 0.0
        return ContextDependencyDecision(
            depends_on_history=depends_on_history,
            reasoning=str(payload.get("reasoning") or "模型未给出原因。").strip(),
            confidence=confidence,
        )

    @staticmethod
    def _dependency_prompt(query: str, history: str) -> str:
        return f"""
你是多轮对话上下文依赖分类器。只判断当前问题是否必须依赖历史对话才能理解，不要回答问题。

判定为依赖历史的情况：
- 使用“它、这个、那个、上述、前者、后者、继续、再说说、为什么、还有呢”等指代或省略。
- 当前问题省略了实体、车型、文档、时间、比较对象或动作，必须从历史中补全。
- 明确要求对上一轮回答继续解释、比较、追问或修正。

判定为不依赖历史的情况：
- 当前问题已经包含完整实体和意图，可以独立理解。
- 话题发生切换，即使历史里有相似词，也不要继承历史意图。
- 只是礼貌语或新的完整问题。

历史对话：
{history}

当前问题：
{query}

严格只输出 JSON：
{{
  "depends_on_history": true,
  "reasoning": "判断依据",
  "confidence": 0.90
}}
"""

    @staticmethod
    def _rewrite_prompt(query: str, history: str) -> str:
        return f"""
你是多轮对话问题重写器。当前问题已被确认依赖历史对话，请将它改写为无需历史也能理解的独立问题。

要求：
1. 只补充历史中明确出现的信息，不要猜测或扩展新意图。
2. 保留当前问题真正询问的动作、范围、时间和约束。
3. 如果历史存在多个实体，只选择当前追问明确指向的实体。
4. 只输出改写后的独立问题，不要解释。

历史对话：
{history}

当前问题：
{query}
"""
