from __future__ import annotations

from typing import Any, Iterator

from sqlalchemy.orm import Session

from app.core.ids import snowflake
from app.rag.graph import OnlineRagGraph


class RagService:
    def __init__(self, db: Session) -> None:
        self.graph = OnlineRagGraph(db)

    def answer(
        self,
        query: str,
        user_id: str,
        top_k: int,
        document_id: int | None = None,
        conversation_id: str | None = None,
    ) -> dict:
        resolved_conversation_id = conversation_id or snowflake.next_id_str()
        return self.graph.answer(
            query=query,
            user_id=user_id,
            top_k=top_k,
            document_id=document_id,
            conversation_id=resolved_conversation_id,
        )

    def stream_answer(
        self,
        query: str,
        user_id: str,
        top_k: int,
        document_id: int | None = None,
        conversation_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        resolved_conversation_id = conversation_id or snowflake.next_id_str()
        yield from self.graph.stream_answer(
            query=query,
            user_id=user_id,
            top_k=top_k,
            document_id=document_id,
            conversation_id=resolved_conversation_id,
        )
