from __future__ import annotations

import json
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.llm import get_embedding_model
from app.ingestion.splitter import CHUNK_ID, PARENT_CHUNK_ID
from app.models.entities import KnowledgeSegment
from app.retrieval.elasticsearch_store import ElasticsearchKnowledgeStore, SearchHit
from app.retrieval.reranker import DashScopeReranker
from app.storage.redis_cache import RedisCache


class SegmentTextRepository:
    def __init__(self, db: Session, cache: RedisCache | None = None) -> None:
        self.db = db
        self.cache = cache or RedisCache()

    def get_text_by_chunk_id(self, chunk_id: str) -> str | None:
        cached = self.cache.get(chunk_id)
        if cached is not None:
            return cached or None

        segment = self.db.execute(
            select(KnowledgeSegment).where(KnowledgeSegment.chunk_id == chunk_id, KnowledgeSegment.deleted == 0)
        ).scalar_one_or_none()
        if segment:
            self.cache.set(chunk_id, segment.text, ttl_seconds=30)
            return segment.text

        self.cache.set(chunk_id, "", ttl_seconds=30)
        return None


class HybridRetriever:
    def __init__(
        self,
        db: Session,
        store: ElasticsearchKnowledgeStore | None = None,
        reranker: DashScopeReranker | None = None,
    ) -> None:
        self.settings = get_settings()
        self.embedding_model = get_embedding_model()
        self.store = store or ElasticsearchKnowledgeStore()
        self.segment_repository = SegmentTextRepository(db)
        self.reranker = reranker or DashScopeReranker()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        document_id: int | None = None,
    ) -> list[SearchHit]:
        top_k = top_k or self.settings.hybrid_top_k
        min_score = self.settings.hybrid_min_score if min_score is None else min_score
        candidate_k = max(top_k, self.settings.hybrid_candidate_k)
        query_vector = self.embedding_model.embed_query(query)
        vector_hits = self.store.vector_search(query_vector, top_k=candidate_k, min_score=min_score, document_id=document_id)
        full_text_hits = self.store.full_text_search(query, top_k=candidate_k, document_id=document_id)
        fused_hits = rrf_fuse([vector_hits, full_text_hits], rrf_k=self.settings.hybrid_rrf_k, limit=candidate_k)
        expanded_hits = self._expand_parent_groups(fused_hits)
        return self.reranker.rerank(query, expanded_hits, top_k)

    def _expand_parent_groups(self, hits: list[SearchHit]) -> list[SearchHit]:
        groups: dict[str, _HitGroup] = {}
        for order, hit in enumerate(hits):
            parent_chunk_id = hit.metadata.get(PARENT_CHUNK_ID)
            group_key = f"parent:{parent_chunk_id}" if parent_chunk_id else f"chunk:{hit_key(hit)}"
            if group_key not in groups:
                groups[group_key] = _HitGroup(parent_chunk_id=str(parent_chunk_id) if parent_chunk_id else None)
            groups[group_key].add(hit, order)

        expanded: list[SearchHit] = []
        for group in groups.values():
            expanded.append(self._group_to_hit(group))

        return sorted(expanded, key=lambda item: item.score or 0, reverse=True)

    def _group_to_hit(self, group: "_HitGroup") -> SearchHit:
        best = group.best_hit
        metadata = dict(best.metadata)
        metadata["matchedChildChunkIds"] = group.child_chunk_ids
        metadata["matchedChildCount"] = len(group.hits)

        if not group.parent_chunk_id:
            return SearchHit(text=best.text, score=group.score, source=group.source, metadata=metadata)

        parent_text = self.segment_repository.get_text_by_chunk_id(group.parent_chunk_id)
        if parent_text:
            metadata[CHUNK_ID] = group.parent_chunk_id
            metadata["expandedFromParentChunkId"] = group.parent_chunk_id
            return SearchHit(
                text=parent_text,
                score=group.score,
                source=f"{group.source}+parent",
                metadata=metadata,
            )

        fallback_text = "\n".join(hit.text for hit in group.hits)
        return SearchHit(
            text=fallback_text,
            score=group.score,
            source=f"{group.source}+children",
            metadata=metadata,
        )


@dataclass
class _HitGroup:
    parent_chunk_id: str | None
    hits: list[SearchHit] | None = None
    score: float = 0.0
    source: str = ""

    def __post_init__(self) -> None:
        self.hits = []

    def add(self, hit: SearchHit, order: int) -> None:
        hit.metadata["_retrievalOrder"] = order
        self.hits.append(hit)
        self.score += hit.score or 0
        self.source = merge_source(self.source, hit.source)

    @property
    def best_hit(self) -> SearchHit:
        return max(self.hits or [], key=lambda hit: hit.score or 0)

    @property
    def child_chunk_ids(self) -> list[str]:
        ids = []
        for hit in self.hits or []:
            chunk_id = hit.metadata.get(CHUNK_ID)
            if chunk_id is not None:
                ids.append(str(chunk_id))
        return ids


def rrf_fuse(ranked_hit_lists: list[list[SearchHit]], rrf_k: int = 60, limit: int | None = None) -> list[SearchHit]:
    fused: dict[str, SearchHit] = {}
    for hits in ranked_hit_lists:
        for rank, hit in enumerate(hits, start=1):
            key = hit_key(hit)
            contribution = 1.0 / (rrf_k + rank)
            if key not in fused:
                fused[key] = SearchHit(
                    text=hit.text,
                    score=contribution,
                    source=hit.source,
                    metadata=dict(hit.metadata),
                )
                continue

            existing = fused[key]
            existing.score = (existing.score or 0) + contribution
            existing.source = merge_source(existing.source, hit.source)
            existing.metadata.update(hit.metadata)

    ordered = sorted(fused.values(), key=lambda item: item.score or 0, reverse=True)
    return ordered[:limit] if limit is not None else ordered


def hit_key(hit: SearchHit) -> str:
    metadata = hit.metadata or {}
    return str(metadata.get(CHUNK_ID) or hash(hit.text))


def merge_source(left: str, right: str) -> str:
    values = []
    for source in [*left.split("+"), *right.split("+")]:
        if source and source not in values:
            values.append(source)
    return "+".join(values)


def segment_metadata(segment: KnowledgeSegment) -> dict:
    if not segment.metadata_json:
        return {}
    return json.loads(segment.metadata_json)
