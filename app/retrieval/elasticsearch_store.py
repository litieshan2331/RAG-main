from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from app.core.config import get_settings


@dataclass
class SearchHit:
    text: str
    score: float | None
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ElasticsearchKnowledgeStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = Elasticsearch(
            self.settings.elasticsearch_url,
            request_timeout=self.settings.elasticsearch_request_timeout_seconds,
            max_retries=3,
            retry_on_timeout=True,
        )
        self.index = self.settings.elasticsearch_index

    def ensure_index(self) -> None:
        if self.client.indices.exists(index=self.index):
            return
        self.client.indices.create(
            index=self.index,
            mappings={
                "properties": {
                    "text": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self.settings.embedding_dimensions,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "metadata": {"type": "object", "enabled": True},
                    "chunk_id": {"type": "keyword"},
                    "document_id": {"type": "long"},
                    "skip_embedding": {"type": "integer"},
                }
            },
        )

    def index_segment(self, embedding_id: str, text: str, embedding: list[float], metadata: dict) -> str:
        self.ensure_index()
        document = {
            "text": text,
            "embedding": embedding,
            "metadata": metadata,
            "chunk_id": metadata.get("chunkId"),
            "document_id": metadata.get("docId"),
            "skip_embedding": metadata.get("skipEmbedding", 0),
        }
        self.client.index(index=self.index, id=embedding_id, document=document, refresh=False)
        return embedding_id

    def index_segments(self, segments: list[tuple[str, str, list[float], dict]]) -> None:
        if not segments:
            return
        self.ensure_index()
        actions = []
        for embedding_id, text, embedding, metadata in segments:
            actions.append(
                {
                    "_index": self.index,
                    "_id": embedding_id,
                    "_source": {
                        "text": text,
                        "embedding": embedding,
                        "metadata": metadata,
                        "chunk_id": metadata.get("chunkId"),
                        "document_id": metadata.get("docId"),
                        "skip_embedding": metadata.get("skipEmbedding", 0),
                    },
                }
            )
        bulk(self.client, actions, refresh=False, request_timeout=self.settings.elasticsearch_request_timeout_seconds)

    def vector_search(self, query_vector: list[float], top_k: int, min_score: float, document_id: int | None = None) -> list[SearchHit]:
        self.ensure_index()
        knn = {
            "field": "embedding",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": max(top_k * 10, 50),
        }
        if document_id is not None:
            knn["filter"] = {"term": {"document_id": document_id}}
        response = self.client.search(
            index=self.index,
            size=top_k,
            knn=knn,
            source=["text", "metadata"],
        )
        hits: list[SearchHit] = []
        for hit in response["hits"]["hits"]:
            score = float(hit.get("_score") or 0)
            if score < min_score:
                continue
            source = hit.get("_source", {})
            hits.append(SearchHit(text=source.get("text", ""), score=score, source="vector", metadata=source.get("metadata") or {}))
        return hits

    def full_text_search(self, query: str, top_k: int, document_id: int | None = None) -> list[SearchHit]:
        self.ensure_index()
        es_query: dict = {"match": {"text": {"query": query}}}
        if document_id is not None:
            es_query = {
                "bool": {
                    "must": [{"match": {"text": {"query": query}}}],
                    "filter": [{"term": {"document_id": document_id}}],
                }
            }
        response = self.client.search(
            index=self.index,
            size=top_k,
            query=es_query,
            source=["text", "metadata"],
        )
        return [
            SearchHit(
                text=(hit.get("_source") or {}).get("text", ""),
                score=float(hit.get("_score") or 0),
                source="full_text",
                metadata=(hit.get("_source") or {}).get("metadata") or {},
            )
            for hit in response["hits"]["hits"]
        ]
