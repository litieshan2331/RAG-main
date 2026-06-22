import json

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.database import Base
from app.ingestion.splitter import CHUNK_ID, PARENT_CHUNK_ID
from app.models.entities import KnowledgeSegment
from app.retrieval.elasticsearch_store import SearchHit
from app.retrieval.hybrid import HybridRetriever, SegmentTextRepository, rrf_fuse
from app.retrieval.reranker import DashScopeReranker


def test_repository_reads_parent_chunk_by_chunk_id() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        session.add(_segment(1, 1, "complete parent text", "parent-1"))
        session.commit()

        text = SegmentTextRepository(session).get_text_by_chunk_id("parent-1")

    assert text == "complete parent text"


def test_rrf_fuse_merges_vector_and_full_text_rankings() -> None:
    vector_hits = [
        SearchHit(text="A", score=0.9, source="vector", metadata={CHUNK_ID: "a"}),
        SearchHit(text="B", score=0.8, source="vector", metadata={CHUNK_ID: "b"}),
    ]
    full_text_hits = [
        SearchHit(text="B", score=12.0, source="full_text", metadata={CHUNK_ID: "b"}),
        SearchHit(text="C", score=9.0, source="full_text", metadata={CHUNK_ID: "c"}),
    ]

    fused = rrf_fuse([vector_hits, full_text_hits], rrf_k=60)

    assert fused[0].metadata[CHUNK_ID] == "b"
    assert fused[0].source == "vector+full_text"
    assert len(fused) == 3


def test_parent_expansion_groups_children_before_replacement() -> None:
    retriever = object.__new__(HybridRetriever)
    retriever.segment_repository = FakeSegmentRepository()
    hits = [
        SearchHit(text="child one", score=0.03, source="vector", metadata={CHUNK_ID: "c1", PARENT_CHUNK_ID: "p1"}),
        SearchHit(text="child two", score=0.02, source="full_text", metadata={CHUNK_ID: "c2", PARENT_CHUNK_ID: "p1"}),
    ]

    expanded = retriever._expand_parent_groups(hits)

    assert len(expanded) == 1
    assert expanded[0].text == "complete parent"
    assert expanded[0].score == 0.05
    assert expanded[0].metadata["matchedChildChunkIds"] == ["c1", "c2"]
    assert retriever.segment_repository.calls == ["p1"]


def test_dashscope_reranker_maps_indices_to_hits() -> None:
    hits = [
        SearchHit(text="first", score=0.1, source="vector", metadata={CHUNK_ID: "a"}),
        SearchHit(text="second", score=0.2, source="full_text", metadata={CHUNK_ID: "b"}),
    ]

    ranked = DashScopeReranker._apply_results(
        hits,
        [{"index": 1, "relevance_score": 0.98}, {"index": 0, "relevance_score": 0.31}],
        top_k=2,
    )

    assert [hit.text for hit in ranked] == ["second", "first"]
    assert ranked[0].score == 0.98
    assert ranked[0].source == "full_text+rerank"
    assert ranked[0].metadata["rerankerScore"] == 0.98


class FakeSegmentRepository:
    def __init__(self) -> None:
        self.calls = []

    def get_text_by_chunk_id(self, chunk_id: str) -> str:
        self.calls.append(chunk_id)
        return "complete parent"


def _segment(segment_id: int, document_id: int, text: str, chunk_id: str) -> KnowledgeSegment:
    return KnowledgeSegment(
        id=segment_id,
        text=text,
        chunk_id=chunk_id,
        metadata_json=json.dumps(
            {
                CHUNK_ID: chunk_id,
                PARENT_CHUNK_ID: None,
            }
        ),
        document_id=document_id,
        chunk_order=1,
        status="VECTOR_STORED",
        skip_embedding=0,
    )
