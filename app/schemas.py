from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app: str


class DependencyStatus(BaseModel):
    name: str
    ok: bool
    detail: str | None = None


class ReadinessResponse(BaseModel):
    status: str
    dependencies: list[DependencyStatus]


class DocumentUploadResponse(BaseModel):
    doc_id: int
    doc_url: str | None
    status: str


class ConvertResponse(BaseModel):
    doc_id: int
    converted_doc_url: str
    status: str


class ChunkRequest(BaseModel):
    chunk_size: int = Field(default=1000, ge=100)
    overlap: int = Field(default=80, ge=0)
    strip_headers: bool = False


class ChunkResponse(BaseModel):
    doc_id: int
    segment_count: int
    status: str


class EmbedResponse(BaseModel):
    doc_id: int
    embedded_count: int
    status: str


class IngestResponse(BaseModel):
    doc_id: int
    converted_doc_url: str | None = None
    segment_count: int
    embedded_count: int
    status: str


class UploadIngestResponse(BaseModel):
    doc_id: int
    doc_url: str | None = None
    converted_doc_url: str | None = None
    segment_count: int
    embedded_count: int
    status: str


class AsyncUploadIngestResponse(BaseModel):
    task_id: str
    doc_id: int
    doc_url: str | None = None
    knowledge_base_type: str
    status: str


class IngestionTaskResponse(BaseModel):
    task_id: str
    doc_id: int
    status: str
    stage: str
    progress: int
    message: str | None = None
    error_message: str | None = None
    cancel_requested: bool = False
    segment_count: int = 0
    embedded_count: int = 0


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    min_score: float = Field(default=0.5, ge=0, le=1)
    document_id: int | None = None


class RetrievedDocument(BaseModel):
    text: str
    score: float | None = None
    source: str
    metadata: dict = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    query: str
    results: list[RetrievedDocument]


class TextQueryResponse(BaseModel):
    query: str
    generated_query: str
    rows: list[dict] | None = None
    result_text: str | None = None


class RagRequest(BaseModel):
    query: str
    conversation_id: str | None = None
    user_id: str = "123321"
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: int | None = None


class RagResponse(BaseModel):
    query: str
    conversation_id: str
    rewritten_query: str
    context_decision: dict | None = None
    route: str
    route_decision: dict | None = None
    answer: str
    contexts: list[RetrievedDocument] = Field(default_factory=list)
    tool_trace: list[dict] = Field(default_factory=list)
