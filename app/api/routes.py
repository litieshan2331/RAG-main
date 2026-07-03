from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import SessionLocal, get_db
from app.core.errors import to_http_exception
from app.core.health import HealthService
from app.evaluation.ragas_evaluator import RagasDependencyError, RagasEvaluator
from app.models.entities import KnowledgeDocument
from app.rag.service import RagService
from app.retrieval.hybrid import HybridRetriever
from app.schemas import (
    ChunkRequest,
    ChunkResponse,
    ConvertResponse,
    AsyncUploadIngestResponse,
    DocumentUploadResponse,
    EmbedResponse,
    HealthResponse,
    IngestResponse,
    IngestionTaskResponse,
    QueryRequest,
    RagasEvaluationRequest,
    RagasEvaluationResponse,
    RagRequest,
    RagResponse,
    ReadinessResponse,
    RetrievalResponse,
    RetrievedDocument,
    TextQueryResponse,
    UploadIngestResponse,
)
from app.services.documents import DocumentService
from app.services.ingestion_tasks import IngestionTaskManager
from app.text2cypher.service import Text2CypherService
from app.text2sql.service import Text2SQLService

router = APIRouter()
WEB_DIR = Path(__file__).resolve().parents[1] / "web"


def _ingestion_task_response(task) -> IngestionTaskResponse:
    return IngestionTaskResponse(
        task_id=task.task_id,
        doc_id=task.document_id,
        status=task.status,
        stage=task.stage,
        progress=task.progress,
        message=task.message,
        error_message=task.error_message,
        cancel_requested=bool(task.cancel_requested),
        segment_count=task.segment_count,
        embedded_count=task.embedded_count,
    )


@router.get("/", response_class=HTMLResponse)
def web_app() -> HTMLResponse:
    return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8"))


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", app=get_settings().app_name)


@router.get("/health/ready", response_model=ReadinessResponse)
def readiness() -> ReadinessResponse:
    probes = HealthService().readiness()
    status = "ok" if all(probe.ok for probe in probes) else "degraded"
    return ReadinessResponse(status=status, dependencies=[probe.__dict__ for probe in probes])


@router.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    upload_user: str | None = Form(default=None),
    accessible_by: str | None = Form(default=None),
    description: str | None = Form(default=None),
    knowledge_base_type: str = Form(default="DOCUMENT_SEARCH"),
    table_name: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> DocumentUploadResponse:
    try:
        data = await file.read()
        service = DocumentService(db)
        document = service.upload(
            file_name=file.filename or "document",
            data=data,
            title=title or file.filename or "document",
            upload_user=upload_user,
            accessible_by=accessible_by,
            description=description,
            knowledge_base_type=knowledge_base_type,
            table_name=table_name,
        )
        return DocumentUploadResponse(doc_id=document.doc_id, doc_url=document.doc_url, status=document.status)
    except Exception as exc:
        raise to_http_exception(exc) from exc


@router.post("/api/documents/{doc_id}/convert", response_model=ConvertResponse)
def convert_document(doc_id: int, db: Session = Depends(get_db)) -> ConvertResponse:
    try:
        document = DocumentService(db).convert(doc_id)
        return ConvertResponse(doc_id=document.doc_id, converted_doc_url=document.converted_doc_url or "", status=document.status)
    except Exception as exc:
        raise to_http_exception(exc) from exc


@router.post("/api/documents/{doc_id}/chunk", response_model=ChunkResponse)
def chunk_document(doc_id: int, request: ChunkRequest, db: Session = Depends(get_db)) -> ChunkResponse:
    try:
        count = DocumentService(db).chunk(doc_id, request.chunk_size, request.overlap, request.strip_headers)
        return ChunkResponse(doc_id=doc_id, segment_count=count, status="CHUNKED")
    except Exception as exc:
        raise to_http_exception(exc) from exc


@router.post("/api/documents/{doc_id}/embed", response_model=EmbedResponse)
def embed_document(doc_id: int, db: Session = Depends(get_db)) -> EmbedResponse:
    try:
        count = DocumentService(db).embed_and_store(doc_id)
        return EmbedResponse(doc_id=doc_id, embedded_count=count, status="VECTOR_STORED")
    except Exception as exc:
        raise to_http_exception(exc) from exc


@router.post("/api/documents/{doc_id}/ingest", response_model=IngestResponse)
def ingest_document(doc_id: int, request: ChunkRequest, db: Session = Depends(get_db)) -> IngestResponse:
    try:
        document, segment_count, embedded_count = DocumentService(db).ingest(
            doc_id,
            request.chunk_size,
            request.overlap,
            request.strip_headers,
        )
        return IngestResponse(
            doc_id=document.doc_id,
            converted_doc_url=document.converted_doc_url,
            segment_count=segment_count,
            embedded_count=embedded_count,
            status=document.status,
        )
    except Exception as exc:
        raise to_http_exception(exc) from exc


@router.post("/api/documents/upload-and-ingest", response_model=UploadIngestResponse)
async def upload_and_ingest_document(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    upload_user: str | None = Form(default="web"),
    accessible_by: str | None = Form(default=None),
    description: str | None = Form(default=None),
    knowledge_base_type: str = Form(default="DOCUMENT_SEARCH"),
    table_name: str | None = Form(default=None),
    chunk_size: int = Form(default=1000),
    overlap: int = Form(default=80),
    strip_headers: bool = Form(default=False),
    db: Session = Depends(get_db),
) -> UploadIngestResponse:
    try:
        data = await file.read()
        service = DocumentService(db)
        document = service.upload(
            file_name=file.filename or "document",
            data=data,
            title=title or file.filename or "document",
            upload_user=upload_user,
            accessible_by=accessible_by,
            description=description,
            knowledge_base_type=knowledge_base_type,
            table_name=table_name,
        )
        document, segment_count, embedded_count = service.ingest(
            document.doc_id,
            chunk_size,
            overlap,
            strip_headers,
        )
        return UploadIngestResponse(
            doc_id=document.doc_id,
            doc_url=document.doc_url,
            converted_doc_url=document.converted_doc_url,
            segment_count=segment_count,
            embedded_count=embedded_count,
            status=document.status,
        )
    except Exception as exc:
        raise to_http_exception(exc) from exc


@router.post("/api/documents/upload-and-ingest-async", response_model=AsyncUploadIngestResponse, status_code=202)
async def upload_and_ingest_document_async(
    request: Request,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    upload_user: str | None = Form(default="web"),
    accessible_by: str | None = Form(default=None),
    description: str | None = Form(default=None),
    knowledge_base_type: str = Form(default="DOCUMENT_SEARCH"),
    table_name: str | None = Form(default=None),
    chunk_size: int = Form(default=1000),
    overlap: int = Form(default=80),
    strip_headers: bool = Form(default=False),
    db: Session = Depends(get_db),
) -> AsyncUploadIngestResponse:
    try:
        data = await file.read()
        document = DocumentService(db).upload(
            file_name=file.filename or "document",
            data=data,
            title=title or file.filename or "document",
            upload_user=upload_user,
            accessible_by=accessible_by,
            description=description,
            knowledge_base_type=knowledge_base_type,
            table_name=table_name,
        )
        manager: IngestionTaskManager = request.app.state.ingestion_manager
        task = manager.create_task(
            db,
            document_id=document.doc_id,
            chunk_size=max(100, chunk_size),
            overlap=max(0, overlap),
            strip_headers=strip_headers,
        )
        manager.submit(task.task_id)
        return AsyncUploadIngestResponse(
            task_id=task.task_id,
            doc_id=document.doc_id,
            doc_url=document.doc_url,
            knowledge_base_type=(document.knowledge_base_type or "DOCUMENT_SEARCH").upper(),
            status=task.status,
        )
    except Exception as exc:
        raise to_http_exception(exc) from exc


@router.get("/api/ingestion-tasks/{task_id}", response_model=IngestionTaskResponse)
def get_ingestion_task(task_id: str, request: Request, db: Session = Depends(get_db)) -> IngestionTaskResponse:
    manager: IngestionTaskManager = request.app.state.ingestion_manager
    task = manager.get_task(db, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="ingestion task not found")
    return _ingestion_task_response(task)


@router.post("/api/documents/{doc_id}/ingest-async", response_model=AsyncUploadIngestResponse, status_code=202)
def ingest_existing_document_async(
    doc_id: int,
    chunk_request: ChunkRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> AsyncUploadIngestResponse:
    document = db.get(KnowledgeDocument, doc_id)
    if document is None or document.deleted:
        raise HTTPException(status_code=404, detail="document not found")

    manager: IngestionTaskManager = request.app.state.ingestion_manager
    task = manager.get_active_task(db, doc_id)
    if task is None:
        task = manager.create_task(
            db,
            document_id=doc_id,
            chunk_size=chunk_request.chunk_size,
            overlap=chunk_request.overlap,
            strip_headers=chunk_request.strip_headers,
        )
        manager.submit(task.task_id)

    return AsyncUploadIngestResponse(
        task_id=task.task_id,
        doc_id=document.doc_id,
        doc_url=document.doc_url,
        knowledge_base_type=(document.knowledge_base_type or "DOCUMENT_SEARCH").upper(),
        status=task.status,
    )


@router.post("/api/ingestion-tasks/{task_id}/cancel", response_model=IngestionTaskResponse)
def cancel_ingestion_task(task_id: str, request: Request, db: Session = Depends(get_db)) -> IngestionTaskResponse:
    manager: IngestionTaskManager = request.app.state.ingestion_manager
    task = manager.request_cancel(db, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="ingestion task not found")
    return _ingestion_task_response(task)


@router.post("/api/retrieval/hybrid", response_model=RetrievalResponse)
def hybrid_retrieval(request: QueryRequest, db: Session = Depends(get_db)) -> RetrievalResponse:
    try:
        hits = HybridRetriever(db).retrieve(
            request.query,
            top_k=request.top_k,
            min_score=request.min_score,
            document_id=request.document_id,
        )
        return RetrievalResponse(
            query=request.query,
            results=[
                RetrievedDocument(text=hit.text, score=hit.score, source=hit.source, metadata=hit.metadata)
                for hit in hits
            ],
        )
    except Exception as exc:
        raise to_http_exception(exc) from exc


@router.post("/api/query/text2sql", response_model=TextQueryResponse)
def text2sql(request: QueryRequest, db: Session = Depends(get_db)) -> TextQueryResponse:
    try:
        sql, rows = Text2SQLService(db).execute(request.query)
        return TextQueryResponse(query=request.query, generated_query=sql, rows=rows)
    except Exception as exc:
        raise to_http_exception(exc) from exc


@router.post("/api/query/text2cypher", response_model=TextQueryResponse)
def text2cypher(request: QueryRequest) -> TextQueryResponse:
    service = Text2CypherService()
    try:
        cypher, rows = service.execute(request.query)
        return TextQueryResponse(query=request.query, generated_query=cypher, rows=rows)
    except Exception as exc:
        raise to_http_exception(exc) from exc
    finally:
        service.close()


@router.post("/api/chat/rag", response_model=RagResponse)
def rag_chat(request: RagRequest, db: Session = Depends(get_db)) -> RagResponse:
    try:
        result = RagService(db).answer(
            request.query,
            user_id=request.user_id,
            top_k=request.top_k,
            document_id=request.document_id,
            conversation_id=request.conversation_id,
        )
        return RagResponse(**result)
    except Exception as exc:
        raise to_http_exception(exc) from exc


@router.post("/api/chat/rag/stream")
def rag_chat_stream(request: RagRequest) -> StreamingResponse:
    def event_stream():
        try:
            with SessionLocal() as db:
                events = RagService(db).stream_answer(
                    request.query,
                    user_id=request.user_id,
                    top_k=request.top_k,
                    document_id=request.document_id,
                    conversation_id=request.conversation_id,
                )
                for item in events:
                    yield _encode_sse(str(item.get("event") or "message"), item.get("data") or {})
        except GeneratorExit:
            raise
        except Exception as exc:
            yield _encode_sse("error", {"message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/api/evaluation/ragas", response_model=RagasEvaluationResponse)
def evaluate_rag_with_ragas(
    request: RagasEvaluationRequest,
    db: Session = Depends(get_db),
) -> RagasEvaluationResponse:
    def run_rag(sample):
        return RagService(db).answer(
            sample.question,
            user_id=sample.user_id,
            top_k=sample.top_k,
            document_id=sample.document_id,
        )

    try:
        return RagasEvaluator(run_rag).evaluate_sync(request.samples)
    except RagasDependencyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _encode_sse(event: str, data: object) -> str:
    payload = json.dumps(jsonable_encoder(data), ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"
