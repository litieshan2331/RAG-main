from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import SessionLocal, engine
from app.core.ids import snowflake
from app.models.entities import IngestionTask, KnowledgeDocument
from app.services.documents import DocumentService


logger = logging.getLogger(__name__)
TERMINAL_TASK_STATUSES = {"COMPLETED", "FAILED", "CANCELLED"}


class IngestionTaskManager:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.executor = ThreadPoolExecutor(
            max_workers=max(1, self.settings.ingestion_worker_count),
            thread_name_prefix="ingestion",
        )
        self.futures: dict[str, Future] = {}
        self.cancel_events: dict[str, threading.Event] = {}
        self.lock = threading.Lock()

    def start(self) -> None:
        IngestionTask.__table__.create(bind=engine, checkfirst=True)
        with SessionLocal() as db:
            db.execute(
                update(IngestionTask)
                .where(IngestionTask.status.in_(["PENDING", "RUNNING", "CONVERTING", "CHUNKING", "EMBEDDING"]))
                .values(
                    status="FAILED",
                    stage="INTERRUPTED",
                    message="Task was interrupted by an application restart.",
                    error_message="Application restarted before ingestion completed.",
                )
            )
            db.commit()

    def shutdown(self) -> None:
        with self.lock:
            for event in self.cancel_events.values():
                event.set()
        self.executor.shutdown(wait=False, cancel_futures=True)

    def create_task(
        self,
        db: Session,
        *,
        document_id: int,
        chunk_size: int,
        overlap: int,
        strip_headers: bool,
    ) -> IngestionTask:
        task = IngestionTask(
            task_id=snowflake.next_id_str(),
            document_id=document_id,
            status="PENDING",
            stage="QUEUED",
            progress=5,
            message="File uploaded. Waiting for background ingestion.",
            cancel_requested=0,
            parameters={
                "chunk_size": chunk_size,
                "overlap": overlap,
                "strip_headers": strip_headers,
            },
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        return task

    @staticmethod
    def get_active_task(db: Session, document_id: int) -> IngestionTask | None:
        return db.execute(
            select(IngestionTask)
            .where(IngestionTask.document_id == document_id)
            .where(IngestionTask.status.not_in(TERMINAL_TASK_STATUSES))
            .order_by(IngestionTask.id.desc())
        ).scalars().first()

    def submit(self, task_id: str) -> None:
        event = threading.Event()
        with self.lock:
            self.cancel_events[task_id] = event
            self.futures[task_id] = self.executor.submit(self._run_task, task_id, event)

    def request_cancel(self, db: Session, task_id: str) -> IngestionTask | None:
        task = self.get_task(db, task_id)
        if task is None or task.status in TERMINAL_TASK_STATUSES:
            return task
        task.cancel_requested = 1
        task.status = "CANCEL_REQUESTED"
        task.message = "Cancellation requested. The current MinerU request may need to finish first."
        db.commit()
        db.refresh(task)
        with self.lock:
            event = self.cancel_events.get(task_id)
            future = self.futures.get(task_id)
            if event:
                event.set()
            if future and future.cancel():
                task.status = "CANCELLED"
                task.stage = "CANCELLED"
                task.message = "Task cancelled before execution."
                db.commit()
                db.refresh(task)
                self.cancel_events.pop(task_id, None)
                self.futures.pop(task_id, None)
        return task

    @staticmethod
    def get_task(db: Session, task_id: str) -> IngestionTask | None:
        return db.execute(select(IngestionTask).where(IngestionTask.task_id == task_id)).scalar_one_or_none()

    def _run_task(self, task_id: str, cancel_event: threading.Event) -> None:
        try:
            with SessionLocal() as db:
                task = self.get_task(db, task_id)
                if task is None:
                    return
                params = task.parameters or {}
                document = db.get(KnowledgeDocument, task.document_id)
                if document is None:
                    raise RuntimeError(f"document {task.document_id} does not exist")
                service = DocumentService(db)

                if self._cancelled(task_id, cancel_event):
                    self._mark_cancelled(task_id)
                    return

                knowledge_base_type = (document.knowledge_base_type or "DOCUMENT_SEARCH").upper()
                if knowledge_base_type != "DOCUMENT_SEARCH":
                    self._update(task_id, status="RUNNING", stage="INGESTING", progress=20, message=f"Importing {knowledge_base_type} data.")
                    _, segment_count, embedded_count = service.ingest(
                        document.doc_id,
                        int(params.get("chunk_size", 1000)),
                        int(params.get("overlap", 80)),
                        bool(params.get("strip_headers", False)),
                    )
                    if self._cancelled(task_id, cancel_event):
                        self._mark_cancelled(task_id)
                        return
                    self._complete(task_id, segment_count, embedded_count)
                    return

                self._update(task_id, status="CONVERTING", stage="CONVERTING", progress=10, message="MinerU is converting the document to Markdown.")
                document = service.convert(document.doc_id)
                if self._cancelled(task_id, cancel_event):
                    self._mark_cancelled(task_id)
                    return

                self._update(task_id, status="CHUNKING", stage="CHUNKING", progress=65, message="Splitting Markdown into document and parent-child chunks.")
                segment_count = service.chunk(
                    document.doc_id,
                    int(params.get("chunk_size", 1000)),
                    int(params.get("overlap", 80)),
                    bool(params.get("strip_headers", False)),
                )
                if self._cancelled(task_id, cancel_event):
                    self._mark_cancelled(task_id)
                    return

                self._update(task_id, status="EMBEDDING", stage="EMBEDDING", progress=80, message="Embedding chunks and writing the Elasticsearch index.", segment_count=segment_count)
                embedded_count = service.embed_and_store(
                    document.doc_id,
                    cancel_check=lambda: self._cancelled(task_id, cancel_event),
                    progress_callback=lambda completed, total: self._embedding_progress(
                        task_id,
                        completed,
                        total,
                        segment_count,
                    ),
                )
                if self._cancelled(task_id, cancel_event):
                    self._mark_cancelled(task_id)
                    return
                self._complete(task_id, segment_count, embedded_count)
        except Exception as exc:
            logger.exception("Background ingestion task %s failed", task_id)
            self._update(
                task_id,
                status="FAILED",
                stage="FAILED",
                message="Background ingestion failed.",
                error_message=str(exc),
            )
        finally:
            with self.lock:
                self.cancel_events.pop(task_id, None)
                self.futures.pop(task_id, None)

    def _cancelled(self, task_id: str, cancel_event: threading.Event) -> bool:
        if cancel_event.is_set():
            return True
        with SessionLocal() as db:
            task = self.get_task(db, task_id)
            return bool(task and task.cancel_requested)

    def _mark_cancelled(self, task_id: str) -> None:
        self._update(
            task_id,
            status="CANCELLED",
            stage="CANCELLED",
            message="Task cancelled. Completed stages were preserved.",
        )

    def _complete(self, task_id: str, segment_count: int, embedded_count: int) -> None:
        self._update(
            task_id,
            status="COMPLETED",
            stage="COMPLETED",
            progress=100,
            message="Document ingestion completed.",
            segment_count=segment_count,
            embedded_count=embedded_count,
        )

    def _embedding_progress(self, task_id: str, completed: int, total: int, segment_count: int) -> None:
        ratio = completed / max(1, total)
        self._update(
            task_id,
            progress=min(98, 80 + int(ratio * 18)),
            message=f"Embedded {completed} of {total} searchable chunks.",
            segment_count=segment_count,
            embedded_count=completed,
        )

    @staticmethod
    def _update(task_id: str, **values: Any) -> None:
        with SessionLocal() as db:
            db.execute(update(IngestionTask).where(IngestionTask.task_id == task_id).values(**values))
            db.commit()
