from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.ids import snowflake
from app.core.llm import get_embedding_model
from app.ingestion.graph_data import GRAPH_SUFFIXES, GraphDataIngestor
from app.ingestion.markdown_cleaner import MarkdownCleaner
from app.ingestion.mineru import MinerUClient, MinerUConverter
from app.ingestion.splitter import ACCESSIBLE_BY, DOC_ID, FILE_NAME, SKIP_EMBEDDING, URL, MarkdownParentChildSplitter
from app.ingestion.structured_table import TABLE_SUFFIXES, StructuredTableIngestor
from app.models.entities import KnowledgeDocument, KnowledgeSegment
from app.retrieval.elasticsearch_store import ElasticsearchKnowledgeStore
from app.retrieval.hybrid import segment_metadata
from app.storage.minio_store import MinioStorage, safe_object_part


class DocumentService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.storage = MinioStorage()
        self.cleaner = MarkdownCleaner()
        self.mineru = MinerUClient()
        self.converter = MinerUConverter(self.storage, self.cleaner)

    def upload(
        self,
        *,
        file_name: str,
        data: bytes,
        title: str,
        upload_user: str | None,
        accessible_by: str | None,
        description: str | None,
        knowledge_base_type: str,
        table_name: str | None,
    ) -> KnowledgeDocument:
        object_name = f"uploads/{snowflake.next_id_str()}_{safe_object_part(file_name)}"
        content_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
        doc_url = self.storage.upload_bytes(object_name, data, content_type)
        extension = json.dumps({"tableName": table_name}, ensure_ascii=False) if table_name else None
        document = KnowledgeDocument(
            doc_title=title or file_name,
            upload_user=upload_user,
            doc_url=doc_url,
            status="UPLOADED",
            accessible_by=accessible_by,
            description=description,
            knowledge_base_type=knowledge_base_type,
            extension=extension,
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        return document

    def convert(self, doc_id: int) -> KnowledgeDocument:
        document = self._get_document(doc_id)
        self._ensure_document_search(document)
        if document.status in {"CONVERTED", "CHUNKED", "VECTOR_STORED"} and document.converted_doc_url:
            return document
        if not document.doc_url:
            raise RuntimeError("document has no source URL")

        document.status = "CONVERTING"
        self.db.commit()
        try:
            source_bytes = self.storage.download_bytes_by_url(document.doc_url)
            suffix = Path(document.doc_title).suffix.lower()
            if suffix in {".md", ".markdown", ".txt"}:
                converted_url = self._upload_clean_markdown(document.doc_title, source_bytes)
            else:
                zip_bytes = self.mineru.parse_to_zip(document.doc_title, source_bytes)
                converted_url = self.converter.zip_to_markdown_url(document.doc_title, zip_bytes)

            document.converted_doc_url = converted_url
            document.status = "CONVERTED"
            self.db.commit()
            self.db.refresh(document)
            return document
        except Exception:
            self.db.rollback()
            document = self._get_document(doc_id)
            document.status = "UPLOADED"
            self.db.commit()
            raise

    def chunk(self, doc_id: int, chunk_size: int, overlap: int, strip_headers: bool = False) -> int:
        document = self._get_document(doc_id)
        self._ensure_document_search(document)
        if document.status in {"CHUNKED", "VECTOR_STORED"}:
            return self._segment_count(doc_id)
        if document.status != "CONVERTED" or not document.converted_doc_url:
            raise RuntimeError("document status must be CONVERTED before chunking")

        markdown = self.storage.download_bytes_by_url(document.converted_doc_url).decode("utf-8")
        markdown = self.cleaner.clean(markdown)
        splitter = MarkdownParentChildSplitter(chunk_size=chunk_size, overlap=overlap, strip_headers=strip_headers)
        base_metadata = {
            DOC_ID: document.doc_id,
            FILE_NAME: document.doc_title,
            URL: document.doc_url,
            ACCESSIBLE_BY: document.accessible_by,
        }
        chunks = splitter.split(markdown, base_metadata)

        segments = []
        for index, chunk in enumerate(chunks):
            skip_embedding = int(chunk.metadata.get(SKIP_EMBEDDING, 0) or 0)
            segments.append(
                KnowledgeSegment(
                    text=chunk.text,
                    chunk_id=chunk.metadata.get("chunkId"),
                    metadata_json=json.dumps(chunk.metadata, ensure_ascii=False),
                    document_id=document.doc_id,
                    chunk_order=index,
                    status="STORED",
                    skip_embedding=skip_embedding,
                )
            )

        self.db.add_all(segments)
        document.status = "CHUNKED"
        self.db.commit()
        return len(segments)

    def ingest(self, doc_id: int, chunk_size: int, overlap: int, strip_headers: bool = False) -> tuple[KnowledgeDocument, int, int]:
        document = self._get_document(doc_id)
        knowledge_base_type = (document.knowledge_base_type or "DOCUMENT_SEARCH").upper()
        if knowledge_base_type == "STRUCTURED_TABLE":
            return self.ingest_structured_table(document)
        if knowledge_base_type == "GRAPH_DATA":
            return self.ingest_graph_data(document)

        document = self.convert(doc_id)
        segment_count = self.chunk(document.doc_id, chunk_size, overlap, strip_headers)
        embedded_count = self.embed_and_store(document.doc_id)
        self.db.refresh(document)
        return document, segment_count, embedded_count

    def embed_and_store(
        self,
        doc_id: int,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        document = self._get_document(doc_id)
        self._ensure_document_search(document)
        if document.status == "VECTOR_STORED":
            return 0
        if document.status != "CHUNKED":
            raise RuntimeError("document status must be CHUNKED before embedding")

        embedding_model = get_embedding_model()
        vector_store = ElasticsearchKnowledgeStore()
        segments = self.db.execute(
            select(KnowledgeSegment)
            .where(KnowledgeSegment.document_id == doc_id)
            .where(KnowledgeSegment.status == "STORED")
            .where(KnowledgeSegment.embedding_id.is_(None))
            .where(KnowledgeSegment.skip_embedding == 0)
            .order_by(KnowledgeSegment.chunk_order)
        ).scalars().all()

        embedded_count = 0
        total_count = len(segments)
        batch_size = max(1, get_settings().embedding_batch_size)
        for start in range(0, len(segments), batch_size):
            if cancel_check and cancel_check():
                break
            batch = list(segments[start : start + batch_size])
            embeddings = embedding_model.embed_documents([segment.text for segment in batch])
            index_batch = []
            for segment, embedding in zip(batch, embeddings, strict=True):
                embedding_id = segment.chunk_id or snowflake.next_id_str()
                index_batch.append((embedding_id, segment.text, embedding, segment_metadata(segment)))
            vector_store.index_segments(index_batch)
            for segment, (embedding_id, _, _, _) in zip(batch, index_batch, strict=True):
                segment.embedding_id = embedding_id
                segment.status = "VECTOR_STORED"
                embedded_count += 1
            self.db.commit()
            if progress_callback:
                progress_callback(embedded_count, total_count)

        remaining = self.db.execute(
            select(KnowledgeSegment)
            .where(KnowledgeSegment.document_id == doc_id)
            .where(KnowledgeSegment.status == "STORED")
            .where(KnowledgeSegment.skip_embedding == 0)
        ).first()
        if remaining is None:
            document.status = "VECTOR_STORED"
            self.db.commit()
        return embedded_count

    def ingest_structured_table(self, document: KnowledgeDocument) -> tuple[KnowledgeDocument, int, int]:
        if document.status == "STRUCTURED_STORED":
            return document, int((self._extension_dict(document).get("rowCount") or 0)), 0
        if not document.doc_url:
            raise RuntimeError("document has no source URL")

        suffix = Path(document.doc_title).suffix.lower()
        if suffix not in TABLE_SUFFIXES:
            raise RuntimeError("STRUCTURED_TABLE supports CSV and Excel files only")

        document.status = "STRUCTURING"
        self.db.commit()
        try:
            source_bytes = self.storage.download_bytes_by_url(document.doc_url)
            extension = self._extension_dict(document)
            result = StructuredTableIngestor(self.db).ingest(
                file_name=document.doc_title,
                data=source_bytes,
                table_name=extension.get("tableName"),
                doc_id=document.doc_id,
            )
            extension.update(
                {
                    "tableName": result.table_name,
                    "rowCount": len(result.rows),
                    "columnCount": len(result.columns),
                    "ingestMode": "STRUCTURED_TABLE",
                }
            )
            document.extension = json.dumps(extension, ensure_ascii=False)
            document.converted_doc_url = document.doc_url
            document.status = "STRUCTURED_STORED"
            self.db.commit()
            self.db.refresh(document)
            return document, len(result.rows), 0
        except Exception:
            self.db.rollback()
            document = self._get_document(document.doc_id)
            document.status = "UPLOADED"
            self.db.commit()
            raise

    def ingest_graph_data(self, document: KnowledgeDocument) -> tuple[KnowledgeDocument, int, int]:
        if document.status == "GRAPH_STORED":
            extension = self._extension_dict(document)
            return document, int(extension.get("relationshipCount") or 0), 0
        if not document.doc_url:
            raise RuntimeError("document has no source URL")

        suffix = Path(document.doc_title).suffix.lower()
        if suffix not in GRAPH_SUFFIXES:
            raise RuntimeError("GRAPH_DATA supports CSV, Excel, and JSON files only")

        document.status = "GRAPH_IMPORTING"
        self.db.commit()
        try:
            source_bytes = self.storage.download_bytes_by_url(document.doc_url)
            result = GraphDataIngestor(self.db).ingest(
                file_name=document.doc_title,
                data=source_bytes,
                doc_id=document.doc_id,
            )
            extension = self._extension_dict(document)
            extension.update(
                {
                    "nodeCount": result.node_count,
                    "relationshipCount": result.relationship_count,
                    "ingestMode": "GRAPH_DATA",
                }
            )
            document.extension = json.dumps(extension, ensure_ascii=False)
            document.converted_doc_url = document.doc_url
            document.status = "GRAPH_STORED"
            self.db.commit()
            self.db.refresh(document)
            return document, result.relationship_count, 0
        except Exception:
            self.db.rollback()
            document = self._get_document(document.doc_id)
            document.status = "UPLOADED"
            self.db.commit()
            raise

    def _upload_clean_markdown(self, doc_title: str, data: bytes) -> str:
        markdown = data.decode("utf-8")
        cleaned = self.cleaner.clean(markdown)
        safe_title = safe_object_part(doc_title)
        object_name = f"converted/{safe_title}/{safe_object_part(Path(doc_title).stem)}.md"
        return self.storage.upload_bytes(object_name, cleaned.encode("utf-8"), "text/markdown")

    def _get_document(self, doc_id: int) -> KnowledgeDocument:
        document = self.db.get(KnowledgeDocument, doc_id)
        if document is None or document.deleted:
            raise RuntimeError(f"document not found: {doc_id}")
        return document

    @staticmethod
    def _ensure_document_search(document: KnowledgeDocument) -> None:
        knowledge_base_type = (document.knowledge_base_type or "DOCUMENT_SEARCH").upper()
        if knowledge_base_type != "DOCUMENT_SEARCH":
            raise RuntimeError(f"{knowledge_base_type} does not use document chunking or vector embedding")

    @staticmethod
    def _extension_dict(document: KnowledgeDocument) -> dict:
        if not document.extension:
            return {}
        try:
            value = json.loads(document.extension)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def _segment_count(self, doc_id: int) -> int:
        return len(
            self.db.execute(
                select(KnowledgeSegment.id).where(KnowledgeSegment.document_id == doc_id, KnowledgeSegment.deleted == 0)
            ).all()
        )
