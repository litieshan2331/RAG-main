from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.ingestion.structured_table import parse_table_file


GRAPH_SUFFIXES = {".csv", ".xlsx", ".xls", ".json"}


@dataclass(frozen=True)
class GraphIngestionResult:
    node_count: int
    relationship_count: int


class GraphDataIngestor:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.settings = get_settings()

    def ingest(self, *, file_name: str, data: bytes, doc_id: int) -> GraphIngestionResult:
        nodes, relationships = parse_graph_file(file_name, data)
        if not nodes and not relationships:
            raise RuntimeError("graph data contains no nodes or relationships")

        driver = GraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_username, self.settings.neo4j_password),
        )
        try:
            with driver.session(database=self.settings.neo4j_database) as session:
                for node in nodes:
                    session.run(
                        """
                        MERGE (n:ImportedEntity {name: $name})
                        SET n.sourceDocId = $doc_id,
                            n.kind = coalesce($kind, n.kind)
                        """,
                        name=node["name"],
                        kind=node.get("kind"),
                        doc_id=doc_id,
                    )
                for rel in relationships:
                    session.run(
                        """
                        MERGE (s:ImportedEntity {name: $source})
                        MERGE (t:ImportedEntity {name: $target})
                        MERGE (s)-[r:RELATED_TO {type: $relation, sourceDocId: $doc_id}]->(t)
                        SET r.properties = $properties
                        """,
                        source=rel["source"],
                        target=rel["target"],
                        relation=rel.get("relation") or "RELATED_TO",
                        properties=json.dumps(rel.get("properties") or {}, ensure_ascii=False),
                        doc_id=doc_id,
                    )
        finally:
            driver.close()

        return GraphIngestionResult(node_count=len(nodes), relationship_count=len(relationships))


def parse_graph_file(file_name: str, data: bytes) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".json":
        payload = json.loads(data.decode("utf-8-sig"))
        return parse_graph_json(payload)
    if suffix in {".csv", ".xlsx", ".xls"}:
        return parse_graph_rows(parse_table_file(file_name, data))
    raise RuntimeError(f"unsupported graph data file type: {suffix}")


def parse_graph_json(payload: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(payload, list):
        return [], [normalize_relationship(item) for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        raise RuntimeError("graph json must be an object or an array")

    raw_nodes = payload.get("nodes") or []
    raw_relationships = payload.get("relationships") or payload.get("edges") or []
    nodes = [normalize_node(item) for item in raw_nodes if isinstance(item, dict)]
    relationships = [normalize_relationship(item) for item in raw_relationships if isinstance(item, dict)]
    return nodes, relationships


def parse_graph_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    relationships = [normalize_relationship(row) for row in rows]
    return [], relationships


def normalize_node(item: dict[str, Any]) -> dict[str, Any]:
    name = item.get("name") or item.get("id") or item.get("label")
    if not name:
        raise RuntimeError("graph node requires name/id/label")
    return {"name": str(name), "kind": item.get("kind") or item.get("type")}


def normalize_relationship(item: dict[str, Any]) -> dict[str, Any]:
    lowered = {str(key).strip().lower(): value for key, value in item.items()}
    source = lowered.get("source") or lowered.get("from") or lowered.get("start") or lowered.get("head")
    target = lowered.get("target") or lowered.get("to") or lowered.get("end") or lowered.get("tail")
    relation = lowered.get("relation") or lowered.get("relationship") or lowered.get("type") or lowered.get("label")
    if not source or not target:
        raise RuntimeError("graph relationship requires source/from and target/to columns")
    return {
        "source": str(source),
        "target": str(target),
        "relation": str(relation or "RELATED_TO"),
        "properties": {key: value for key, value in item.items() if value is not None},
    }
