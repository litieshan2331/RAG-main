from __future__ import annotations

import json
import re
from typing import Any

from neo4j import GraphDatabase

from app.core.config import get_settings
from app.core.llm import get_chat_model


FORBIDDEN_CYPHER = re.compile(
    r"\b(create|merge|delete|detach|set|remove|drop|call\s+dbms|call\s+apoc\.periodic)\b",
    flags=re.IGNORECASE,
)


class Text2CypherService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.driver = GraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_username, self.settings.neo4j_password),
        )
        self.llm = get_chat_model(temperature=0)

    def generate_cypher(self, query: str) -> str:
        prompt = self._prompt(query)
        response = self.llm.invoke(prompt).content
        return self._clean_cypher(str(response))

    def execute(self, query: str) -> tuple[str, list[dict[str, Any]]]:
        cypher = self.generate_cypher(query)
        if not cypher:
            return "", []
        self._validate_read_only(cypher)
        with self.driver.session(database=self.settings.neo4j_database) as session:
            result = session.run(cypher)
            rows = [dict(record) for record in result.fetch(self.settings.text2cypher_max_rows)]
        return cypher, json.loads(json.dumps(rows, ensure_ascii=False, default=str))

    def close(self) -> None:
        self.driver.close()

    def _prompt(self, query: str) -> str:
        return f"""
你是一个 Neo4j Text2Cypher 专家。请基于图数据库 Schema，把用户问题转换成一条只读 Cypher 查询。

Schema:
{self._schema_text()}

用户问题：
{query}

要求：
1. 只返回 Cypher，不要解释，不要 Markdown 代码块。
2. 只允许 MATCH、OPTIONAL MATCH、WITH、RETURN、WHERE、ORDER BY、LIMIT 等只读查询。
3. 不允许 CREATE、MERGE、SET、DELETE、REMOVE、DROP、写入类 CALL。
4. 必须使用 Schema 中存在的 label、relationship type 和 property。
5. 如果 Schema 无法支持该问题，返回空字符串 ""。
6. 对可能返回多行明细的查询，除非用户明确指定数量，否则添加 LIMIT {self.settings.text2cypher_max_rows}。
7. 关系方向优先参考 relationshipPatterns；不确定时可以用无向关系 `-[]-`。
"""

    def _schema_text(self) -> str:
        with self.driver.session(database=self.settings.neo4j_database) as session:
            labels = self._label_properties(session)
            relationships = self._relationship_patterns(session)
            property_keys = self._property_keys(session)
        return json.dumps(
            {
                "labels": labels,
                "relationshipPatterns": relationships,
                "propertyKeys": property_keys,
            },
            ensure_ascii=False,
            indent=2,
        )

    def _label_properties(self, session) -> list[dict[str, Any]]:
        query = """
        MATCH (n)
        UNWIND labels(n) AS label
        WITH label, keys(n) AS nodeKeys
        UNWIND nodeKeys AS property
        RETURN label, collect(DISTINCT property) AS properties
        ORDER BY label
        """
        try:
            return [
                {"label": record["label"], "properties": sorted(record["properties"] or [])}
                for record in session.run(query)
            ]
        except Exception:
            return [{"label": record["label"], "properties": []} for record in session.run("CALL db.labels() YIELD label RETURN label")]

    def _relationship_patterns(self, session) -> list[dict[str, Any]]:
        query = """
        MATCH (a)-[r]->(b)
        RETURN
          labels(a) AS startLabels,
          type(r) AS type,
          labels(b) AS endLabels,
          collect(DISTINCT keys(r))[0..10] AS propertySamples,
          count(*) AS count
        ORDER BY type
        LIMIT $limit
        """
        try:
            rows = session.run(query, limit=max(self.settings.text2cypher_schema_sample_limit * 20, 20))
            return [
                {
                    "startLabels": record["startLabels"],
                    "type": record["type"],
                    "endLabels": record["endLabels"],
                    "properties": sorted({key for sample in (record["propertySamples"] or []) for key in sample}),
                    "count": record["count"],
                }
                for record in rows
            ]
        except Exception:
            return [
                {"startLabels": [], "type": record["relationshipType"], "endLabels": [], "properties": [], "count": None}
                for record in session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
            ]

    @staticmethod
    def _property_keys(session) -> list[str]:
        try:
            return [record["propertyKey"] for record in session.run("CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey")]
        except Exception:
            return []

    @staticmethod
    def _clean_cypher(value: str) -> str:
        cypher = value.strip()
        fence = re.search(r"```(?:cypher)?\s*(.*?)```", cypher, flags=re.IGNORECASE | re.DOTALL)
        if fence:
            cypher = fence.group(1).strip()
        if cypher in {'""', "''"}:
            return ""
        return cypher.rstrip(";").strip()

    @staticmethod
    def _validate_read_only(cypher: str) -> None:
        normalized = cypher.strip().lower()
        if ";" in normalized:
            raise ValueError("multiple Cypher statements are not allowed")
        if not (
            normalized.startswith("match")
            or normalized.startswith("optional match")
            or normalized.startswith("with")
            or normalized.startswith("return")
        ):
            raise ValueError("only read-only Cypher queries are allowed")
        if FORBIDDEN_CYPHER.search(normalized):
            raise ValueError("write or administrative Cypher is not allowed")
