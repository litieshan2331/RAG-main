from __future__ import annotations

import json
import re
from datetime import date
from importlib import resources
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.llm import get_chat_model
from app.models.entities import TableMeta


FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|replace|grant|revoke|call|execute)\b",
    flags=re.IGNORECASE,
)


class Text2SQLService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.settings = get_settings()
        self.llm = get_chat_model(temperature=0)

    def generate_sql(self, query: str) -> str:
        prompt = self._prompt_template().format(
            today=date.today().isoformat(),
            database_structure=self._database_structure(),
            max_rows=self.settings.text2sql_max_rows,
            query=query,
        )
        response = self.llm.invoke(prompt).content
        return self._clean_sql(str(response))

    def execute(self, query: str) -> tuple[str, list[dict[str, Any]]]:
        sql = self.generate_sql(query)
        if not sql:
            return "", []
        self._validate_read_only(sql)
        rows = self.db.execute(text(sql)).mappings().fetchmany(self.settings.text2sql_max_rows)
        return sql, [self._jsonable(dict(row)) for row in rows]

    def _database_structure(self) -> str:
        base = self._resource("sql/retrieve_tables.sql").read_text(encoding="utf-8")
        dynamic = [self._format_table_meta(row) for row in self.db.query(TableMeta).filter(TableMeta.deleted == 0).all()]
        dynamic = [item for item in dynamic if item.strip()]
        if dynamic:
            return f"{base}\n\n-- Dynamic tables from table_meta\n\n" + "\n\n".join(dynamic)
        return base

    @staticmethod
    def is_effectively_empty_result(rows: list[dict[str, Any]]) -> bool:
        return len(rows) == 0

    @staticmethod
    def _format_table_meta(row: TableMeta) -> str:
        parts = [f"-- table_meta: {row.table_name}"]
        if row.description:
            parts.append(f"-- description: {row.description}")
        if row.create_sql and row.create_sql.strip():
            parts.append(row.create_sql.strip())
        columns_info = Text2SQLService._parse_columns_info(row.columns_info)
        if columns_info:
            parts.append("-- columns_info:")
            for column in columns_info:
                name = column.get("name")
                original_name = column.get("originalName") or column.get("original_name")
                sql_type = column.get("type") or column.get("sqlType") or column.get("sql_type")
                if name:
                    parts.append(f"--   name={name}; originalName={original_name or name}; type={sql_type or 'unknown'}")
        return "\n".join(parts)

    @staticmethod
    def _parse_columns_info(value: str | None) -> list[dict[str, Any]]:
        if not value:
            return []
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    @staticmethod
    def _prompt_template() -> str:
        return resources.files("app.resources.prompts").joinpath("text_to_sql_prompt.txt").read_text(encoding="utf-8")

    @staticmethod
    def _resource(relative: str):
        root, name = relative.split("/", 1)
        return resources.files(f"app.resources.{root}").joinpath(name)

    @staticmethod
    def _clean_sql(value: str) -> str:
        sql = value.strip()
        fence = re.search(r"```(?:sql)?\s*(.*?)```", sql, flags=re.IGNORECASE | re.DOTALL)
        if fence:
            sql = fence.group(1).strip()
        if sql in {'""', "''"}:
            return ""
        return sql.rstrip(";").strip()

    @staticmethod
    def _validate_read_only(sql: str) -> None:
        normalized = sql.strip().lower()
        if ";" in normalized:
            raise ValueError("multiple SQL statements are not allowed")
        if not (normalized.startswith("select") or normalized.startswith("with")):
            raise ValueError("only SELECT/WITH queries are allowed")
        if FORBIDDEN_SQL.search(normalized):
            raise ValueError("write or administrative SQL is not allowed")

    @staticmethod
    def _jsonable(row: dict[str, Any]) -> dict[str, Any]:
        return json.loads(json.dumps(row, ensure_ascii=False, default=str))
