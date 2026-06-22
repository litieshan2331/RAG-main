import pytest
from types import SimpleNamespace

from app.models.entities import TableMeta
from app.rag.tools import RelationalDbTool
from app.text2cypher.service import Text2CypherService
from app.text2sql.service import Text2SQLService


def test_text2sql_rejects_write_statements() -> None:
    with pytest.raises(ValueError):
        Text2SQLService._validate_read_only("delete from car_order")


def test_text2sql_accepts_select() -> None:
    Text2SQLService._validate_read_only("select * from car_info")


def test_text2sql_formats_table_meta_columns_info() -> None:
    table_meta = TableMeta(
        table_name="structured_sales",
        description="销量表",
        create_sql="CREATE TABLE `structured_sales` (`amount` DOUBLE)",
        columns_info='[{"name":"amount","originalName":"销量","type":"DOUBLE"}]',
    )

    schema = Text2SQLService._format_table_meta(table_meta)

    assert "structured_sales" in schema
    assert "description: 销量表" in schema
    assert "originalName=销量" in schema


def test_text2sql_empty_rows_trigger_tool_fallback(monkeypatch) -> None:
    class FakeService:
        @staticmethod
        def execute(_query):
            return "select * from structured_sales", []

        @staticmethod
        def is_effectively_empty_result(rows):
            return len(rows) == 0

    tool = object.__new__(RelationalDbTool)
    tool.service = FakeService()
    monkeypatch.setattr("app.rag.tools.get_settings", lambda: SimpleNamespace(text2sql_empty_result_triggers_fallback=True))

    with pytest.raises(RuntimeError, match="no rows"):
        tool.run("查销量")


def test_text2cypher_rejects_write_statements() -> None:
    with pytest.raises(ValueError):
        Text2CypherService._validate_read_only("MATCH (n) DELETE n")


def test_text2cypher_prompt_contains_schema_and_limit() -> None:
    service = object.__new__(Text2CypherService)
    service.settings = type("Settings", (), {"text2cypher_max_rows": 50})()
    service._schema_text = lambda: '{"labels":[{"label":"ImportedEntity","properties":["name"]}]}'

    prompt = service._prompt("查询节点关系")

    assert "ImportedEntity" in prompt
    assert "LIMIT 50" in prompt
