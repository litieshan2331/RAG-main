from app.ingestion.structured_table import (
    infer_columns,
    normalize_table_name,
    parse_csv,
)


def test_parse_csv_and_infer_columns() -> None:
    rows = parse_csv("name,age,price\nAlice,18,12.5\nBob,20,9.8\n".encode("utf-8"))
    columns = infer_columns(rows)

    assert rows[0]["name"] == "Alice"
    assert [column.name for column in columns] == ["name", "age", "price"]
    assert [column.sql_type for column in columns] == ["VARCHAR(512)", "BIGINT", "DOUBLE"]


def test_normalize_table_name_adds_safe_prefix() -> None:
    assert normalize_table_name("2026 导入表") == "t_2026"
    assert normalize_table_name("vehicle-order") == "vehicle_order"
