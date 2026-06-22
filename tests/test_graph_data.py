from app.ingestion.graph_data import parse_graph_file


def test_parse_graph_json_edges() -> None:
    data = b'{"nodes":[{"name":"A","kind":"car"}],"edges":[{"source":"A","target":"B","relation":"USES"}]}'

    nodes, relationships = parse_graph_file("graph.json", data)

    assert nodes == [{"name": "A", "kind": "car"}]
    assert relationships[0]["source"] == "A"
    assert relationships[0]["target"] == "B"
    assert relationships[0]["relation"] == "USES"


def test_parse_graph_csv_relationships() -> None:
    data = "from,to,type\nA,B,RELATED\n".encode("utf-8")

    nodes, relationships = parse_graph_file("graph.csv", data)

    assert nodes == []
    assert relationships[0]["source"] == "A"
    assert relationships[0]["target"] == "B"
    assert relationships[0]["relation"] == "RELATED"
