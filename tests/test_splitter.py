from app.ingestion.splitter import (
    PARENT_CHUNK_ID,
    SKIP_EMBEDDING,
    MarkdownParentChildSplitter,
)


def test_splitter_adds_parent_chunk_for_oversized_section() -> None:
    markdown = "# Manual\n## Brake\n" + "Brake noise troubleshooting. " * 30
    chunks = MarkdownParentChildSplitter(chunk_size=60, overlap=10).split(markdown)

    parent_chunks = [chunk for chunk in chunks if chunk.metadata.get(SKIP_EMBEDDING) == 1]
    child_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata.get(PARENT_CHUNK_ID) and chunk.metadata.get(SKIP_EMBEDDING) != 1
    ]

    assert parent_chunks
    assert child_chunks
    assert all(len(chunk.text) <= 60 for chunk in child_chunks)


def test_splitter_ignores_headers_inside_code_blocks() -> None:
    markdown = "# Title\n```text\n# not a header\n```\nBody"
    chunks = MarkdownParentChildSplitter(chunk_size=1000, overlap=0).split(markdown)

    assert len(chunks) == 1
    assert "# not a header" in chunks[0].text


def test_splitter_only_sets_parent_for_oversized_children() -> None:
    markdown = "# Manual\n## Brake\nShort section."
    chunks = MarkdownParentChildSplitter(chunk_size=1000, overlap=0).split(markdown)

    assert all(PARENT_CHUNK_ID not in chunk.metadata for chunk in chunks)
