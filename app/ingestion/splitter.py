from __future__ import annotations

from dataclasses import dataclass, field

from app.core.ids import snowflake

FILE_NAME = "fileName"
DOC_ID = "docId"
CHUNK_ID = "chunkId"
PARENT_CHUNK_ID = "parentChunkId"
BROTHER_CHUNK_ID = "brotherChunkId"
BROTHER_CHUNK_INDEX = "brotherChunkIndex"
BROTHER_CHUNK_TOTAL = "brotherChunkTotal"
HEADER_LEVEL = "headerLevel"
ACCESSIBLE_BY = "accessibleBy"
URL = "url"
SKIP_EMBEDDING = "skipEmbedding"


HEADER_KEYS = {
    1: "title",
    2: "subtitle",
    3: "subsubtitle",
    4: "subsubsubtitle",
    5: "subsubsubsubtitle",
    6: "subsubsubsubsubtitle",
}


@dataclass
class TextChunk:
    text: str
    metadata: dict = field(default_factory=dict)


class MarkdownParentChildSplitter:
    def __init__(self, chunk_size: int = 1000, overlap: int = 80, strip_headers: bool = False) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strip_headers = strip_headers

    def split(self, markdown: str, base_metadata: dict | None = None) -> list[TextChunk]:
        base_metadata = dict(base_metadata or {})
        sections = self._split_by_headers(markdown, base_metadata)
        return self._split_large_sections(sections)

    def _split_by_headers(self, markdown: str, base_metadata: dict) -> list[TextChunk]:
        lines = [line for line in markdown.split("\n") if line.strip()]
        chunks: list[TextChunk] = []
        current_lines: list[str] = []
        current_metadata = dict(base_metadata)
        rolling_metadata = dict(base_metadata)
        header_stack: list[tuple[int, str, str]] = []
        in_code_block = False
        opening_fence = ""

        for raw_line in lines:
            line = raw_line.strip()
            if not in_code_block and (line.startswith("```") or line.startswith("~~~")):
                in_code_block = True
                opening_fence = line[:3]
                current_lines.append(line)
                continue
            if in_code_block:
                current_lines.append(line)
                if line.startswith(opening_fence):
                    in_code_block = False
                    opening_fence = ""
                continue

            header_level = self._header_level(line)
            if header_level:
                if current_lines:
                    chunks.append(TextChunk("\n".join(current_lines), dict(current_metadata)))
                    current_lines = []

                while header_stack and header_stack[-1][0] >= header_level:
                    _, key, _ = header_stack.pop()
                    rolling_metadata.pop(key, None)

                header_key = HEADER_KEYS[header_level]
                header_value = line[header_level:].strip()
                header_stack.append((header_level, header_key, header_value))
                rolling_metadata[header_key] = header_value
                rolling_metadata[HEADER_LEVEL] = header_level
                rolling_metadata[CHUNK_ID] = snowflake.next_id_str()

                current_metadata = dict(rolling_metadata)
                if not self.strip_headers:
                    current_lines.append(line)
                continue

            current_lines.append(line)
            current_metadata = dict(rolling_metadata)

        if current_lines:
            chunks.append(TextChunk("\n".join(current_lines), dict(current_metadata)))

        return chunks

    @staticmethod
    def _header_level(line: str) -> int | None:
        if not line.startswith("#"):
            return None
        count = len(line) - len(line.lstrip("#"))
        if 1 <= count <= 6 and (len(line) == count or line[count] == " "):
            return count
        return None

    def _split_large_sections(self, sections: list[TextChunk]) -> list[TextChunk]:
        if self.chunk_size <= 0:
            return sections

        result: list[TextChunk] = []
        for section in sections:
            if len(section.text) <= self.chunk_size:
                result.append(section)
                continue

            parent_chunk_id = snowflake.next_id_str()
            parent_metadata = dict(section.metadata)
            parent_metadata[CHUNK_ID] = parent_chunk_id
            parent_metadata[SKIP_EMBEDDING] = 1
            result.append(TextChunk(section.text, parent_metadata))

            children = self._sliding_children(section.text, section.metadata, parent_chunk_id)
            result.extend(children)
        return result

    def _sliding_children(self, text: str, metadata: dict, parent_chunk_id: str) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            child_metadata = dict(metadata)
            child_metadata[CHUNK_ID] = snowflake.next_id_str()
            child_metadata[PARENT_CHUNK_ID] = parent_chunk_id
            chunks.append(TextChunk(text[start:end], child_metadata))
            if end == len(text):
                break
            start = end - min(self.overlap, end)

        return chunks
