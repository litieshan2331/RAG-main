from __future__ import annotations

import re


class MarkdownCleaner:
    """Clean MinerU markdown while preserving structural signals for chunking."""

    def clean(self, markdown: str) -> str:
        text = markdown.replace("\r\n", "\n").replace("\r", "\n")
        text = self._normalize_image_alt(text)
        text = self._collapse_blank_lines(text)
        text = self._trim_table_spacing(text)
        return text.strip() + "\n"

    def _normalize_image_alt(self, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            alt = match.group(1).strip()
            url = match.group(2).strip()
            if alt:
                return f"![{alt}]({url})"
            fallback = url.rsplit("/", 1)[-1] or "image"
            return f"![{fallback}]({url})"

        return re.sub(r"!\[(.*?)\]\((.*?)\)", replace, text)

    @staticmethod
    def _collapse_blank_lines(text: str) -> str:
        return re.sub(r"\n{3,}", "\n\n", text)

    @staticmethod
    def _trim_table_spacing(text: str) -> str:
        lines = [line.rstrip() for line in text.split("\n")]
        return "\n".join(lines)
