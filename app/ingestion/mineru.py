from __future__ import annotations

import base64
from dataclasses import dataclass
import logging
import re
import tempfile
import zipfile
from pathlib import Path

import httpx

from app.core.config import get_settings
from app.core.llm import get_vision_model
from app.ingestion.markdown_cleaner import MarkdownCleaner
from app.storage.minio_store import MinioStorage, safe_object_part


logger = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


@dataclass(frozen=True)
class ImageAsset:
    url: str
    description: str | None = None


class MinerUClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    def parse_to_zip(self, file_name: str, data: bytes) -> bytes:
        url = self.settings.mineru_parse_api_url.rstrip("/") + "/file_parse"
        response_timeout = (
            None
            if self.settings.mineru_response_timeout_seconds <= 0
            else self.settings.mineru_response_timeout_seconds
        )
        timeout = httpx.Timeout(
            connect=self.settings.mineru_connect_timeout_seconds,
            read=response_timeout,
            write=response_timeout,
            pool=self.settings.mineru_connect_timeout_seconds,
        )
        files = {"files": (file_name, data, "application/octet-stream")}
        form = {
            "backend": "pipeline",
            "response_format_zip": "true",
            "return_images": "true",
            "return_model_output": "false",
            "return_middle_json": "false",
        }
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            response = client.post(url, files=files, data=form, headers={"Accept": "application/json"})
            response.raise_for_status()
            return response.content


class MinerUConverter:
    def __init__(self, storage: MinioStorage | None = None, cleaner: MarkdownCleaner | None = None) -> None:
        self.settings = get_settings()
        self.storage = storage or MinioStorage()
        self.cleaner = cleaner or MarkdownCleaner()

    def zip_to_markdown_url(self, doc_title: str, zip_bytes: bytes) -> str:
        safe_title = safe_object_part(doc_title)
        with tempfile.TemporaryDirectory(prefix="know-engine-mineru-") as temp_dir:
            extract_dir = Path(temp_dir) / "extract"
            extract_dir.mkdir(parents=True, exist_ok=True)
            zip_path = Path(temp_dir) / "mineru.zip"
            zip_path.write_bytes(zip_bytes)
            self._safe_extract(zip_path, extract_dir)

            md_file = self._find_markdown(extract_dir)
            image_assets = self._upload_images(doc_title, extract_dir)
            md_content = md_file.read_text(encoding="utf-8")
            md_content = self._replace_image_urls(md_content, image_assets)
            cleaned = self.cleaner.clean(md_content)
            object_name = f"converted/{safe_title}/{safe_object_part(md_file.name)}"
            return self.storage.upload_bytes(object_name, cleaned.encode("utf-8"), "text/markdown")

    @staticmethod
    def _safe_extract(zip_path: Path, extract_dir: Path) -> None:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                target = (extract_dir / member.filename).resolve()
                if not str(target).startswith(str(extract_dir.resolve())):
                    continue
                archive.extract(member, extract_dir)

    @staticmethod
    def _find_markdown(extract_dir: Path) -> Path:
        markdown_files = list(extract_dir.rglob("*.md"))
        if not markdown_files:
            raise RuntimeError("MinerU zip does not contain a markdown file")
        return markdown_files[0]

    def _upload_images(self, doc_title: str, extract_dir: Path) -> dict[str, ImageAsset]:
        safe_title = safe_object_part(doc_title)
        image_assets: dict[str, ImageAsset] = {}
        for image_path in extract_dir.rglob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            content_type = self._image_content_type(image_path)
            object_name = f"converted/{safe_title}/images/{safe_object_part(image_path.name)}"
            image_bytes = image_path.read_bytes()
            description = self._describe_image(image_path.name, image_bytes, content_type)
            url = self.storage.upload_bytes(object_name, image_bytes, content_type)
            image_assets[image_path.name] = ImageAsset(url=url, description=description)
        return image_assets

    @staticmethod
    def _image_content_type(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix == ".png":
            return "image/png"
        if suffix == ".gif":
            return "image/gif"
        if suffix == ".webp":
            return "image/webp"
        if suffix == ".bmp":
            return "image/bmp"
        return "application/octet-stream"

    def _describe_image(self, image_name: str, image_bytes: bytes, content_type: str) -> str | None:
        if not self.settings.mineru_generate_image_descriptions:
            return None
        if not self.settings.openai_api_key:
            return None
        if len(image_bytes) > self.settings.mineru_image_description_max_bytes:
            logger.warning("Skip image description for %s because it is larger than configured limit.", image_name)
            return None

        data_url = f"data:{content_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        prompt = (
            "请描述这张图片的内容，优先提取图中的文字、表格、公式、结构关系、零部件名称和操作步骤。"
            "直接输出一段简洁中文描述，不要使用 Markdown，不要解释你的过程。"
        )
        try:
            response = get_vision_model(temperature=0.2).invoke(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ]
            )
            description = str(response.content).strip()
            return self._clean_image_alt(description) or None
        except Exception as exc:
            logger.warning("Failed to generate image description for %s: %s", image_name, exc)
            return None

    @staticmethod
    def _replace_image_urls(markdown: str, image_assets: dict[str, ImageAsset]) -> str:
        def replace(match: re.Match[str]) -> str:
            alt, raw_path = match.group(1), match.group(2)
            file_name = Path(raw_path).name
            asset = image_assets.get(file_name)
            if not asset:
                return match.group(0)
            next_alt = MinerUConverter._clean_image_alt(asset.description or alt or file_name)
            return f"![{next_alt}]({asset.url})"

        return re.sub(r"!\[(.*?)\]\(([^)]+)\)", replace, markdown)

    @staticmethod
    def _clean_image_alt(value: str) -> str:
        cleaned = re.sub(r"\s+", " ", value).strip()
        cleaned = cleaned.replace("[", "（").replace("]", "）")
        return cleaned or "image"
