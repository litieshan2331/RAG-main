from __future__ import annotations

from io import BytesIO
import re
from urllib.parse import unquote, urlparse

from minio import Minio

from app.core.config import get_settings


class MinioStorage:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = Minio(
            endpoint=self.settings.minio_endpoint,
            access_key=self.settings.minio_access_key,
            secret_key=self.settings.minio_secret_key,
            secure=self.settings.minio_secure,
        )

    def ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.settings.minio_bucket):
            self.client.make_bucket(self.settings.minio_bucket)

    def upload_bytes(self, object_name: str, data: bytes, content_type: str) -> str:
        self.ensure_bucket()
        self.client.put_object(
            bucket_name=self.settings.minio_bucket,
            object_name=object_name,
            data=BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return self.public_url(object_name)

    def download_bytes_by_url(self, url: str) -> bytes:
        object_name = self.object_name_from_url(url)
        response = self.client.get_object(self.settings.minio_bucket, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def public_url(self, object_name: str) -> str:
        endpoint = self.settings.minio_public_endpoint.rstrip("/")
        bucket = self.settings.minio_bucket.strip("/")
        return f"{endpoint}/{bucket}/{object_name.lstrip('/')}"

    def object_name_from_url(self, url: str) -> str:
        parsed = urlparse(url)
        path = unquote(parsed.path).lstrip("/")
        bucket_prefix = f"{self.settings.minio_bucket}/"
        if path.startswith(bucket_prefix):
            return path[len(bucket_prefix) :]
        return path


def safe_object_part(value: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|]+", "_", value).strip().strip(".")
    return cleaned or "document"
