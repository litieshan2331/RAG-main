from __future__ import annotations

from fastapi import HTTPException


UPSTREAM_MODULE_PREFIXES = (
    "httpx",
    "elasticsearch",
    "minio",
    "neo4j",
    "redis",
    "pymysql",
    "sqlalchemy",
)


def to_http_exception(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, (ValueError, RuntimeError)):
        return HTTPException(status_code=400, detail=str(exc))

    module = exc.__class__.__module__.lower()
    if module.startswith(UPSTREAM_MODULE_PREFIXES):
        return HTTPException(status_code=502, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))
