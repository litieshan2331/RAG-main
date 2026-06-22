from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text

from app.core.config import get_settings
from app.core.database import engine


@dataclass
class DependencyProbe:
    name: str
    ok: bool
    detail: str | None = None


class HealthService:
    def readiness(self) -> list[DependencyProbe]:
        return [
            self._probe_mysql(),
            self._probe_redis(),
            self._probe_elasticsearch(),
            self._probe_minio(),
            self._probe_neo4j(),
            self._probe_mineru(),
            self._probe_llm_config(),
            self._probe_web_search_config(),
        ]

    def _probe_mysql(self) -> DependencyProbe:
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return DependencyProbe("mysql", True)
        except Exception as exc:
            return DependencyProbe("mysql", False, str(exc))

    @staticmethod
    def _probe_redis() -> DependencyProbe:
        try:
            import redis

            client = redis.Redis.from_url(get_settings().redis_url)
            client.ping()
            return DependencyProbe("redis", True)
        except Exception as exc:
            return DependencyProbe("redis", False, str(exc))

    @staticmethod
    def _probe_elasticsearch() -> DependencyProbe:
        try:
            from elasticsearch import Elasticsearch

            client = Elasticsearch(get_settings().elasticsearch_url)
            ok = bool(client.ping())
            return DependencyProbe("elasticsearch", ok, None if ok else "ping returned false")
        except Exception as exc:
            return DependencyProbe("elasticsearch", False, str(exc))

    @staticmethod
    def _probe_minio() -> DependencyProbe:
        try:
            from app.storage.minio_store import MinioStorage

            storage = MinioStorage()
            exists = storage.client.bucket_exists(get_settings().minio_bucket)
            return DependencyProbe("minio", True, f"bucket_exists={exists}")
        except Exception as exc:
            return DependencyProbe("minio", False, str(exc))

    @staticmethod
    def _probe_neo4j() -> DependencyProbe:
        try:
            from neo4j import GraphDatabase

            settings = get_settings()
            driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password))
            try:
                with driver.session(database=settings.neo4j_database) as session:
                    session.run("RETURN 1 AS ok").single()
            finally:
                driver.close()
            return DependencyProbe("neo4j", True)
        except Exception as exc:
            return DependencyProbe("neo4j", False, str(exc))

    @staticmethod
    def _probe_mineru() -> DependencyProbe:
        try:
            import httpx

            url = get_settings().mineru_parse_api_url.rstrip("/")
            response = httpx.get(f"{url}/health", timeout=5, trust_env=False)
            return DependencyProbe("mineru", response.status_code < 500, f"http_status={response.status_code}")
        except Exception as exc:
            return DependencyProbe("mineru", False, str(exc))

    @staticmethod
    def _probe_llm_config() -> DependencyProbe:
        settings = get_settings()
        if not settings.openai_api_key:
            return DependencyProbe("llm_config", False, "OPENAI_API_KEY is empty")
        return DependencyProbe("llm_config", True, f"chat={settings.chat_model}, embedding={settings.embedding_model}")

    @staticmethod
    def _probe_web_search_config() -> DependencyProbe:
        settings = get_settings()
        if not settings.web_search_enabled:
            return DependencyProbe("web_search", True, "disabled")
        if not settings.tavily_api_key:
            return DependencyProbe("web_search", True, "enabled but TAVILY_API_KEY is empty; web_search fallback will be skipped")
        return DependencyProbe("web_search", True, "provider=tavily")
