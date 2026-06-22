from __future__ import annotations

import logging

import httpx

from app.core.config import get_settings
from app.retrieval.elasticsearch_store import SearchHit


logger = logging.getLogger(__name__)


class DashScopeReranker:
    def __init__(self) -> None:
        self.settings = get_settings()

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not self.settings.reranker_enabled or len(hits) <= 1:
            return hits[:top_k]
        if not self.settings.openai_api_key:
            logger.warning("Reranker is enabled but OPENAI_API_KEY is empty; fallback to RRF order.")
            return hits[:top_k]

        try:
            results = self._request_rerank(query, hits, top_k)
            return self._apply_results(hits, results, top_k)
        except Exception as exc:
            logger.warning("DashScope reranker failed, fallback to RRF order: %s", exc)
            return hits[:top_k]

    def _request_rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[dict]:
        documents = [hit.text[: self.settings.reranker_max_input_chars] for hit in hits]
        payload = {
            "model": self.settings.reranker_model,
            "query": query,
            "documents": documents,
            "top_n": min(top_k, len(documents)),
        }
        if self.settings.reranker_instruct:
            payload["instruct"] = self.settings.reranker_instruct

        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.settings.reranker_timeout_seconds, trust_env=False) as client:
            response = client.post(self.settings.reranker_api_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        output = data.get("output", data)
        results = output.get("results", [])
        if not isinstance(results, list):
            raise RuntimeError("reranker response does not contain a results list")
        return results

    @staticmethod
    def _apply_results(hits: list[SearchHit], results: list[dict], top_k: int) -> list[SearchHit]:
        ranked: list[SearchHit] = []
        seen: set[int] = set()

        for result in results:
            if not isinstance(result, dict):
                continue
            index = result.get("index")
            if index is None:
                continue
            try:
                candidate_index = int(index)
            except (TypeError, ValueError):
                continue
            if candidate_index in seen or candidate_index < 0 or candidate_index >= len(hits):
                continue

            source_hit = hits[candidate_index]
            metadata = dict(source_hit.metadata)
            score = result.get("relevance_score")
            if score is not None:
                metadata["rerankerScore"] = score
            ranked.append(
                SearchHit(
                    text=source_hit.text,
                    score=float(score) if isinstance(score, int | float) else source_hit.score,
                    source=f"{source_hit.source}+rerank",
                    metadata=metadata,
                )
            )
            seen.add(candidate_index)

        for index, hit in enumerate(hits):
            if index not in seen:
                ranked.append(hit)

        return ranked[:top_k]
