from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re

import httpx


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str


class SearXNGClient:
    def __init__(self, base_url: str, timeout_seconds: int = 20) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def search(self, query: str, limit: int = 8) -> list[SearchResult]:
        params = {
            "q": query,
            "format": "json",
            "language": "auto",
            "time_range": "day",
        }
        url = f"{self.base_url}/search"

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()

        results: list[SearchResult] = []
        seen: set[str] = set()
        for item in payload.get("results", []):
            raw_url = (item.get("url") or "").strip()
            if not raw_url or raw_url in seen:
                continue
            seen.add(raw_url)

            results.append(
                SearchResult(
                    title=self._clean(item.get("title", ""), 220),
                    url=raw_url,
                    snippet=self._clean(item.get("content", ""), 400),
                    source=self._clean(item.get("engine", ""), 64),
                )
            )
            if len(results) >= limit:
                break

        return results

    def optimize_query(self, user_text: str) -> str:
        """Build a retrieval-friendly query and avoid forwarding raw user text as-is."""
        cleaned = self._clean(user_text, 280).lower()
        words = re.findall(r"[a-z0-9]+", cleaned)

        stopwords = {
            "the", "a", "an", "is", "are", "am", "i", "we", "you", "he", "she", "it", "they",
            "please", "tell", "me", "about", "what", "when", "why", "how", "can", "could", "would",
            "should", "do", "does", "did", "latest", "today", "news", "update", "updates", "now",
        }

        terms = [w for w in words if w not in stopwords and len(w) > 2]
        if not terms:
            terms = ["global", "top", "headlines"]

        terms = terms[:8]
        date_hint = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        optimized = " ".join(terms)
        optimized = f"{optimized} latest developments {date_hint}"

        # Guard: ensure we are not passing raw text unchanged.
        raw_normalized = re.sub(r"\s+", " ", user_text.strip().lower())
        optimized_normalized = re.sub(r"\s+", " ", optimized.strip().lower())
        if optimized_normalized == raw_normalized:
            optimized = f"current verified update {optimized}"

        return optimized

    @staticmethod
    def extract_urls(results: list[SearchResult]) -> list[str]:
        return [r.url for r in results if r.url]

    @staticmethod
    def _clean(text: str, max_chars: int) -> str:
        text = re.sub(r"\s+", " ", (text or "")).strip()
        return text[:max_chars]
