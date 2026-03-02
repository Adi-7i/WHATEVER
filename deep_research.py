from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from openai import AsyncAzureOpenAI, AsyncOpenAI

from scraper import ArticleScraper, ScrapedArticle
from search import SearXNGClient


logger = logging.getLogger(__name__)


@dataclass
class SourceSummary:
    url: str
    title: str
    summary: str


class DeepResearchService:
    def __init__(
        self,
        search_client: SearXNGClient,
        scraper: ArticleScraper,
        client: AsyncOpenAI | AsyncAzureOpenAI,
        model: str,
    ) -> None:
        self.search_client = search_client
        self.scraper = scraper
        self.client = client
        self.model = model

    async def run(self, query: str) -> str:
        logger.info("deep_research.start query=%s", query[:240])

        research_queries = await self._plan_queries(query)
        logger.info("deep_research.planner generated=%s", len(research_queries))

        urls = await self._multi_search_urls(research_queries)
        logger.info("deep_research.search urls=%s", len(urls))

        articles = await self.scraper.scrape_many(urls)
        logger.info("deep_research.scrape success=%s", len(articles))

        source_summaries = await self._summarize_sources(articles)
        logger.info("deep_research.summaries count=%s", len(source_summaries))

        cross_verification = await self._cross_verify(source_summaries)
        report = await self._final_report(
            query=query,
            research_queries=research_queries,
            source_summaries=source_summaries,
            cross_verification=cross_verification,
        )

        return self._enforce_no_unavailable(report)

    async def _plan_queries(self, query: str) -> list[str]:
        prompt = (
            "Break this topic into 5 to 8 focused web research queries for deep fact-finding. "
            "Return only a JSON array of strings. No explanation.\n\n"
            f"Topic: {query}"
        )
        raw = await self._chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are a research planner. Output only valid JSON array.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=300,
        )

        parsed = self._parse_json_array(raw)
        if not parsed:
            fallback = self.search_client.optimize_query(query)
            return [fallback]

        cleaned = [self._clean_line(item, 180) for item in parsed if self._clean_line(item, 180)]
        return cleaned[:8] if cleaned else [self.search_client.optimize_query(query)]

    async def _multi_search_urls(self, queries: list[str]) -> list[str]:
        tasks = [self.search_client.search(query=q, limit=5) for q in queries]
        try:
            batches = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception:
            logger.exception("deep_research.multi_search gather failure")
            return []

        urls: list[str] = []
        seen: set[str] = set()

        for batch in batches:
            if isinstance(batch, Exception):
                logger.warning("deep_research.search query failed: %s", str(batch)[:200])
                continue
            for result in batch:
                if result.url in seen:
                    continue
                seen.add(result.url)
                urls.append(result.url)
                if len(urls) >= 30:
                    return urls
        return urls

    async def _summarize_sources(self, articles: list[ScrapedArticle]) -> list[SourceSummary]:
        tasks = [self._summarize_source(article) for article in articles]
        outputs = await asyncio.gather(*tasks, return_exceptions=True)

        summaries: list[SourceSummary] = []
        for item in outputs:
            if isinstance(item, SourceSummary):
                summaries.append(item)
            elif isinstance(item, Exception):
                logger.warning("deep_research.source_summary failed: %s", str(item)[:200])

        return summaries

    async def _summarize_source(self, article: ScrapedArticle) -> SourceSummary:
        content = self._clean_line(article.content, 12000)
        raw = await self._chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize one source in professional neutral tone. "
                        "Return plain text with sections exactly:\n"
                        "Key Facts:\n"
                        "Dates:\n"
                        "Data Points:\n"
                        "Claims:\n"
                        "Keep under 300 words. "
                        "Never use the phrase 'information not available'."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Title: {article.title}\n"
                        f"URL: {article.url}\n"
                        f"Article text:\n{content}"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=520,
        )

        return SourceSummary(
            url=article.url,
            title=article.title,
            summary=self._enforce_no_unavailable(raw),
        )

    async def _cross_verify(self, summaries: list[SourceSummary]) -> str:
        if not summaries:
            return (
                "Verified Facts:\n"
                "- Available evidence indicates no source-level confirmations yet.\n"
                "Conflicting Claims:\n"
                "- No direct conflicts identified from fetched sources.\n"
                "Weakly Supported Claims:\n"
                "- Most claims remain weak due to limited successfully scraped material."
            )

        payload = "\n\n".join(
            f"Source: {item.title}\nURL: {item.url}\n{item.summary}"
            for item in summaries
        )

        raw = await self._chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Cross-verify claims across multiple source summaries. "
                        "Return plain text with exact sections:\n"
                        "Verified Facts (2+ Sources):\n"
                        "Conflicting Claims:\n"
                        "Weakly Supported Claims:\n"
                        "Do not invent data. Never use 'information not available'."
                    ),
                },
                {"role": "user", "content": payload},
            ],
            temperature=0.1,
            max_tokens=1200,
        )

        return self._enforce_no_unavailable(raw)

    async def _final_report(
        self,
        query: str,
        research_queries: list[str],
        source_summaries: list[SourceSummary],
        cross_verification: str,
    ) -> str:
        source_count = len(source_summaries)
        query_block = "\n".join(f"- {q}" for q in research_queries)

        source_brief = "\n\n".join(
            f"Title: {s.title}\nURL: {s.url}\nSummary:\n{s.summary}"
            for s in source_summaries
        )
        source_brief = self._clean_line(source_brief, 32000)

        raw = await self._chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a deep research report in professional neutral tone. "
                        "No hallucinated statistics. Use verified facts where possible. "
                        "If evidence is limited, state reduced confidence. "
                        "Never use 'information not available'. "
                        "Output plain text with exactly these sections:\n"
                        "Executive Summary\n"
                        "Background\n"
                        "Current Developments\n"
                        "Verified Key Facts\n"
                        "Conflicting Narratives\n"
                        "Strategic Implications\n"
                        "Conclusion\n"
                        "Confidence Level"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research question: {query}\n"
                        f"Planner queries used:\n{query_block}\n\n"
                        f"Total validated source summaries: {source_count}\n\n"
                        f"Cross verification output:\n{cross_verification}\n\n"
                        f"Source summaries:\n{source_brief}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=1500,
        )

        return self._enforce_no_unavailable(raw)

    async def _chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            logger.exception("deep_research.llm_call failed")
            return "Available evidence indicates processing degradation; partial outputs may have reduced confidence."

    @staticmethod
    def _parse_json_array(raw: str) -> list[str]:
        if not raw:
            return []

        text = raw.strip()
        try:
            value = json.loads(text)
            if isinstance(value, list):
                return [str(v) for v in value]
        except Exception:
            pass

        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            try:
                value = json.loads(match.group(0))
                if isinstance(value, list):
                    return [str(v) for v in value]
            except Exception:
                return []

        lines = [line.strip("-• ").strip() for line in text.splitlines() if line.strip()]
        return [line for line in lines if line]

    @staticmethod
    def _clean_line(text: str, max_chars: int) -> str:
        compact = re.sub(r"\s+", " ", (text or "")).strip()
        return compact[:max_chars]

    @staticmethod
    def _enforce_no_unavailable(text: str) -> str:
        return re.sub(
            r"information not available",
            "available evidence indicates limited certainty",
            text,
            flags=re.IGNORECASE,
        )


_DEEP_RESEARCH_SERVICE: DeepResearchService | None = None


def configure_deep_research(
    search_client: SearXNGClient,
    scraper: ArticleScraper,
    client: AsyncOpenAI | AsyncAzureOpenAI,
    model: str,
) -> None:
    global _DEEP_RESEARCH_SERVICE
    _DEEP_RESEARCH_SERVICE = DeepResearchService(
        search_client=search_client,
        scraper=scraper,
        client=client,
        model=model,
    )


async def run_deep_research(query: str) -> str:
    if _DEEP_RESEARCH_SERVICE is None:
        raise RuntimeError("Deep research service is not configured")
    return await _DEEP_RESEARCH_SERVICE.run(query)
