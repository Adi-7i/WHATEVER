from __future__ import annotations

from dataclasses import dataclass
import asyncio
import re

import httpx
from bs4 import BeautifulSoup


@dataclass
class ScrapedArticle:
    url: str
    title: str
    content: str
    word_count: int


class ArticleScraper:
    def __init__(
        self,
        timeout_seconds: int = 15,
        max_words: int = 1200,
        concurrency: int = 4,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_words = max_words
        self.concurrency = concurrency

    async def scrape_many(self, urls: list[str]) -> list[ScrapedArticle]:
        semaphore = asyncio.Semaphore(self.concurrency)

        async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
            tasks = [self._scrape_single(url, client, semaphore) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        articles: list[ScrapedArticle] = []
        for item in results:
            if isinstance(item, ScrapedArticle):
                articles.append(item)
        return articles

    async def _scrape_single(
        self,
        url: str,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
    ) -> ScrapedArticle | None:
        async with semaphore:
            try:
                response = await client.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; telegram-news-bot/1.0)"},
                )
                response.raise_for_status()
                if "html" not in (response.headers.get("content-type") or "").lower():
                    return None

                title, content = self._extract_clean_text(response.text)
                if not content:
                    return None

                words = content.split()
                clipped = " ".join(words[: self.max_words])
                return ScrapedArticle(
                    url=url,
                    title=title,
                    content=clipped,
                    word_count=min(len(words), self.max_words),
                )
            except Exception:
                return None

    def _extract_clean_text(self, html: str) -> tuple[str, str]:
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript", "iframe", "svg", "footer", "nav", "form"]):
            tag.decompose()

        title = (soup.title.string or "").strip() if soup.title and soup.title.string else "Untitled"

        article_node = soup.find("article")
        root = article_node if article_node else soup.body or soup

        chunks: list[str] = []
        for tag in root.find_all(["h1", "h2", "h3", "p", "li"]):
            text = tag.get_text(" ", strip=True)
            if not text:
                continue
            if len(text) < 30 and tag.name == "li":
                continue
            chunks.append(text)

        text = "\n".join(chunks)
        text = re.sub(r"\s+", " ", text).strip()
        return title, text
