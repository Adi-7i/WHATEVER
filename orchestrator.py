from __future__ import annotations

from dataclasses import dataclass

from openai import AsyncAzureOpenAI, AsyncOpenAI

from config import Settings
from deep_research import configure_deep_research, run_deep_research
from intent import IntentType, RuleBasedIntentClassifier
from scraper import ArticleScraper
from search import SearXNGClient
from summarizer import Summarizer


@dataclass
class OrchestrationResult:
    intent: IntentType
    optimized_query: str
    search_results_count: int
    scraped_articles_count: int
    response_text: str


class QueryOrchestrator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.classifier = RuleBasedIntentClassifier()
        self.search_client = SearXNGClient(
            base_url=settings.searxng_base_url,
            timeout_seconds=settings.searxng_timeout_seconds,
        )
        self.scraper = ArticleScraper(
            timeout_seconds=settings.scraper_timeout_seconds,
            max_words=settings.article_word_limit,
            concurrency=settings.scraper_concurrency,
        )
        self.deep_scraper = ArticleScraper(
            timeout_seconds=settings.scraper_timeout_seconds,
            max_words=1500,
            concurrency=settings.scraper_concurrency,
        )

        llm_client, model = self._build_llm_client(settings)
        self.summarizer = Summarizer(client=llm_client, model=model)
        configure_deep_research(
            search_client=self.search_client,
            scraper=self.deep_scraper,
            client=llm_client,
            model=model,
        )

    async def handle_query(self, user_text: str) -> OrchestrationResult:
        intent = self.classifier.classify(user_text)

        if intent != IntentType.REALTIME_NEWS:
            generic = await self.summarizer.generic_response(user_text)
            return OrchestrationResult(
                intent=intent,
                optimized_query="",
                search_results_count=0,
                scraped_articles_count=0,
                response_text=generic,
            )

        optimized_query = self.search_client.optimize_query(user_text)
        search_results = await self.search_client.search(
            query=optimized_query,
            limit=self.settings.searxng_max_results,
        )

        urls = self.search_client.extract_urls(search_results)
        scraped_articles = await self.scraper.scrape_many(urls)

        article_summaries = []
        for article in scraped_articles:
            summary = await self.summarizer.summarize_article(
                title=article.title,
                url=article.url,
                content=article.content,
                chunk_tokens=self.settings.chunk_token_target,
            )
            article_summaries.append(summary)

        final = await self.summarizer.build_final_events_response(
            user_query=user_text,
            optimized_query=optimized_query,
            article_summaries=article_summaries,
        )

        return OrchestrationResult(
            intent=intent,
            optimized_query=optimized_query,
            search_results_count=len(search_results),
            scraped_articles_count=len(scraped_articles),
            response_text=final,
        )

    async def handle_deep_research(self, user_text: str) -> str:
        return await run_deep_research(user_text)

    @staticmethod
    def _build_llm_client(settings: Settings) -> tuple[AsyncOpenAI | AsyncAzureOpenAI, str]:
        has_azure = bool(
            settings.azure_openai_endpoint
            and settings.azure_openai_api_key
            and settings.azure_openai_deployment
        )
        if has_azure:
            client = AsyncAzureOpenAI(
                api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
            )
            return client, settings.azure_openai_deployment

        client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
        )
        return client, settings.openai_model
