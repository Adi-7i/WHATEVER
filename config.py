import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    openai_api_key: str
    openai_model: str
    openai_base_url: str

    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str
    azure_openai_deployment: str

    searxng_base_url: str

    request_timeout_seconds: int
    searxng_timeout_seconds: int
    scraper_timeout_seconds: int
    scraper_concurrency: int

    searxng_max_results: int
    article_word_limit: int
    chunk_token_target: int


def _get_required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer") from exc


def load_settings() -> Settings:
    return Settings(
        telegram_bot_token=_get_required("TELEGRAM_BOT_TOKEN"),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip(),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "").strip(),
        azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "").strip(),
        azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY", "").strip(),
        azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview").strip(),
        azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip(),
        searxng_base_url=os.getenv("SEARXNG_BASE_URL", "https://searxng.cynerza.in").strip(),
        request_timeout_seconds=_get_int("REQUEST_TIMEOUT_SECONDS", 60),
        searxng_timeout_seconds=_get_int("SEARXNG_TIMEOUT_SECONDS", 20),
        scraper_timeout_seconds=_get_int("SCRAPER_TIMEOUT_SECONDS", 15),
        scraper_concurrency=_get_int("SCRAPER_CONCURRENCY", 4),
        searxng_max_results=_get_int("SEARXNG_MAX_RESULTS", 8),
        article_word_limit=_get_int("ARTICLE_WORD_LIMIT", 1200),
        chunk_token_target=_get_int("CHUNK_TOKEN_TARGET", 1000),
    )


def validate_settings(settings: Settings) -> None:
    has_azure = bool(
        settings.azure_openai_endpoint
        and settings.azure_openai_api_key
        and settings.azure_openai_deployment
    )
    has_openai = bool(settings.openai_api_key)
    if not has_azure and not has_openai:
        raise ValueError(
            "Set either OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY + AZURE_OPENAI_DEPLOYMENT"
        )
