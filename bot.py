from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import defaultdict

import telebot
from telebot import apihelper

from config import load_settings, validate_settings
from orchestrator import QueryOrchestrator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

settings = load_settings()
validate_settings(settings)

bot = telebot.TeleBot(settings.telegram_bot_token, parse_mode=None)
orchestrator = QueryOrchestrator(settings=settings)

locks = defaultdict(threading.Lock)


def split_message(text: str, max_chars: int = 3900) -> list[str]:
    payload = (text or "").strip() or "Available evidence indicates partial certainty; generated best effort answer."
    return [payload[i : i + max_chars] for i in range(0, len(payload), max_chars)]


def reply_large(chat_id: int, text: str) -> None:
    for part in split_message(text):
        bot.send_message(chat_id, part)


def run_orchestration(query: str) -> str:
    result = asyncio.run(orchestrator.handle_query(query))
    logging.info(
        "intent=%s optimized_query=%s search_results=%s scraped_articles=%s",
        result.intent,
        result.optimized_query,
        result.search_results_count,
        result.scraped_articles_count,
    )
    return result.response_text


def run_deep_orchestration(query: str) -> str:
    return asyncio.run(orchestrator.handle_deep_research(query))


@bot.message_handler(commands=["start", "help"])
def handle_start(message):
    bot.reply_to(
        message,
        "Bot ready. Send a query.\nRealtime queries use search+scrape+summarization orchestration.",
    )


@bot.message_handler(content_types=["text"])
def handle_text(message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    query = (message.text or "").strip()

    if not query:
        bot.reply_to(message, "Send a valid text query.")
        return

    if query.startswith("/deep"):
        parts = query.split(maxsplit=1)
        deep_query = parts[1].strip() if len(parts) > 1 else ""
        if not deep_query:
            bot.reply_to(message, "Usage: /deep <query>")
            return

        with locks[user_id]:
            status = bot.send_message(chat_id, "Deep research in progress...")
            try:
                report = run_deep_orchestration(deep_query)
                chunks = split_message(report)
                bot.edit_message_text(
                    chunks[0],
                    chat_id=chat_id,
                    message_id=status.message_id,
                )
                for extra in chunks[1:]:
                    bot.send_message(chat_id, extra)
            except Exception as exc:
                logging.exception("deep research failed")
                bot.edit_message_text(
                    (
                        "Available evidence indicates a temporary deep-research issue. "
                        "Please retry with a narrower query. "
                        f"Error: {str(exc)[:200]}"
                    ),
                    chat_id=chat_id,
                    message_id=status.message_id,
                )
        return

    with locks[user_id]:
        try:
            response_text = run_orchestration(query)
            reply_large(chat_id, response_text)
        except Exception as exc:
            logging.exception("orchestration failed")
            fallback = (
                "Available evidence indicates a temporary processing issue. "
                "Please retry with a slightly more specific query. "
                f"Error: {str(exc)[:200]}"
            )
            bot.send_message(chat_id, fallback)


if __name__ == "__main__":
    apihelper.RETRY_ON_ERROR = True
    apihelper.RETRY_TIMEOUT = 2

    logging.info("Bot started")
    backoff_seconds = 2
    while True:
        try:
            bot.infinity_polling(
                timeout=20,
                long_polling_timeout=20,
                skip_pending=True,
                allowed_updates=["message"],
            )
            backoff_seconds = 2
        except Exception:
            logging.exception("Polling crashed; restarting in %s seconds", backoff_seconds)
            time.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, 30)
