from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import re
import threading
import time
from collections import defaultdict

import telebot
from openai import AsyncAzureOpenAI, AsyncOpenAI
from telebot import apihelper

from config import Settings, load_settings, validate_settings
from orchestrator import QueryOrchestrator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

settings = load_settings()
validate_settings(settings)

bot = telebot.TeleBot(settings.telegram_bot_token, parse_mode="Markdown")
orchestrator = QueryOrchestrator(settings=settings)

locks = defaultdict(threading.Lock)


def _build_async_llm_client(cfg: Settings) -> tuple[AsyncOpenAI | AsyncAzureOpenAI, str]:
    has_azure = bool(
        cfg.azure_openai_endpoint
        and cfg.azure_openai_api_key
        and cfg.azure_openai_deployment
    )
    if has_azure:
        return (
            AsyncAzureOpenAI(
                api_key=cfg.azure_openai_api_key,
                azure_endpoint=cfg.azure_openai_endpoint,
                api_version=cfg.azure_openai_api_version,
            ),
            cfg.azure_openai_deployment,
        )

    return (
        AsyncOpenAI(
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url or None,
        ),
        cfg.openai_model,
    )


vision_client, vision_model = _build_async_llm_client(settings)


IDENTITY_RESPONSE = (
    "*WilloFire*\n"
    "Created by: lucifer\n"
    "Powered by: Cynerza Systems Private Limited\n\n"
    "WilloFire is a professional AI assistant with:\n"
    "- Smart AI Assistance\n"
    "- Deep Research (`/deep <query>`)\n"
    "- Advanced Code Debugging\n"
    "- Image Understanding and Analysis\n"
    "- Multilingual Communication"
)


START_MESSAGE = (
    "Welcome to *WilloFire* 🔥\n\n"
    "Created by: lucifer\n"
    "Powered by: Cynerza Systems Private Limited\n\n"
    "Core Features:\n"
    "- Smart AI Assistant\n"
    "- Deep Research Mode (`/deep <query>`)\n"
    "- Advanced Code Debugging\n"
    "- Image Understanding & Analysis\n"
    "- Multilingual Support\n\n"
    "Understands and communicates fluently in English and Hinglish, "
    "and supports multiple languages."
)


IDENTITY_PATTERNS = [
    r"\bwho made you\b",
    r"\bwho created you\b",
    r"\bwhat is your name\b",
    r"\bwho are you\b",
    r"\bwho is your developer\b",
    r"\bwhat is willofire\b",
]


CODE_HINT_PATTERNS = [
    r"\bdef\s+\w+\s*\(",
    r"\bclass\s+\w+",
    r"\bimport\s+\w+",
    r"\bfor\s+\w+\s+in\s+",
    r"\bwhile\s+",
    r"\bprint\s*\(",
    r"\btraceback\b",
]


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


def is_identity_query(text: str) -> bool:
    lowered = (text or "").strip().lower()
    return any(re.search(pattern, lowered) for pattern in IDENTITY_PATTERNS)


def is_debug_query(text: str) -> bool:
    if not text:
        return False

    lowered = text.lower()
    if "traceback" in lowered or "error" in lowered:
        return True

    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False

    indented_lines = sum(1 for line in lines if line.startswith("    ") or line.startswith("\t"))
    code_like_lines = 0
    for line in lines:
        compact = line.strip()
        if re.match(r"^(def|class|import|from|for|while|if|elif|else|try|except|finally|print\()\b", compact):
            code_like_lines += 1
            continue
        if "=" in compact and not compact.startswith("http"):
            code_like_lines += 1

    return indented_lines >= 1 or code_like_lines >= 3


def build_debug_prompt(user_input: str) -> str:
    return (
        "You are an expert Python debugger.\n"
        "When user provides buggy code:\n"
        "- Identify issue clearly.\n"
        "- Provide corrected FULL code.\n"
        "- Explain fix simply.\n"
        "- Preserve original intent.\n"
        "- Use proper markdown formatting.\n\n"
        "Return in this exact format:\n"
        "🔎 Issue Found:\n"
        "<clear explanation>\n\n"
        "🛠 Fixed Code:\n"
        "<corrected full code>\n\n"
        "📌 Explanation:\n"
        "<why error occurred and how fix resolves it>\n\n"
        "User content:\n"
        f"{user_input}"
    )


def format_code_blocks(text: str) -> str:
    if not text:
        return text
    if "```" in text:
        return text

    has_code_signal = any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in CODE_HINT_PATTERNS)
    if not has_code_signal:
        return text

    fixed_header = "🛠 Fixed Code:"
    explain_header = "📌 Explanation:"

    if fixed_header in text and explain_header in text:
        before, after_fixed = text.split(fixed_header, maxsplit=1)
        code_part, after_explain = after_fixed.split(explain_header, maxsplit=1)
        code_block = code_part.strip()
        wrapped = f"```python\n{code_block}\n```"
        return f"{before.strip()}\n\n{fixed_header}\n{wrapped}\n\n{explain_header}{after_explain}"

    return f"```python\n{text.strip()}\n```"


def _is_image_document(filename: str, mime_type: str) -> bool:
    lower = (filename or "").lower()
    return mime_type.startswith("image/") or lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"))


async def _analyze_image_async(image_bytes: bytes, mime_type: str, user_caption: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt = user_caption.strip() or "Analyze this image."

    response = await vision_client.chat.completions.create(
        model=vision_model,
        temperature=0.2,
        max_tokens=1200,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are WilloFire, created by lucifer and powered by Cynerza Systems Private Limited. "
                    "Provide professional structured image analysis. "
                    "Do not hallucinate. If unclear, say: 'Some parts are unclear, based on visible elements...'. "
                    "If image includes code/errors, provide corrected code and debugging explanation. "
                    "Use this exact format:\n"
                    "🖼 Image Analysis:\n"
                    "📖 Extracted Content:\n"
                    "🧠 Interpretation:\n"
                    "🛠 If image shows error or code:\n"
                    "Never say you are an AI model. Never mention OpenAI."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                    },
                ],
            },
        ],
    )

    return response.choices[0].message.content or "Some parts are unclear, based on visible elements..."


def run_image_analysis(image_bytes: bytes, mime_type: str, user_caption: str) -> str:
    try:
        return asyncio.run(_analyze_image_async(image_bytes, mime_type, user_caption))
    except Exception:
        logging.exception("image analysis failed")
        return "Some parts are unclear, based on visible elements..."


@bot.message_handler(commands=["start", "help"])
def handle_start(message):
    bot.reply_to(message, START_MESSAGE)


@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    user_id = message.from_user.id
    chat_id = message.chat.id

    with locks[user_id]:
        try:
            photo = message.photo[-1]
            file_info = bot.get_file(photo.file_id)
            image_bytes = bot.download_file(file_info.file_path)
            caption = (message.caption or "").strip()
            result = run_image_analysis(image_bytes=image_bytes, mime_type="image/jpeg", user_caption=caption)
            reply_large(chat_id, format_code_blocks(result))
        except Exception as exc:
            logging.exception("photo handler failed")
            bot.send_message(
                chat_id,
                "Some parts are unclear, based on visible elements... "
                f"Error: {str(exc)[:160]}",
            )


@bot.message_handler(content_types=["document"])
def handle_document(message):
    user_id = message.from_user.id
    chat_id = message.chat.id

    with locks[user_id]:
        try:
            doc = message.document
            filename = doc.file_name or "uploaded_file"
            mime_type = doc.mime_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"

            if not _is_image_document(filename, mime_type):
                bot.send_message(chat_id, "Please upload an image file for image analysis.")
                return

            file_info = bot.get_file(doc.file_id)
            image_bytes = bot.download_file(file_info.file_path)
            caption = (message.caption or "").strip()
            result = run_image_analysis(image_bytes=image_bytes, mime_type=mime_type, user_caption=caption)
            reply_large(chat_id, format_code_blocks(result))
        except Exception as exc:
            logging.exception("document handler failed")
            bot.send_message(
                chat_id,
                "Some parts are unclear, based on visible elements... "
                f"Error: {str(exc)[:160]}",
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
                chunks = split_message(format_code_blocks(report))
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

    if is_identity_query(query):
        bot.send_message(chat_id, IDENTITY_RESPONSE)
        return

    with locks[user_id]:
        try:
            final_query = build_debug_prompt(query) if is_debug_query(query) else query
            response_text = run_orchestration(final_query)
            reply_large(chat_id, format_code_blocks(response_text))
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
