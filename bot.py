from __future__ import annotations

import asyncio
import base64
import html
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

bot = telebot.TeleBot(settings.telegram_bot_token, parse_mode=None)
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
    "WilloFire\n"
    "Created by: lucifer\n"
    "Powered by: Cynerza Systems Private Limited\n\n"
    "WilloFire is a professional AI assistant with:\n"
    "- Smart AI Assistance\n"
    "- Deep Research (/deep <query>)\n"
    "- Advanced Code Debugging\n"
    "- Image Understanding and Analysis\n"
    "- Multilingual Communication"
)


START_MESSAGE = (
    "Welcome to WilloFire\n\n"
    "Created by: lucifer\n"
    "Powered by: Cynerza Systems Private Limited\n\n"
    "Core Features:\n"
    "- Smart AI Assistant\n"
    "- Deep Research Mode (/deep <query>)\n"
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


DEEP_SECTION_ORDER = [
    "Executive Summary",
    "Background",
    "Key Developments",
    "Strategic Implications",
    "Conclusion",
    "Confidence Level",
]


DEEP_SECTION_ALIASES = {
    "executive summary": "Executive Summary",
    "background": "Background",
    "key developments": "Key Developments",
    "strategic implications": "Strategic Implications",
    "conclusion": "Conclusion",
    "confidence": "Confidence Level",
    "confidence level": "Confidence Level",
}


def _normalize_newlines(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def split_message(text: str, max_chars: int = 3900) -> list[str]:
    payload = _normalize_newlines(text) or "Available evidence indicates partial certainty; generated best effort answer."
    if len(payload) <= max_chars:
        return [payload]
    return [payload[i : i + max_chars] for i in range(0, len(payload), max_chars)]


def split_sections(text: str, max_chars: int = 3900) -> list[str]:
    payload = _normalize_newlines(text) or "Available evidence indicates partial certainty; generated best effort answer."
    if len(payload) <= max_chars:
        return [payload]

    blocks = [b.strip() for b in payload.split("\n\n") if b.strip()]
    chunks: list[str] = []
    current = ""

    for block in blocks:
        candidate = block if not current else f"{current}\n\n{block}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(block) <= max_chars:
            current = block
            continue

        over = split_message(block, max_chars=max_chars)
        chunks.extend(over[:-1])
        current = over[-1]

    if current:
        chunks.append(current)

    return chunks or [payload[:max_chars]]


def _sanitize_plain(text: str) -> str:
    payload = _normalize_newlines(text)
    payload = re.sub(r"</?b>", "", payload, flags=re.IGNORECASE)
    payload = re.sub(r"</?i>", "", payload, flags=re.IGNORECASE)
    payload = re.sub(r"</?pre>", "", payload, flags=re.IGNORECASE)
    payload = re.sub(r"</?code>", "", payload, flags=re.IGNORECASE)
    return payload


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


def _is_indented_block(text: str) -> bool:
    lines = [line for line in _normalize_newlines(text).splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    return any(line.startswith("    ") or line.startswith("\t") for line in lines)


def _looks_like_code(text: str) -> bool:
    source = _normalize_newlines(text)
    if not source:
        return False

    lowered = source.lower()
    if "traceback" in lowered or "syntaxerror" in lowered:
        return True

    if _is_indented_block(source):
        return True

    return any(re.search(pattern, source, flags=re.IGNORECASE) for pattern in CODE_HINT_PATTERNS)


def _normalize_section_key(line: str) -> str:
    cleaned = line.strip().rstrip(":").strip()
    return DEEP_SECTION_ALIASES.get(cleaned.lower(), "")


def _extract_sections(text: str) -> dict[str, str]:
    lines = _normalize_newlines(text).splitlines()
    section_lines: dict[str, list[str]] = {key: [] for key in DEEP_SECTION_ORDER}
    current_section = ""

    for line in lines:
        maybe_section = _normalize_section_key(line)
        if maybe_section:
            current_section = maybe_section
            continue
        if current_section:
            section_lines[current_section].append(line)

    result: dict[str, str] = {}
    for key in DEEP_SECTION_ORDER:
        value = "\n".join(section_lines[key]).strip()
        if value:
            result[key] = value

    return result


def _looks_like_deep_structured(text: str) -> bool:
    payload = _normalize_newlines(text)
    if not payload:
        return False

    lines = [line.strip().rstrip(":").strip().lower() for line in payload.splitlines() if line.strip()]
    if not lines:
        return False

    if lines[0] in DEEP_SECTION_ALIASES:
        return True

    section_hits = sum(1 for line in lines[:20] if line in DEEP_SECTION_ALIASES)
    return section_hits >= 2


def _escape_html_text(text: str) -> str:
    return html.escape(_normalize_newlines(text), quote=False)


def _normalize_bullet_line(line: str) -> str:
    content = re.sub(r"^[\-\*\u2022]\s*", "", line.strip())
    return f"• {content}" if content else ""


def format_deep_research_html(text: str) -> str:
    sections = _extract_sections(text)
    if not sections:
        sections["Executive Summary"] = _normalize_newlines(text) or "Not available."

    out: list[str] = []
    for section in DEEP_SECTION_ORDER:
        content = sections.get(section, "").strip()
        if not content:
            continue

        out.append(f"<b>{section}</b>")
        out.append("")

        if section == "Key Developments":
            lines = [line for line in content.splitlines() if line.strip()]
            if not lines:
                lines = [content]
            for line in lines:
                out.append(_escape_html_text(_normalize_bullet_line(line)))
        elif section == "Confidence Level":
            confidence_line = content.splitlines()[0].strip()
            rest = "\n".join(content.splitlines()[1:]).strip()
            out.append(_escape_html_text(confidence_line))
            if rest:
                out.append("")
                out.append(_escape_html_text(rest))
        else:
            out.append(_escape_html_text(content))

        out.append("")

    return "\n".join(out).strip()


def format_code_html(text: str) -> list[str]:
    chunks = split_message(text, max_chars=3400)
    return [f"<pre><code>{html.escape(chunk, quote=False)}</code></pre>" for chunk in chunks]


def format_debug_html(text: str) -> str:
    payload = _normalize_newlines(text)

    issue_match = re.search(
        r"(Issue Found:|🔎\s*Issue Found:)(.*?)(Fixed Code:|🛠\s*Fixed Code:)",
        payload,
        flags=re.IGNORECASE | re.DOTALL,
    )
    code_match = re.search(
        r"(Fixed Code:|🛠\s*Fixed Code:)(.*?)(Explanation:|📌\s*Explanation:)",
        payload,
        flags=re.IGNORECASE | re.DOTALL,
    )
    explanation_match = re.search(
        r"(Explanation:|📌\s*Explanation:)(.*)$",
        payload,
        flags=re.IGNORECASE | re.DOTALL,
    )

    issue = issue_match.group(2).strip() if issue_match else "Code issue detected."
    fixed_code = code_match.group(2).strip() if code_match else payload
    explanation = explanation_match.group(2).strip() if explanation_match else "Applied a safe correction."

    return (
        "<b>Issue Found</b>\n\n"
        f"{_escape_html_text(issue)}\n\n"
        "<b>Fixed Code</b>\n\n"
        f"<pre><code>{html.escape(fixed_code, quote=False)}</code></pre>\n\n"
        "<b>Explanation</b>\n\n"
        f"{_escape_html_text(explanation)}"
    )


def detect_message_type(text: str, message_type: str = "normal") -> str:
    requested = (message_type or "normal").strip().lower()
    if requested in {"deep", "code", "debug", "image"}:
        return requested

    payload = _normalize_newlines(text)
    if not payload:
        return "normal"

    if "<b>image analysis</b>" in payload.lower():
        return "image"

    if _looks_like_code(payload):
        return "code"

    if _looks_like_deep_structured(payload):
        return "deep"

    return "normal"


async def _send_message_async(chat_id: int, text: str, parse_mode: str | None = None):
    return await asyncio.to_thread(bot.send_message, chat_id, text, parse_mode=parse_mode)


async def safe_send_message(chat_id: int, text: str, message_type: str = "normal"):
    payload = _normalize_newlines(text) or "Available evidence indicates partial certainty; generated best effort answer."
    resolved_type = detect_message_type(payload, message_type=message_type)

    parse_mode: str | None
    chunks: list[str]

    if resolved_type == "code":
        parse_mode = "HTML"
        chunks = format_code_html(payload)
    elif resolved_type == "deep":
        parse_mode = "HTML"
        chunks = split_sections(format_deep_research_html(payload), max_chars=3800)
    elif resolved_type == "debug":
        parse_mode = "HTML"
        chunks = split_sections(format_debug_html(payload), max_chars=3800)
    elif resolved_type == "image":
        parse_mode = "HTML"
        chunks = split_sections(payload, max_chars=3800)
    else:
        parse_mode = None
        chunks = split_message(_sanitize_plain(payload), max_chars=3900)

    for chunk in chunks:
        try:
            await _send_message_async(chat_id, chunk, parse_mode=parse_mode)
        except Exception:
            logging.exception("Telegram send failed with parse_mode=%s; retrying plain text", parse_mode)
            plain = _sanitize_plain(html.unescape(re.sub(r"<[^>]+>", "", chunk)))
            await _send_message_async(
                chat_id,
                plain or "Available evidence indicates partial certainty; generated best effort answer.",
                parse_mode=None,
            )


def _is_image_document(filename: str, mime_type: str) -> bool:
    lower = (filename or "").lower()
    return mime_type.startswith("image/") or lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"))


async def handle_smart_image(image_bytes: bytes) -> str:
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    prompt = (
        "You are an expert technical image analyst.\n"
        "Analyze the image thoroughly.\n"
        "Extract text precisely.\n"
        "If code or error present, debug it.\n"
        "Do not hallucinate.\n"
        "If something unclear, mention uncertainty.\n\n"
        "Always respond using SAFE HTML formatting.\n"
        "Allowed tags only:\n"
        "<b>\n"
        "<pre>\n"
        "<code>\n\n"
        "Do NOT use MarkdownV2.\n"
        "Do NOT use unsupported HTML tags.\n"
        "Do NOT use bullet hyphens.\n"
        "Use • bullet symbol only.\n\n"
        "Format:\n\n"
        "<b>Image Analysis</b>\n\n"
        "Clear description of what is visible.\n\n"
        "<b>Extracted Content</b>\n\n"
        "Visible text or code.\n\n"
        "<b>Technical Interpretation</b>\n\n"
        "Explanation of meaning.\n\n"
        "If error detected:\n\n"
        "<b>Issue Detected</b>\n\n"
        "Explanation.\n\n"
        "<b>Suggested Fix</b>\n\n"
        "<pre><code>\n"
        "corrected code if applicable\n"
        "</code></pre>\n\n"
        "<b>Confidence</b>\n\n"
        "High / Medium / Low based on clarity."
    )

    try:
        response = await vision_client.chat.completions.create(
            model=vision_model,
            temperature=0.0,
            max_tokens=1500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                        },
                    ],
                },
            ],
        )
        return response.choices[0].message.content or "<b>Image Analysis</b>\n\nNo visible content detected."
    except Exception as e:
        logging.exception("Vision analysis failed")
        return f"<b>Image Analysis</b>\n\nFailed to process image: {html.escape(str(e))}"


@bot.message_handler(commands=["start", "help"])
def handle_start(message):
    asyncio.run(safe_send_message(message.chat.id, START_MESSAGE, message_type="normal"))


@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    user_id = message.from_user.id
    chat_id = message.chat.id

    with locks[user_id]:
        try:
            photo = message.photo[-1]
            file_info = bot.get_file(photo.file_id)
            image_bytes = bot.download_file(file_info.file_path)
            result = asyncio.run(handle_smart_image(image_bytes))
            asyncio.run(safe_send_message(chat_id, result, message_type="image"))
        except Exception as exc:
            logging.exception("photo handler failed")
            asyncio.run(
                safe_send_message(
                    chat_id,
                    f"<b>Image Analysis</b>\n\nError: {html.escape(str(exc)[:160])}",
                    message_type="image",
                )
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
                asyncio.run(safe_send_message(chat_id, "Please upload an image file for image analysis.", message_type="normal"))
                return

            file_info = bot.get_file(doc.file_id)
            image_bytes = bot.download_file(file_info.file_path)
            result = asyncio.run(handle_smart_image(image_bytes))
            asyncio.run(safe_send_message(chat_id, result, message_type="image"))
        except Exception as exc:
            logging.exception("document handler failed")
            asyncio.run(
                safe_send_message(
                    chat_id,
                    f"<b>Image Analysis</b>\n\nError: {html.escape(str(exc)[:160])}",
                    message_type="image",
                )
            )


@bot.message_handler(content_types=["text"])
def handle_text(message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    query = (message.text or "").strip()

    if not query:
        asyncio.run(safe_send_message(chat_id, "Send a valid text query.", message_type="normal"))
        return

    if query.startswith("/deep"):
        parts = query.split(maxsplit=1)
        deep_query = parts[1].strip() if len(parts) > 1 else ""
        if not deep_query:
            asyncio.run(safe_send_message(chat_id, "Usage: /deep <query>", message_type="normal"))
            return

        with locks[user_id]:
            status = bot.send_message(chat_id, "Deep research in progress...", parse_mode=None)
            try:
                report = run_deep_orchestration(deep_query)
                bot.edit_message_text(
                    "Deep research completed.",
                    chat_id=chat_id,
                    message_id=status.message_id,
                    parse_mode=None,
                )
                asyncio.run(safe_send_message(chat_id, report, message_type="deep"))
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
                    parse_mode=None,
                )
        return

    if is_identity_query(query):
        asyncio.run(safe_send_message(chat_id, IDENTITY_RESPONSE, message_type="normal"))
        return

    with locks[user_id]:
        try:
            debug_mode = is_debug_query(query)
            final_query = build_debug_prompt(query) if debug_mode else query
            response_text = run_orchestration(final_query)
            asyncio.run(safe_send_message(chat_id, response_text, message_type="debug" if debug_mode else "normal"))
        except Exception as exc:
            logging.exception("orchestration failed")
            fallback = (
                "Available evidence indicates a temporary processing issue. "
                "Please retry with a slightly more specific query. "
                f"Error: {str(exc)[:200]}"
            )
            asyncio.run(safe_send_message(chat_id, fallback, message_type="normal"))


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
