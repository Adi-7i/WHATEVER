import base64
import datetime as dt
import html
import json
import mimetypes
import os
import re
import threading
import time
from urllib.parse import urlparse
from collections import defaultdict, deque

import requests
import telebot
from dotenv import load_dotenv
from telebot import apihelper

load_dotenv()


def get_env_required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"Environment variable {name} must be an integer.") from e


TELEGRAM_BOT_TOKEN = get_env_required("TELEGRAM_BOT_TOKEN")
AZURE_OPENAI_ENDPOINT = get_env_required("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = get_env_required("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview").strip()
AZURE_OPENAI_DEPLOYMENT = get_env_required("AZURE_OPENAI_DEPLOYMENT")

REQUEST_TIMEOUT_SECONDS = get_env_int("REQUEST_TIMEOUT_SECONDS", 90)
MAX_HISTORY_MESSAGES = get_env_int("MAX_HISTORY_MESSAGES", 16)
MAX_TEXT_CHARS = get_env_int("MAX_TEXT_CHARS", 12000)
MAX_FILE_BYTES = get_env_int("MAX_FILE_BYTES", 350000)

SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "https://searxng.cynerza.in").strip()
SEARXNG_TIMEOUT_SECONDS = get_env_int("SEARXNG_TIMEOUT_SECONDS", 20)
SEARXNG_MAX_RESULTS = get_env_int("SEARXNG_MAX_RESULTS", 8)
SEARXNG_PAGE_FETCH_LIMIT = get_env_int("SEARXNG_PAGE_FETCH_LIMIT", 4)
SEARXNG_PAGE_TIMEOUT_SECONDS = get_env_int("SEARXNG_PAGE_TIMEOUT_SECONDS", 10)

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, parse_mode=None)

SYSTEM_PROMPT = """
You are an advanced AI Telegram assistant.

Core behavior:
1) Reply in the user's preferred language automatically.
   - Support Hindi, English, and Hinglish naturally.
   - If user writes in mixed language, mirror that style.
2) Be strong at coding tasks:
   - Write production-quality code.
   - Explain cleanly when asked.
   - Debug buggy code with step-by-step fixes.
3) For debugging:
   - Identify root cause first.
   - Provide corrected code.
   - Mention test steps.
4) Be concise but complete.
5) If user sends image, describe and analyze accurately.
6) Never expose secrets.
7) Always respond in professional plain text.
8) Do not use markdown formatting symbols like headings/bullets with '#' or '*'.
9) Keep answers tightly relevant to the user's exact request.
10) Do not add optional sections, metadata, or extra explanation unless user asks.
""".strip()

# In-memory state per user
chat_history = defaultdict(lambda: deque(maxlen=MAX_HISTORY_MESSAGES))
last_uploaded_code = {}
locks = defaultdict(threading.Lock)

CODE_FILE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".cc", ".h", ".hpp",
    ".cs", ".go", ".rs", ".php", ".rb", ".swift", ".kt", ".kts", ".m", ".mm", ".scala",
    ".sql", ".sh", ".bash", ".zsh", ".ps1", ".r", ".dart", ".lua", ".json", ".yaml", ".yml",
    ".html", ".css", ".scss", ".xml", ".toml", ".ini", ".md"
}


def build_azure_url() -> str:
    endpoint = AZURE_OPENAI_ENDPOINT.rstrip("/")
    return (
        f"{endpoint}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}"
        f"/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    )


def azure_chat_completion(messages, temperature=0.3, max_tokens=1400):
    url = build_azure_url()
    headers = {
        "api-key": AZURE_OPENAI_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def azure_chat_completion_json(messages, temperature=0.0, max_tokens=220):
    url = build_azure_url()
    headers = {
        "api-key": AZURE_OPENAI_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    resp = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    data = resp.json()
    raw = data["choices"][0]["message"]["content"]
    return json.loads(raw)


def clamp_text(text: str) -> str:
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return text[:MAX_TEXT_CHARS] + "\n\n[Truncated due to length]"


def split_long_message(text: str, chunk_size=3900):
    text = text.strip() or "(empty response)"
    chunks = []
    while text:
        chunks.append(text[:chunk_size])
        text = text[chunk_size:]
    return chunks


def sanitize_markdownish(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        # Remove markdown-like heading/bullet prefixes only.
        line = re.sub(r"^\s*#{1,6}\s*", "", line)
        line = re.sub(r"^\s*\*+\s*", "", line)
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def build_intent_instruction(user_text: str) -> str:
    t = user_text.lower()
    if any(k in t for k in ["email", "mail", "e-mail"]):
        return "Respond as a professional email draft, concise and directly usable."
    if any(k in t for k in ["letter", "application", "cover letter"]):
        return "Respond as a professional letter draft, concise and directly usable."
    if any(k in t for k in ["story", "novel", "katha", "kahani"]):
        return "Respond with a clear narrative matching the user's requested style and length."
    if any(k in t for k in ["code", "debug", "bug", "error", "fix"]):
        return "Respond with direct debugging help and corrected code when needed."
    return "Respond professionally and directly, without extra optional details."


def safe_send_message(chat_id: int, text: str):
    cleaned = sanitize_markdownish(text)
    for part in split_long_message(cleaned):
        bot.send_message(chat_id, part)


def add_to_history(user_id: int, role: str, content: str):
    chat_history[user_id].append({"role": role, "content": clamp_text(content)})


def build_text_messages(user_id: int, user_text: str):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(chat_history[user_id])
    messages.append({"role": "user", "content": clamp_text(user_text)})
    return messages


def infer_debug_request(user_text: str) -> bool:
    lowered = user_text.lower()
    patterns = [
        r"\bdebug\b", r"\bfix\b", r"\berror\b", r"\bbug\b",
        r"\bissue\b", r"\bproblem\b", r"\bcrash\b",
        r"\bसुधार\b", r"\bगलती\b", r"\berror\b", r"\bठीक\b",
    ]
    return any(re.search(p, lowered) for p in patterns)


def should_use_realtime_web(user_text: str) -> dict:
    messages = [
        {
            "role": "system",
            "content": (
                "Classify if a query requires real-time internet search. "
                "Return strict JSON with keys: needs_realtime_search (boolean), reason (string). "
                "Set true for recent/current/live facts like news, prices, weather, sports, "
                "stock/crypto, schedules, or anything likely to have changed."
            ),
        },
        {"role": "user", "content": user_text},
    ]
    try:
        result = azure_chat_completion_json(messages)
        return {
            "needs_realtime_search": bool(result.get("needs_realtime_search", False)),
            "reason": str(result.get("reason", "")).strip()[:180],
        }
    except Exception:
        lowered = user_text.lower()
        realtime_hints = [
            "today", "latest", "breaking", "news", "now", "current", "price", "weather",
            "score", "result", "live", "2026", "2025", "schedule", "update", "recent",
            "stock", "crypto", "bitcoin", "ethereum", "match",
        ]
        return {
            "needs_realtime_search": any(h in lowered for h in realtime_hints),
            "reason": "fallback keyword routing",
        }


def analyze_user_intent(user_text: str) -> dict:
    messages = [
        {
            "role": "system",
            "content": (
                "Analyze the user query for routing and answer style. "
                "Return strict JSON with keys: "
                "needs_realtime_search (boolean), "
                "optimized_search_query (string), "
                "response_length (short|normal|detailed), "
                "answer_directive (string), "
                "quality_level (basic|standard|high), "
                "expected_item_count (integer), "
                "structured_output (boolean). "
                "Set response_length=short for direct/simple asks, detailed only when user explicitly asks details."
            ),
        },
        {"role": "user", "content": user_text},
    ]
    try:
        parsed = azure_chat_completion_json(messages)
        response_length = str(parsed.get("response_length", "normal")).strip().lower()
        if response_length not in {"short", "normal", "detailed"}:
            response_length = "normal"
        quality_level = str(parsed.get("quality_level", "standard")).strip().lower()
        if quality_level not in {"basic", "standard", "high"}:
            quality_level = "standard"
        expected_item_count = parsed.get("expected_item_count", 5)
        try:
            expected_item_count = int(expected_item_count)
        except Exception:
            expected_item_count = 5
        expected_item_count = max(2, min(expected_item_count, 12))
        optimized = clean_text(parsed.get("optimized_search_query", user_text), 220) or user_text
        directive = clean_text(parsed.get("answer_directive", ""), 220)
        structured_output = bool(parsed.get("structured_output", False))
        return {
            "needs_realtime_search": bool(parsed.get("needs_realtime_search", False)),
            "optimized_search_query": optimized,
            "response_length": response_length,
            "answer_directive": directive,
            "quality_level": quality_level,
            "expected_item_count": expected_item_count,
            "structured_output": structured_output,
        }
    except Exception:
        route = should_use_realtime_web(user_text)
        lowered = user_text.lower()
        response_length = "normal"
        if any(k in lowered for k in ["short", "brief", "one line", "in short", "summarize"]):
            response_length = "short"
        if any(k in lowered for k in ["detailed", "deep", "step by step", "explain fully"]):
            response_length = "detailed"
        quality_level = "standard"
        if any(k in lowered for k in ["upsc", "exam", "professional", "analysis", "perspective"]):
            quality_level = "high"
        expected_item_count = 5
        if response_length == "short":
            expected_item_count = 3
        elif response_length == "detailed":
            expected_item_count = 8
        structured_output = any(
            k in lowered for k in ["upsc", "important", "detailed", "professional", "long", "analysis"]
        )
        return {
            "needs_realtime_search": route["needs_realtime_search"],
            "optimized_search_query": user_text,
            "response_length": response_length,
            "answer_directive": "Answer only what the user asked.",
            "quality_level": quality_level,
            "expected_item_count": expected_item_count,
            "structured_output": structured_output,
        }


def resolve_max_tokens(response_length: str) -> int:
    if response_length == "short":
        return 300
    if response_length == "detailed":
        return 1400
    return 850


def build_output_style_instruction(intent: dict, for_realtime: bool) -> str:
    base = (
        "Answer only with data relevant to the user query. "
        "Do not add unrelated details."
    )

    quality = intent.get("quality_level", "standard")
    expected_items = intent.get("expected_item_count", 5)
    structured = bool(intent.get("structured_output", False))
    length = intent.get("response_length", "normal")
    directive = intent.get("answer_directive", "").strip() or "Answer only what the user asked."

    if structured:
        return (
            f"{base}\n"
            f"Quality level: {quality}. Target points: {expected_items}. Length: {length}.\n"
            "Use professional plain-text structure (no markdown symbols):\n"
            "Main Heading\n"
            "Subheading: Core Updates\n"
            "1. ...\n"
            "2. ...\n"
            "Subheading: UPSC Relevance\n"
            "1. Prelims focus ...\n"
            "2. Mains focus ...\n"
            "Subheading: Important Notes\n"
            "1. ...\n"
            "2. ...\n"
            "Keep points factual, concise, and exam-oriented.\n"
            f"Directive: {directive}"
        )

    return (
        f"{base}\n"
        f"Quality level: {quality}. Target points: {expected_items}. Length: {length}.\n"
        "Give direct final answer only.\n"
        f"Directive: {directive}"
    )


def clean_text(value: str, max_len: int) -> str:
    v = re.sub(r"\s+", " ", (value or "")).strip()
    return v[:max_len]


def build_search_variants(query: str):
    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
    lowered = query.lower()
    variants = [query]
    if any(k in lowered for k in ["today", "current", "latest", "recent"]):
        variants.append(f"{query} {today}")
        variants.append(f"{query} daily update")
    return variants[:3]


def search_searxng(query: str, time_range=None):
    url = SEARXNG_BASE_URL.rstrip("/") + "/search"
    params = {
        "q": query,
        "format": "json",
        "language": "auto",
    }
    if time_range:
        params["time_range"] = time_range

    resp = requests.get(url, params=params, timeout=SEARXNG_TIMEOUT_SECONDS)
    resp.raise_for_status()
    data = resp.json()
    raw_results = data.get("results", []) or []

    cleaned = []
    seen_urls = set()
    for item in raw_results:
        link = (item.get("url") or "").strip()
        if not link or link in seen_urls:
            continue
        seen_urls.add(link)

        cleaned.append(
            {
                "title": clean_text(item.get("title", ""), 180),
                "url": link,
                "content": clean_text(item.get("content", ""), 420),
                "engine": clean_text(item.get("engine", ""), 40),
                "score": item.get("score"),
                "published_date": clean_text(item.get("publishedDate", ""), 40),
            }
        )
        if len(cleaned) >= SEARXNG_MAX_RESULTS:
            break
    return {
        "results": cleaned,
        "answers": [clean_text(a, 280) for a in (data.get("answers", []) or []) if a],
        "infoboxes": data.get("infoboxes", []) or [],
    }


def aggregate_search_data(user_text: str):
    variants = build_search_variants(user_text)
    all_results = []
    answers = []
    seen_urls = set()

    for idx, q in enumerate(variants):
        # First query keeps default searx behavior, additional queries bias toward recency.
        payload = search_searxng(q, time_range="day" if idx > 0 else None)
        answers.extend(payload.get("answers", []))
        for item in payload.get("results", []):
            link = item.get("url", "").strip()
            if not link or link in seen_urls:
                continue
            seen_urls.add(link)
            all_results.append(item)

    return {
        "queries_used": variants,
        "answers": answers[:4],
        "results": all_results[:SEARXNG_MAX_RESULTS],
    }


def extract_page_text(raw_html: str, max_len=2200) -> str:
    no_script = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", raw_html)
    text = re.sub(r"(?is)<[^>]+>", " ", no_script)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


def fetch_page_extract(url: str):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; realtime-bot/1.0)"}
    resp = requests.get(url, timeout=SEARXNG_PAGE_TIMEOUT_SECONDS, headers=headers)
    resp.raise_for_status()
    content_type = (resp.headers.get("Content-Type") or "").lower()
    if "html" not in content_type:
        return ""
    return extract_page_text(resp.text)


def fetch_top_page_evidence(search_results):
    page_evidence = []
    for item in search_results[:SEARXNG_PAGE_FETCH_LIMIT]:
        url = item["url"]
        try:
            parsed = urlparse(url)
            # Avoid binary/media links and very noisy non-http sources.
            if parsed.scheme not in {"http", "https"}:
                continue
            page_text = fetch_page_extract(url)
            if page_text:
                page_evidence.append(
                    {
                        "url": url,
                        "title": item.get("title", ""),
                        "extract": clean_text(page_text, 1700),
                    }
                )
        except Exception:
            continue
    return page_evidence


def build_realtime_prompt(user_text: str, search_bundle, page_evidence, intent: dict) -> str:
    lines = []
    search_results = search_bundle.get("results", [])
    for idx, r in enumerate(search_results, start=1):
        lines.append(
            f"{idx}. Title: {r['title'] or 'N/A'}\n"
            f"   URL: {r['url']}\n"
            f"   Snippet: {r['content'] or 'N/A'}\n"
            f"   Engine: {r['engine'] or 'N/A'}\n"
            f"   Published: {r['published_date'] or 'N/A'}"
        )
    evidence = "\n".join(lines) if lines else "No usable web evidence found."
    answers = search_bundle.get("answers", [])
    answers_block = "\n".join(f"- {a}" for a in answers) if answers else "None"
    pages_block = (
        "\n\n".join(
            f"Page {i+1}:\nTitle: {p['title'] or 'N/A'}\nURL: {p['url']}\nExtract: {p['extract']}"
            for i, p in enumerate(page_evidence)
        )
        if page_evidence
        else "No page extracts available."
    )

    return (
        "User query:\n"
        f"{user_text}\n\n"
        "SearXNG queries used:\n"
        + "\n".join(f"- {q}" for q in search_bundle.get("queries_used", []))
        + "\n\n"
        "SearXNG direct answers:\n"
        f"{answers_block}\n\n"
        "Web evidence from SearXNG (JSON cleaned):\n"
        f"{evidence}\n\n"
        "Fetched page extracts from top links:\n"
        f"{pages_block}\n\n"
        "Task:\n"
        "1) Verify and clean facts before answering.\n"
        "2) Prefer facts from page extracts over generic snippets.\n"
        "3) Do not invent facts not present in evidence.\n"
        "4) If evidence is weak/conflicting, say that briefly.\n"
        f"5) {build_output_style_instruction(intent, for_realtime=True)}\n"
    )


def download_telegram_file(file_id: str) -> bytes:
    file_info = bot.get_file(file_id)
    return bot.download_file(file_info.file_path)


def detect_mime_from_name(filename: str) -> str:
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "application/octet-stream"


def is_image_document(filename: str, mime_type: str) -> bool:
    return mime_type.startswith("image/") or filename.lower().endswith(
        (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
    )


def is_likely_code_file(filename: str) -> bool:
    lower = filename.lower()
    return any(lower.endswith(ext) for ext in CODE_FILE_EXTENSIONS)


def ask_llm_and_reply(chat_id: int, user_id: int, user_text: str):
    with locks[user_id]:
        try:
            intent = analyze_user_intent(user_text)
            search_bundle = {"queries_used": [], "answers": [], "results": []}
            page_evidence = []
            realtime_meta = ""
            final_user_text = user_text
            max_tokens = resolve_max_tokens(intent["response_length"])

            if intent["needs_realtime_search"]:
                try:
                    search_bundle = aggregate_search_data(intent["optimized_search_query"])
                    page_evidence = fetch_top_page_evidence(search_bundle.get("results", []))
                    realtime_meta = (
                        "[Routing] Realtime web search used via SearXNG. "
                        f"Query: {intent['optimized_search_query']}. "
                        f"Results: {len(search_bundle.get('results', []))}, "
                        f"Page extracts: {len(page_evidence)}."
                    )
                    final_user_text = build_realtime_prompt(
                        user_text,
                        search_bundle,
                        page_evidence,
                        intent,
                    )
                except Exception as se:
                    realtime_meta = (
                        "[Routing] Realtime requested but SearXNG failed; "
                        f"proceeding without web data. Error: {str(se)[:200]}"
                    )
                    intent_rule = build_intent_instruction(user_text)
                    final_user_text = (
                        f"{user_text}\n\nFormatting rule: {intent_rule}\n{realtime_meta}"
                    )
            else:
                intent_rule = build_intent_instruction(user_text)
                final_user_text = (
                    f"{user_text}\n\n"
                    f"Formatting rule: {intent_rule}\n"
                    f"{build_output_style_instruction(intent, for_realtime=False)}"
                )

            messages = build_text_messages(user_id, final_user_text)
            response = azure_chat_completion(messages, max_tokens=max_tokens)

            history_user = final_user_text
            if realtime_meta:
                history_user = f"{final_user_text}\n\n{realtime_meta}"
            if search_bundle.get("results"):
                history_user += f"\n[Web results count: {len(search_bundle.get('results', []))}]"
            if page_evidence:
                history_user += f"\n[Page extracts count: {len(page_evidence)}]"

            add_to_history(user_id, "user", history_user)
            add_to_history(user_id, "assistant", response)

            safe_send_message(chat_id, response)
        except requests.HTTPError as e:
            detail = ""
            try:
                detail = e.response.text[:500]
            except Exception:
                pass
            safe_send_message(chat_id, f"Azure OpenAI HTTP error:\n{detail or str(e)}")
        except Exception as e:
            safe_send_message(chat_id, f"Request failed:\n{str(e)}")


@bot.message_handler(commands=["start", "help"])
def handle_start(message):
    help_text = (
        "Advanced AI Telegram Bot is ready.\n\n"
        "Features:\n"
        "- Hindi / English / Hinglish conversation\n"
        "- Image understanding\n"
        "- Code generation\n"
        "- Debug uploaded code files\n\n"
        "Commands:\n"
        "- /reset : clear chat memory\n"
        "- /debug : debug last uploaded code file"
    )
    bot.reply_to(message, help_text)


@bot.message_handler(commands=["reset"])
def handle_reset(message):
    user_id = message.from_user.id
    chat_history[user_id].clear()
    last_uploaded_code.pop(user_id, None)
    bot.reply_to(message, "Memory cleared for this chat.")


@bot.message_handler(commands=["debug"])
def handle_debug_command(message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    code_blob = last_uploaded_code.get(user_id)
    if not code_blob:
        bot.reply_to(
            message,
            "No uploaded code found yet. Upload a code file first, then use /debug.",
        )
        return

    prompt = (
        "Debug the following code.\n"
        "1) Find bugs/root causes\n"
        "2) Return corrected code\n"
        "3) Explain fixes briefly\n"
        "4) Add test steps\n\n"
        f"Filename: {code_blob['filename']}\n\n"
        f"```\n{code_blob['content']}\n```"
    )
    ask_llm_and_reply(chat_id, user_id, prompt)


@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    user_id = message.from_user.id
    chat_id = message.chat.id

    try:
        photo = message.photo[-1]
        img_bytes = download_telegram_file(photo.file_id)
        b64 = base64.b64encode(img_bytes).decode("utf-8")

        caption = message.caption or ""
        user_prompt = caption.strip() or "Analyze this image in detail."

        vision_user_content = [
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            },
        ]

        with locks[user_id]:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(chat_history[user_id])
            messages.append({"role": "user", "content": vision_user_content})

            response = azure_chat_completion(messages)
            add_to_history(user_id, "user", f"[Image] {user_prompt}")
            add_to_history(user_id, "assistant", response)

        safe_send_message(chat_id, response)

    except Exception as e:
        safe_send_message(chat_id, f"Image processing failed:\n{str(e)}")


@bot.message_handler(content_types=["document"])
def handle_document(message):
    user_id = message.from_user.id
    chat_id = message.chat.id

    doc = message.document
    filename = doc.file_name or "uploaded_file"
    mime_type = doc.mime_type or detect_mime_from_name(filename)

    if doc.file_size and doc.file_size > MAX_FILE_BYTES:
        bot.send_message(
            chat_id,
            f"File too large ({doc.file_size} bytes). Keep under {MAX_FILE_BYTES} bytes.",
        )
        return

    try:
        raw = download_telegram_file(doc.file_id)

        if is_image_document(filename, mime_type):
            b64 = base64.b64encode(raw).decode("utf-8")
            prompt = message.caption or "Analyze this uploaded image."
            user_content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                },
            ]

            with locks[user_id]:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                messages.extend(chat_history[user_id])
                messages.append({"role": "user", "content": user_content})

                response = azure_chat_completion(messages)
                add_to_history(user_id, "user", f"[Image upload] {prompt}")
                add_to_history(user_id, "assistant", response)

            safe_send_message(chat_id, response)
            return

        # Treat as text/code document
        text = raw.decode("utf-8", errors="replace")
        text = clamp_text(text)

        if is_likely_code_file(filename):
            last_uploaded_code[user_id] = {"filename": filename, "content": text}

            auto_prompt = (
                "User uploaded code. Analyze quickly and provide:\n"
                "1) What this code does\n"
                "2) Potential bugs/improvements\n"
                "3) Better version if needed\n\n"
                f"Filename: {filename}\n\n"
                f"```\n{text}\n```"
            )
            ask_llm_and_reply(chat_id, user_id, auto_prompt)
            return

        generic_prompt = (
            f"User uploaded a file named {filename}."
            " Summarize its content and help based on the caption/instruction.\n\n"
            f"Caption: {message.caption or 'N/A'}\n\n"
            f"Content:\n{text}"
        )
        ask_llm_and_reply(chat_id, user_id, generic_prompt)

    except Exception as e:
        bot.send_message(chat_id, f"Document handling failed: `{str(e)}`")


@bot.message_handler(content_types=["text"])
def handle_text(message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    user_text = (message.text or "").strip()
    if not user_text:
        bot.reply_to(message, "Send a text query.")
        return

    # If user asks to debug and a code file exists, enrich prompt with uploaded code.
    if infer_debug_request(user_text) and user_id in last_uploaded_code:
        code_blob = last_uploaded_code[user_id]
        merged_prompt = (
            f"{user_text}\n\n"
            "Use this uploaded code for debugging:\n"
            f"Filename: {code_blob['filename']}\n"
            f"```\n{code_blob['content']}\n```"
        )
        ask_llm_and_reply(chat_id, user_id, merged_prompt)
        return

    ask_llm_and_reply(chat_id, user_id, user_text)


def validate_config():
    required = [
        "TELEGRAM_BOT_TOKEN",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
    ]
    missing = [name for name in required if not os.getenv(name, "").strip()]
    if missing:
        raise ValueError("Please set required environment variables: " + ", ".join(missing))


if __name__ == "__main__":
    validate_config()
    # Enable retries in the underlying Telegram HTTP client.
    apihelper.RETRY_ON_ERROR = True
    apihelper.RETRY_TIMEOUT = 2

    print("Bot started...")
    backoff = 2
    while True:
        try:
            bot.infinity_polling(
                timeout=20,
                long_polling_timeout=20,
                skip_pending=True,
                allowed_updates=["message"],
            )
        except Exception as e:
            print(f"Polling crashed: {e}. Restarting in {backoff}s...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
