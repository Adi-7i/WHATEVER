# Advanced Azure OpenAI Telegram Bot

This bot uses:
- `telebot` (pyTelegramBotAPI) for Telegram
- `requests` for direct Azure OpenAI API calls
- `python-dotenv` to load secrets from `.env`
- SearXNG (`format=json`) for real-time internet search
- environment variables from `.env`

## Setup
1. Create virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy env template and set real values:
   ```bash
   cp .env.example .env
   ```
4. Run:
   ```bash
   python3 bot.py
   ```

## Features
- Multilingual chat (Hindi / English / Hinglish)
- Image understanding (photo or image document)
- Code generation and explanation
- Code debug for uploaded code files (`/debug` or text debug request)
- Per-user short conversation memory
- Realtime intent routing:
  - Azure GPT first decides if realtime internet data is needed
  - If needed, bot fetches JSON data from SearXNG
  - Azure GPT verifies/cleans evidence and returns structured output

## Commands
- `/start` or `/help`
- `/reset`
- `/debug`
# WHATEVER
