# Whatever

**Bot Name:** WilloFire  
**Created by:** lucifer  
**Powered by:** Cynerza Systems Private Limited

---

## Project Overview

Whatever is an advanced AI-powered Telegram bot engineered for production environments. It combines GPT-4.1 intelligence with multi-source verification, real-time web research capabilities, and sophisticated code analysis to deliver reliable, context-aware responses. The bot operates with a modular architecture designed for scalability, performance, and maintainability.

---

## Key Features

- **Smart AI Assistant:** GPT-4.1 powered conversational AI with contextual understanding
- **Deep Research Mode:** `/deep <query>` command for comprehensive multi-source research and verification
- **Multi-source Verification Engine:** Aggregates and validates information from multiple sources
- **Advanced Code Debugging:** Analyzes code snippets and provides structured error analysis
- **Clean Code Responses:** Copy-friendly formatted code output with syntax preservation
- **Image Understanding:** Technical image analysis and visual content interpretation
- **Multilingual Support:** Full support for English, Hindi, Hinglish, and extended global language coverage
- **Asynchronous Architecture:** Non-blocking operations for responsive user interaction
- **Modular Backend Design:** Decoupled components for easy maintenance and extension

---

## Architecture Overview

The bot employs a layered architecture that separates concerns and enables effective scaling:

### Intent Layer

The Intent Layer acts as the entry point for all user messages. It preprocesses input, classifies user intent, and routes requests to appropriate handlers. This layer determines whether a query requires simple direct response, web search, deep research, or specialized handling (code debugging, image analysis).

### Search Layer (SearXNG)

Integrates with SearXNG for real-time internet search capabilities. Retrieves JSON-formatted search results, intelligently handles query optimization, and sources information from multiple search engines simultaneously to ensure coverage and accuracy.

### Scraper Layer (BeautifulSoup + httpx)

Performs efficient web scraping to extract detailed content from identified sources. Uses BeautifulSoup for HTML parsing and httpx for asynchronous HTTP requests. Handles pagination, dynamic content, and gracefully manages network failures.

### Deep Research Pipeline

Orchestrates multi-step research workflows:
1. Initial web search to identify relevant sources
2. Content extraction from high-confidence sources
3. Cross-verification of facts across sources
4. Structured compilation of findings
5. Final GPT-4.1 synthesis with citations

### Summarization Layer

Condenses lengthy content while preserving critical information. Generates concise summaries suitable for messaging platforms with configurable detail levels. Maintains factual accuracy and context through abstractive and extractive techniques.

### Telegram Interface

Manages all bot-to-user communication via the Telegram Bot API. Handles command parsing, state management, and formatted message delivery. Supports inline buttons, code formatting, and multi-message responses for large outputs.

---

## Deep Research Mode Explained

The `/deep` command initiates comprehensive research workflows designed for complex queries requiring multi-source verification.

**Workflow:**
1. User invokes `/deep <query>`
2. System identifies all relevant sources and perspectives
3. Content is scraped and analyzed from multiple sources
4. Information is cross-referenced for factual consistency
5. Contradictions and alternative viewpoints are documented
6. Final synthesis is compiled with source attribution
7. Response is formatted for clarity and actionability

**Use Cases:**
- Technical research requiring current information
- Fact verification and multi-source validation
- Complex problem analysis
- Trend analysis and market research
- Academic research aggregation

---

## Multilingual Capabilities

The bot seamlessly supports multiple languages and code-switching:

- **English:** Full support with technical terminology
- **Hindi:** Native support for Hindi speakers
- **Hinglish:** Code-mixed communication (Hindi + English)
- **Extended Languages:** Framework-ready for additional languages through configuration

Language detection is automatic; users can specify language preference via environment variable or inline commands. Code blocks maintain syntax regardless of language selection.

---

## Code Debugging Engine

The `/debug` command provides structured analysis of code snippets and errors.

**Capabilities:**
- Syntax error identification and correction
- Runtime error analysis with stack trace interpretation
- Logic error detection through code flow analysis
- Performance issue identification
- Security vulnerability scanning
- Best practices enforcement
- Cross-language support (Python, JavaScript, Java, C++, Go, Rust, etc.)

**Input Methods:**
- Text snippets pasted directly
- Code file uploads
- Error traces and logs
- Inline code debugging requests

Output includes root cause analysis, recommended fixes, explanatory notes, and corrected code samples.

---

## Image Understanding Capabilities

The bot processes images to provide technical analysis and understanding:

- **Technical Diagrams:** Architecture and flowchart interpretation
- **Screenshots:** OCR and content extraction
- **Error Screens:** Automated error diagnosis
- **Code Visualization:** Visual code structure analysis
- **Charts and Graphs:** Data extraction and interpretation
- **Visual Troubleshooting:** Problem identification from visual artifacts

Supported formats: JPEG, PNG, WebP, GIF (static).

---

## Tech Stack

- **Language:** Python 3.11+
- **Async Runtime:** asyncio with concurrent task management
- **HTTP Client:** httpx for asynchronous HTTP operations
- **Web Parsing:** BeautifulSoup 4 for HTML/XML parsing
- **AI Model:** GPT-4.1 API with streaming support
- **Telegram Integration:** python-telebot (pyTelegramBotAPI)
- **Web Search:** SearXNG with JSON response format
- **Environment Management:** python-dotenv
- **Concurrency:** asyncio-based task scheduling
- **Deployment:** Container-ready with modular design

---

## Folder Structure

```
whatever/
├── bot.py                    # Main Telegram bot entry point
├── config.py                 # Configuration management and settings
├── intent.py                 # Intent classification and routing
├── search.py                 # SearXNG search integration
├── scraper.py                # Web content extraction (BeautifulSoup + httpx)
├── deep_research.py          # Deep research orchestration pipeline
├── summarizer.py             # Content summarization engine
├── orchestrator.py           # Central request orchestration and state management
├── requirements.txt          # Python dependency specifications
├── .env.example              # Environment variables template
├── README.md                 # This file
└── __pycache__/              # Python compiled bytecode (auto-generated)
```

---

## Installation Guide

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd whatever
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your actual credentials (see section below).

### Step 5: Verify Installation

```bash
python3 -c "import telebot, httpx, bs4; print('All dependencies installed successfully')"
```

---

## Environment Variables

Create a `.env` file in the project root with the following variables:

```
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here

# GPT-4.1 API Configuration
GPT_API_KEY=your_gpt_api_key_here
GPT_MODEL=gpt-4.1
GPT_BASE_URL=https://api.openai.com/v1

# SearXNG Configuration
SEARXNG_INSTANCE_URL=http://localhost:8888
SEARCH_TIMEOUT=10

# Bot Configuration
BOT_NAME=WilloFire
MAX_RESPONSE_LENGTH=4096
DEEP_RESEARCH_MAX_SOURCES=5
REQUEST_TIMEOUT=30

# Logging
LOG_LEVEL=INFO

# Optional: Language Preference
DEFAULT_LANGUAGE=en
```

**Secure Practices:**
- Never commit `.env` files to version control
- Use strong, unique API keys
- Rotate keys periodically
- Restrict key permissions at the API provider level

---

## Running the Bot

### Standard Execution

```bash
python3 bot.py
```

The bot will connect to Telegram and begin polling for messages. Output will display connection status and message activity.

### With Logging

```bash
python3 bot.py --log-level DEBUG
```

### In Background (Production)

```bash
nohup python3 bot.py &
# or with systemd
sudo systemctl start whatever-bot
```

### Docker Execution (if containerized)

```bash
docker build -t whatever-bot .
docker run --env-file .env whatever-bot
```

---

## Usage Examples

### Basic Interaction

Send a message to the bot for instant response:
```
User: "Explain async/await in JavaScript"
Bot: [Provides clear explanation with code examples]
```

### Start Command

```
/start
```

Initiates bot conversation and displays available commands and capabilities.

### Help Command

```
/help
```

Shows detailed usage guide and available features.

### Deep Research Query

```
/deep What are the latest developments in quantum computing?
```

Bot performs comprehensive research, aggregates information from multiple sources, and provides verified insights with citations.

### Code Debugging

```
/debug
[User pastes code with error]
```

Bot analyzes the code and provides structured debugging information.

### Image Analysis

Send an image (screenshot, diagram, or code visualization) to the bot. The bot interprets the image and provides relevant analysis.

### Language Switching

```
/language hindi
```

Switches bot responses to Hindi (if configured).

---

## Performance & Design Philosophy

### Design Principles

1. **Modularity:** Each component operates independently with clear interfaces
2. **Asynchronicity:** Non-blocking operations throughout for responsive interaction
3. **Resilience:** Graceful degradation and error handling
4. **Scalability:** Horizontal scaling support through stateless design
5. **Maintainability:** Clean code structure and comprehensive logging

### Performance Characteristics

- **Response Time:** Sub-2-second response for simple queries
- **Deep Research:** 5-15 seconds depending on source complexity
- **Concurrent Users:** Hundreds of simultaneous users with async architecture
- **Resource Usage:** ~200MB base memory, ~50MB per concurrent research operation
- **Search Optimization:** Parallel source queries with rate limiting

### Optimization Techniques

- Request batching for API calls
- Response streaming for large outputs
- Connection pooling via httpx
- Intelligent caching of research results
- Async task scheduling for background operations

---

## Security Considerations

- **API Key Protection:** Never expose credentials; use environment variables exclusively
- **Input Validation:** All user input is sanitized to prevent injection attacks
- **Rate Limiting:** Implemented per-user and global rate limits to prevent abuse
- **Data Privacy:** User query history is not retained beyond session duration
- **HTTPS:** All external API communications use encrypted HTTPS connections
- **Source Verification:** Web content is validated before use in responses
- **Error Handling:** Sensitive error details are not exposed to users

---

## Future Improvements

- Voice message transcription and response generation
- User preference learning and personalization
- Advanced caching layer for frequently researched topics
- Integration with additional data sources
- Real-time collaborative research features
- Offline fallback responses
- Advanced analytics dashboard

---

## License

Proprietary software. All rights reserved.

---

## Credits

**Development:** lucifer  
**Organization:** Cynerza Systems Private Limited  
**Bot Identity:** WilloFire
