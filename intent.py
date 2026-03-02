from enum import Enum
import re


class IntentType(str, Enum):
    REALTIME_NEWS = "REALTIME_NEWS"
    STATIC_KNOWLEDGE = "STATIC_KNOWLEDGE"
    GENERAL_CHAT = "GENERAL_CHAT"


class RuleBasedIntentClassifier:
    """Simple deterministic classifier. No LLM call here by design."""

    REALTIME_PATTERNS = [
        r"\bbreaking\b",
        r"\blatest\b",
        r"\btoday\b",
        r"\bnow\b",
        r"\bcurrent\b",
        r"\bnews\b",
        r"\bheadline\b",
        r"\blive\b",
        r"\bupdate\b",
        r"\bupdates\b",
        r"\bmarket\b",
        r"\bstock\b",
        r"\bcrypto\b",
        r"\bbitcoin\b",
        r"\bethereum\b",
        r"\bscore\b",
        r"\bmatch\b",
        r"\belection\b",
        r"\bweather\b",
    ]

    STATIC_PATTERNS = [
        r"\bwhat is\b",
        r"\bexplain\b",
        r"\bdefinition\b",
        r"\bhistory of\b",
        r"\barchitecture\b",
        r"\balgorithm\b",
        r"\bhow does\b",
        r"\bcompare\b",
        r"\bdifference between\b",
    ]

    def classify(self, user_text: str) -> IntentType:
        text = (user_text or "").strip().lower()
        if not text:
            return IntentType.GENERAL_CHAT

        if self._matches_any(text, self.REALTIME_PATTERNS):
            return IntentType.REALTIME_NEWS
        if self._matches_any(text, self.STATIC_PATTERNS):
            return IntentType.STATIC_KNOWLEDGE
        return IntentType.GENERAL_CHAT

    @staticmethod
    def _matches_any(text: str, patterns: list[str]) -> bool:
        return any(re.search(pattern, text) for pattern in patterns)
