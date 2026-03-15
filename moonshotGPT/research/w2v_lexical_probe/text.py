"""Text normalization and word tokenization for lexical probes."""

from __future__ import annotations

import re
import string
import unicodedata
from dataclasses import asdict, dataclass

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"<url>|<email>|[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
_PUNCT_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "—": "-",
        "–": "-",
        "\u00a0": " ",
    }
)


@dataclass(frozen=True)
class TextNormalizationConfig:
    lowercase: bool = True
    strip_html: bool = True
    replace_urls: bool = True
    replace_emails: bool = True
    collapse_whitespace: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class WordTokenizer:
    """Normalize detokenized text and extract readable word-level tokens."""

    def __init__(
        self,
        config: TextNormalizationConfig | None = None,
        *,
        drop_tokens: set[str] | None = None,
    ):
        self.config = config or TextNormalizationConfig()
        self.drop_tokens = {str(token).casefold() for token in (drop_tokens or set())}

    def normalize(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text).translate(_PUNCT_TRANSLATION)
        if self.config.strip_html:
            normalized = _HTML_TAG_RE.sub(" ", normalized)
        if self.config.replace_urls:
            normalized = _URL_RE.sub(" <url> ", normalized)
        if self.config.replace_emails:
            normalized = _EMAIL_RE.sub(" <email> ", normalized)
        if self.config.lowercase:
            normalized = normalized.lower()
        if self.config.collapse_whitespace:
            normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
        return normalized

    def tokenize(self, text: str) -> list[str]:
        normalized = self.normalize(text)
        tokens = _WORD_RE.findall(normalized)
        if not self.drop_tokens:
            return tokens
        return [token for token in tokens if token.casefold() not in self.drop_tokens]


class PaperEvalWordTokenizer:
    """Mimic the paper notebook's simple evaluation preprocessing."""

    def __init__(self, *, drop_tokens: set[str] | None = None):
        self.drop_tokens = {str(token).casefold() for token in (drop_tokens or set())}
        self._punct_translation = str.maketrans("", "", string.punctuation)

    def normalize(self, text: str) -> str:
        return str(text).lower().strip().translate(self._punct_translation)

    def tokenize(self, text: str) -> list[str]:
        tokens = self.normalize(text).split()
        if not self.drop_tokens:
            return tokens
        return [token for token in tokens if token.casefold() not in self.drop_tokens]


__all__ = ["PaperEvalWordTokenizer", "TextNormalizationConfig", "WordTokenizer"]
