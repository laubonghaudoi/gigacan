from __future__ import annotations

import re
from collections import Counter


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-/]*|[\u3400-\u9fff]{2,}")
META_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")
SPACE_RE = re.compile(r"\s+")
PUNCT_FOR_SIMILARITY_RE = re.compile(r"[，。！？、,.!?：:；;「」『』（）()\[\]{}'\"`~·…—-]")

COMMON_TERMS = {
    "主席",
    "委員",
    "會議",
    "政府",
    "議員",
    "香港",
    "文件",
    "立法會",
    "今日",
    "我們",
    "你哋",
}


def _build_converter() -> object | None:
    try:
        import opencc

        return opencc.OpenCC("s2hk")
    except Exception:
        return None


class TextNormalizer:
    """Normalize subtitle text for alignment and constrained edits."""

    def __init__(self) -> None:
        self._converter = _build_converter()
        self._regular_errors: list[tuple[re.Pattern[str], str]] = [
            (re.compile(r"俾(?!(?:路支|斯麥|益))"), "畀"),
            (re.compile(r"(?<!(?:聯))[系繫](?!(?:統))"), "係"),
            (re.compile(r"噶"), "㗎"),
            (re.compile(r"咁(?=[我你佢就樣就話係啊呀嘅，。])"), "噉"),
            (re.compile(r"(?<![曝晾])曬(?:[衣太衫褲被命嘢相])"), "晒"),
            (re.compile(r"(?<=[好])翻(?=[去到嚟])"), "返"),
        ]

    def clean(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = META_TOKEN_RE.sub("", cleaned)
        if self._converter is not None:
            try:
                cleaned = self._converter.convert(cleaned)
            except Exception:
                pass
        for pattern, replacement in self._regular_errors:
            cleaned = pattern.sub(replacement, cleaned)
        return SPACE_RE.sub(" ", cleaned).strip()

    def for_similarity(self, text: str) -> str:
        cleaned = self.clean(text)
        cleaned = PUNCT_FOR_SIMILARITY_RE.sub("", cleaned)
        cleaned = SPACE_RE.sub("", cleaned)
        return cleaned.lower()


def tokenize_terms(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def extract_top_terms(texts: list[str], *, top_k: int = 30) -> list[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        for token in tokenize_terms(text):
            if len(token) <= 1:
                continue
            if token in COMMON_TERMS:
                continue
            counter[token] += 1

    if not counter:
        return []

    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ordered[:top_k]]
