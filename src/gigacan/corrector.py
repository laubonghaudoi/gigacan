import re
from typing import List, Tuple, Optional


class RuleBasedCorrector:
    """Apply ordered regex replacements to subtitle text.

    Usage:
    - Instantiate with default rules or provide your own list of (pattern, replacement).
    - Call correct_text for a single cue's text, or correct_entries for a list of VTT entries.
    """

    def __init__(self, rules: Optional[List[Tuple[re.Pattern, str]]] = None) -> None:
        if rules is None:
            # Default Cantonese-focused typo/transcription fixes provided by the user
            rules = [
                (re.compile(r"俾(?!(?:路支|斯麥|益))"), r"畀"),
                (re.compile(r"(?<!(?:聯關))[系繫](?!(?:統))"), r"係"),
                (re.compile(r"噶"), r"㗎"),
                (re.compile(r"姥爺"), r"老爺"),
                (re.compile(r"咁(?=[我你佢就樣就話係啊呀嘅，。])"), r"噉"),
                (re.compile(r"(?<![曝晾])曬(?:[衣太衫褲被命嘢相])"), r"晒"),
                (re.compile(r"(?<=[好])翻(?=[去到嚟])"), r"返"),
                (re.compile(r"嚇([，。！？])"), r"吓\g<1>"),
                (re.compile(r"(唔該|多謝|返|翻)曬"), r"\g<1>晒"),
                (re.compile(r"<\|\w+\|>"), r""),
            ]
        self.regular_errors: List[Tuple[re.Pattern, str]] = rules

    def correct_text(self, text: str) -> str:
        """Return corrected text after applying all rules in order."""
        out = text or ""
        for pat, repl in self.regular_errors:
            out = pat.sub(repl, out)
        return out

    def correct_entries(self, entries: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
        """Return a new entries list with corrected cue text.

        entries: List of (start_seconds, end_seconds, text)
        """
        corrected: List[Tuple[float, float, str]] = []
        for start_s, end_s, text in entries:
            corrected.append((start_s, end_s, self.correct_text(text)))
        return corrected
