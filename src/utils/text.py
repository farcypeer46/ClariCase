"""Canonical text preprocessing and tokenization for improved models.

`clean_text` is imported (not copied) from `data.preparation.advanced_text`
so pickled TfidfVectorizers reference the stable canonical function.
"""
import re

from data.preparation.advanced_text import clean_text

__all__ = ["clean_text", "tokenize", "TOKEN_RE"]

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(clean_text(text))
