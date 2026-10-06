"""TF-IDF vectorizers using the canonical clean_text as preprocessor."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from src.utils.text import clean_text


def build_vectorizer(use_char: bool = True,
                     word_max: int = 150_000,
                     char_max: int = 150_000):
    word = TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 2), min_df=3, max_df=0.95,
        max_features=word_max, sublinear_tf=True,
        strip_accents="unicode", dtype=np.float32,
    )
    if not use_char:
        return word
    char = TfidfVectorizer(
        preprocessor=clean_text,
        analyzer="char_wb", ngram_range=(3, 5), min_df=5,
        max_features=char_max, sublinear_tf=True, dtype=np.float32,
    )
    return FeatureUnion([("word", word), ("char", char)])
