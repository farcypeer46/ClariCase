import numpy as np

from src.improved_model_1.features import build_vectorizer

# Reused corpus — big enough that every retained term has >= min_df occurrences
# in both word and char vectorizers.
_CORPUS = [
    "I disputed a debt collector call", "call to collect debt call call",
    "debt debt debt collector collector collector",
    "collector call call call call debt",
    "I disputed disputed disputed a collector",
    "call collector debt collector debt collector",
] * 2


def test_word_only_vectorizer_fits_and_transforms():
    vec = build_vectorizer(use_char=False, word_max=1000)
    X = vec.fit_transform(_CORPUS)
    assert X.shape[0] == len(_CORPUS)
    assert X.dtype == np.float32


def test_hybrid_word_char_returns_feature_union():
    vec = build_vectorizer(use_char=True, word_max=200, char_max=200)
    X = vec.fit_transform(_CORPUS)
    assert X.shape[0] == len(_CORPUS)
