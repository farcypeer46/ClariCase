from src.utils.text import clean_text, tokenize


def test_reexports_advanced_clean_text():
    from data.preparation.advanced_text import clean_text as canonical
    assert clean_text is canonical


def test_tokenize_lowercases_and_strips_numbers():
    tokens = tokenize("Call 1234567 don't ignore me!!")
    assert "don't" not in tokens  # advanced clean_text converts n't -> " not"
    assert "not" in tokens
    assert "call" in tokens
    assert "1234567" not in tokens
    assert "number" in tokens
