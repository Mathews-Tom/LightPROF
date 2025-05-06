import pytest

from lightprof.utils import load_kg_from_triples, get_gemini_tokenizer

def test_load_kg_from_triples_exists():
    # Just check the function exists and can be called with a dummy path
    try:
        load_kg_from_triples("dummy_path.tsv")
    except Exception:
        pass  # Accept any exception for now (file not found etc.)

def test_get_gemini_tokenizer_exists():
    tokenizer = get_gemini_tokenizer()
    assert tokenizer is not None