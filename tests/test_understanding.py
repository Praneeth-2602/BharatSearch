"""Unit tests for query understanding (rule-based, offline, no services)."""

from __future__ import annotations

from bharatsearch_contracts import Intent, Language
from bharatsearch_retrieval.understanding import (
    _PRODUCT_VOCAB,
    classify_intent,
    correct_typo,
    correct_word,
    detect_language,
    understand,
)

# --- Language detection -----------------------------------------------------


def test_detect_language_english():
    assert detect_language("phone cover") == Language.en
    assert detect_language("water bottle") == Language.en


def test_detect_language_hindi_devanagari():
    assert detect_language("पानी की बोतल") == Language.hi


def test_detect_language_hinglish():
    assert detect_language("sasta phone cover") == Language.hi_latn
    assert detect_language("acchi shoes") == Language.hi_latn


def test_detect_language_mixed():
    assert detect_language("पानी bottle") == Language.mixed


def test_detect_language_other():
    assert detect_language("12345 !@#") == Language.other


# --- Intent classification --------------------------------------------------


def test_classify_intent_product():
    intent, brand = classify_intent("phone cover")
    assert intent == Intent.product
    assert brand is None


def test_classify_intent_brand():
    intent, brand = classify_intent("nike shoes")
    assert intent == Intent.brand
    assert brand == "nike"


def test_classify_intent_attribute():
    intent, brand = classify_intent("red phone cover")
    assert intent == Intent.attribute
    assert brand is None


def test_classify_intent_unknown():
    intent, brand = classify_intent("")
    assert intent == Intent.unknown
    assert brand is None


# --- Typo correction --------------------------------------------------------


def test_correct_typo_fone_to_phone():
    assert correct_typo("fone cover") == "phone cover"


def test_correct_typo_no_change_for_known_words():
    assert correct_typo("phone cover") == "phone cover"


def test_correct_typo_hinglish_preserved():
    assert correct_typo("sasta phone cover") == "sasta phone cover"


def test_correct_typo_devanagari_unchanged():
    assert correct_typo("पानी की बोतल") == "पानी की बोतल"


def test_correct_word_known():
    assert correct_word("phone", _PRODUCT_VOCAB) == "phone"


def test_correct_word_unknown_short():
    assert correct_word("x", _PRODUCT_VOCAB) == "x"


# --- Full understand() ------------------------------------------------------


def test_understand_english_product():
    u = understand("phone cover")
    assert u.corrected_query == "phone cover"
    assert u.language == Language.en
    assert u.intent == Intent.product
    assert u.brand is None
    assert u.source == "rule-based"


def test_understand_hinglish_with_typo():
    u = understand("sasta fone cover")
    assert u.language == Language.hi_latn
    assert u.corrected_query == "sasta phone cover"
    assert u.intent == Intent.product
    assert u.source == "rule-based"


def test_understand_brand():
    u = understand("nike shoes")
    assert u.intent == Intent.brand
    assert u.brand == "nike"
    assert u.language == Language.en


def test_understand_attribute():
    u = understand("red water bottle")
    assert u.intent == Intent.attribute
    assert u.language == Language.en
