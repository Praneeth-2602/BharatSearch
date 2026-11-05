"""Unit tests for noise functions (data/make_variants.py)."""

import random

from data.make_variants import (
    NOISE_FUNCTIONS,
    apply_noise,
    chat_abbreviations,
    drop_vowels,
    keyboard_adjacent,
    phonetic_sub,
    repeat_letters,
    space_merge_split,
)


def test_noise_never_empties():
    rng = random.Random(42)
    text = "smartphone cover"
    for fn in NOISE_FUNCTIONS:
        result = fn(text, rng)
        assert result != "", f"{fn.__name__} emptied non-empty text"


def test_drop_vowels():
    rng = random.Random(0)
    # With prob 1.0 every vowel is dropped.
    assert drop_vowels("phone", rng, prob=1.0) == "phn"


def test_repeat_letters():
    rng = random.Random(0)
    # prob=1.0 repeats every alpha char: s->ss, h->hh, o->oo, e->ee, s->ss
    assert repeat_letters("shoes", rng, prob=1.0) == "sshhooeess"


def test_phonetic_sub():
    rng = random.Random(0)
    assert phonetic_sub("phone", rng, prob=1.0) == "fone"


def test_keyboard_adjacent():
    rng = random.Random(0)
    out = keyboard_adjacent("water", rng, prob=1.0)
    assert out != "water"


def test_space_merge_split():
    rng = random.Random(0)
    assert space_merge_split("water bottle", rng, prob=1.0) == "waterbottle"


def test_chat_abbreviations():
    rng = random.Random(0)
    assert chat_abbreviations("for you", rng, prob=1.0) == "4 u"


def test_apply_noise_deterministic():
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    text = "shoes water bottle"
    assert apply_noise(text, rng1) == apply_noise(text, rng2)
