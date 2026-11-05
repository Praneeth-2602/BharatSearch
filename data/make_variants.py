"""Query variant generator (Implementation Guide, Section 2.2).

Generates Hinglish / vernacular / typo variants from English queries while
preserving their relevance judgments. All noise functions take a seeded
`random.Random` so output is deterministic for a given seed.

Noise functions (each applied with a configurable probability):
  - vowel dropping        smartphone cover -> smrtphn cvr
  - letter repetition     shoes -> shooes
  - phonetic substitution phone -> fone    (ph/f, w/v, aa/a, ee/i)
  - keyboard-adjacent       water -> wster
  - space merge/split     water bottle -> waterbottle
  - chat abbreviations    for -> 4, please -> plz
"""

from __future__ import annotations

import random
import re
from typing import Callable

VARIANT_TYPES = [
    "clean_en",
    "typo_en",
    "hinglish_roman",
    "codemixed",
    "devanagari",
    "other_indic",
]

# Variant type shares (Section 2.2 table)
VARIANT_SHARES = {
    "clean_en": 0.20,
    "typo_en": 0.20,
    "hinglish_roman": 0.25,
    "codemixed": 0.20,
    "devanagari": 0.10,
    "other_indic": 0.05,
}

VOWELS = set("aeiouAEIOU")

MIN_WORDS_FOR_MERGE = 2
MIN_WORD_LEN_FOR_SPLIT = 4

# QWERTY adjacency (compact, lower-case)
KEYBOARD_ADJACENT = {
    "q": "wa",
    "w": "qes",
    "e": "wrd",
    "r": "etf",
    "t": "ryg",
    "y": "tuh",
    "u": "yij",
    "i": "uok",
    "o": "ipl",
    "p": "o",
    "a": "qszx",
    "s": "awdexzc",
    "d": "serfcxv",
    "f": "drtgbv",
    "g": "ftyhcn",
    "h": "gyujnm",
    "j": "huikm",
    "k": "jiolm",
    "l": "kop",
    "z": "asx",
    "x": "zsdc",
    "c": "xdfv",
    "v": "cfgb",
    "b": "vghn",
    "n": "bhjm",
    "m": "njk",
}

CHAT_ABBREVS = {
    "for": "4",
    "please": "plz",
    "you": "u",
    "are": "r",
    "to": "2",
    "too": "2",
    "be": "b",
    "before": "b4",
}

# Longer phonetic patterns first so they take priority.
PHONETIC_SUBS = [
    ("ph", "f"),
    ("f", "ph"),
    ("w", "v"),
    ("v", "w"),
    ("aa", "a"),
    ("a", "aa"),
    ("ee", "i"),
    ("i", "ee"),
]


def drop_vowels(text: str, rng: random.Random, prob: float = 0.3) -> str:
    return "".join(ch for ch in text if ch not in VOWELS or rng.random() >= prob)


def repeat_letters(text: str, rng: random.Random, prob: float = 0.2) -> str:
    out: list[str] = []
    for ch in text:
        out.append(ch)
        if ch.isalpha() and rng.random() < prob:
            out.append(ch)
    return "".join(out)


def _sub_all(text: str, subs: list[tuple[str, str]], rng: random.Random, prob: float) -> str:
    """Apply each substitution pattern at most once per text region.

    Patterns are tried in length-descending order so longer matches win, and a
    substituted region is never re-processed by a later pattern — rules like
    ("ph","f") and ("f","ph") therefore cannot cancel each other out.
    """
    ordered = sorted(subs, key=lambda pair: -len(pair[0]))
    pattern = re.compile("|".join(re.escape(p) for p, _ in ordered))
    repl = dict(ordered)
    out: list[str] = []
    pos = 0
    for m in pattern.finditer(text):
        out.append(text[pos : m.start()])
        if rng.random() < prob:
            out.append(repl[m.group(0)])
        else:
            out.append(m.group(0))
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)


def phonetic_sub(text: str, rng: random.Random, prob: float = 0.2) -> str:
    return _sub_all(text, PHONETIC_SUBS, rng, prob)


def keyboard_adjacent(text: str, rng: random.Random, prob: float = 0.15) -> str:
    out: list[str] = []
    for ch in text:
        lower = ch.lower()
        if lower in KEYBOARD_ADJACENT and rng.random() < prob:
            repl = rng.choice(KEYBOARD_ADJACENT[lower])
            out.append(repl.upper() if ch.isupper() else repl)
        else:
            out.append(ch)
    return "".join(out)


def space_merge_split(text: str, rng: random.Random, prob: float = 0.1) -> str:
    words = text.split()
    if len(words) < MIN_WORDS_FOR_MERGE:
        return text
    if rng.random() < prob:
        i = rng.randrange(len(words) - 1)
        words[i : i + 2] = ["".join(words[i : i + 2])]
    elif rng.random() < prob:
        i = rng.randrange(len(words))
        w = words[i]
        if len(w) >= MIN_WORD_LEN_FOR_SPLIT:
            j = rng.randrange(1, len(w))
            words[i : i + 1] = [w[:j], w[j:]]
    return " ".join(words)


def chat_abbreviations(text: str, rng: random.Random, prob: float = 0.1) -> str:
    words = text.split()
    out: list[str] = []
    for w in words:
        key = w.lower().strip(".,!?")
        if key in CHAT_ABBREVS and rng.random() < prob:
            out.append(CHAT_ABBREVS[key] + w[len(key) :])
        else:
            out.append(w)
    return " ".join(out)


NOISE_FUNCTIONS: list[Callable[[str, random.Random], str]] = [
    drop_vowels,
    repeat_letters,
    phonetic_sub,
    keyboard_adjacent,
    space_merge_split,
    chat_abbreviations,
]


def apply_noise(text: str, rng: random.Random, prob: float = 0.3) -> str:
    """Apply each noise function once with probability `prob`."""
    for fn in NOISE_FUNCTIONS:
        text = fn(text, rng, prob)
    return text


def make_variants(queries: list[str], seed: int = 42) -> dict[str, list[str]]:
    """Generate one variant of each type per query.

    LLM-rewrite types (hinglish_roman, codemixed, devanagari, other_indic)
    are scaffolded as identity here; wire an LLM client in per Section 2.2.
    """
    rng = random.Random(seed)
    variants: dict[str, list[str]] = {vt: [] for vt in VARIANT_TYPES}
    for q in queries:
        variants["clean_en"].append(q)
        variants["typo_en"].append(apply_noise(q, rng))
        variants["hinglish_roman"].append(q)  # TODO: LLM romanized-Hindi rewrite
        variants["codemixed"].append(q)  # TODO: LLM code-mixed rewrite
        variants["devanagari"].append(q)  # TODO: LLM Devanagari translation
        variants["other_indic"].append(q)  # TODO: LLM other Indic translation
    return variants
