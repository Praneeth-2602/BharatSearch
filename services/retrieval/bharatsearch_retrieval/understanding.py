"""Query understanding: language detection, intent classification, typo
correction.

Lightweight, offline, rule-based understanding with no external model
dependencies (Implementation Guide, Section 4). Replaces the ``fallback``
pass-through in ``/v1/search`` with real understanding.
"""

from __future__ import annotations

import re

from bharatsearch_contracts import Intent, Language, Understanding

# --- Language detection -----------------------------------------------------

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]+")
_LATIN_RE = re.compile(r"[A-Za-z]")

# Common Hindi words written in Latin script (Hinglish). Used to distinguish
# hi-Latn from plain English and to avoid "correcting" Hindi words.
_HINGLISH_WORDS = frozenset(
    {
        "sasta",
        "mehenga",
        "mehanga",
        "achha",
        "acchi",
        "accha",
        "khari",
        "khaali",
        "bhara",
        "naya",
        "purana",
        "bada",
        "chota",
        "sundar",
        "badhiya",
        "theek",
        "nhi",
        "nahi",
        "hai",
        "hain",
        "ka",
        "ki",
        "ke",
        "ko",
        "me",
        "mai",
        "se",
        "par",
        "aur",
        "ya",
        "kya",
        "kaise",
        "kabhi",
        "maal",
        "saman",
        "zyada",
        "kam",
        "thoda",
        "bahut",
        "ek",
        "do",
        "teen",
        "chaar",
        "paanch",
    }
)

# --- Intent classification --------------------------------------------------

_BRANDS = frozenset(
    {
        "nike",
        "samsung",
        "apple",
        "boat",
        "oneplus",
        "xiaomi",
        "realme",
        "vivo",
        "oppo",
        "sony",
        "jbl",
        "puma",
        "adidas",
        "levis",
        "woodland",
        "titan",
        "fastrack",
        "hunter",
        "reebok",
        "dell",
        "hp",
        "lenovo",
        "asus",
        "acer",
        "canon",
        "nikon",
        "noise",
        "boult",
    }
)

_ATTRIBUTE_WORDS = frozenset(
    {
        "red",
        "blue",
        "green",
        "black",
        "white",
        "yellow",
        "pink",
        "orange",
        "large",
        "small",
        "big",
        "huge",
        "tiny",
        "medium",
        "xl",
        "xxl",
        "xs",
        "s",
        "m",
        "l",
        "waterproof",
        "wireless",
        "bluetooth",
        "gaming",
        "foldable",
        "smart",
        "cheap",
        "costly",
        "expensive",
        "affordable",
        "new",
        "old",
        "used",
        "5g",
        "4g",
        "wifi",
        "inch",
        "gb",
        "tb",
        "ram",
        "rom",
    }
)

# --- Typo correction --------------------------------------------------------

# Maximum Levenshtein distance accepted for a typo correction.
MAX_EDIT_DISTANCE = 2

# Product vocabulary used for spell correction of Latin-script tokens.
_PRODUCT_VOCAB = frozenset(
    {
        "phone",
        "cover",
        "case",
        "shoes",
        "sneakers",
        "sandals",
        "chappals",
        "bottle",
        "water",
        "headphones",
        "earphones",
        "earbuds",
        "charger",
        "cable",
        "watch",
        "laptop",
        "bag",
        "wallet",
        "belt",
        "shirt",
        "tshirt",
        "jeans",
        "trousers",
        "dress",
        "saree",
        "kurta",
        "sofa",
        "chair",
        "table",
        "bed",
        "mattress",
        "pillow",
        "towel",
        "soap",
        "shampoo",
        "oil",
        "cream",
        "powder",
        "rice",
        "atta",
        "dal",
        "sugar",
        "salt",
        "tea",
        "coffee",
        "milk",
        "biscuit",
        "bread",
        "butter",
        "paneer",
        "curd",
        "pickle",
        "jam",
    }
)


def _tokenize(query: str) -> list[str]:
    """Split a query into Latin or Devanagari word tokens."""
    return re.findall(r"[A-Za-z]+|[ऀ-ॿ]+", query)


def detect_language(query: str) -> Language:
    """Detect the language of a query.

    Returns ``hi`` for Devanagari-only, ``hi_latn`` for Latin-script text
    containing Hinglish words, ``en`` for other Latin-script text, ``mixed``
    when both scripts appear, and ``other`` otherwise.
    """
    toks = _tokenize(query)
    has_devanagari = any(_DEVANAGARI_RE.fullmatch(t) for t in toks)
    has_latin = any(_LATIN_RE.search(t) for t in toks)
    if has_devanagari and has_latin:
        return Language.mixed
    if has_devanagari:
        return Language.hi
    if has_latin:
        if any(t.lower() in _HINGLISH_WORDS for t in toks):
            return Language.hi_latn
        return Language.en
    return Language.other


def classify_intent(query: str) -> tuple[Intent, str | None]:
    """Classify the intent of a query.

    Returns ``(intent, brand)`` where ``brand`` is set only for brand intent.
    Brand detection takes precedence, then attribute words, then product, then
    unknown for empty or unrecognizable input.
    """
    words = {t.lower() for t in _tokenize(query)}
    brand = next((b for b in sorted(_BRANDS) if b in words), None)
    if brand:
        return Intent.brand, brand
    if words & _ATTRIBUTE_WORDS:
        return Intent.attribute, None
    if words:
        return Intent.product, None
    return Intent.unknown, None


def _levenshtein(a: str, b: str) -> int:
    """Bounded Levenshtein distance; returns 999 if distance exceeds MAX_EDIT_DISTANCE."""
    if abs(len(a) - len(b)) > MAX_EDIT_DISTANCE:
        return 999
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        row_min = 999
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                curr[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            )
            row_min = min(row_min, curr[j])
        if row_min > MAX_EDIT_DISTANCE:
            return 999
        prev = curr
    return prev[-1]


def correct_word(word: str, vocab: frozenset[str]) -> str:
    """Correct a single Latin word using bounded edit distance against vocab."""
    if word in vocab or len(word) <= MAX_EDIT_DISTANCE:
        return word
    best = word
    best_dist = MAX_EDIT_DISTANCE + 1  # only accept corrections within MAX_EDIT_DISTANCE
    for v in sorted(vocab):
        d = _levenshtein(word, v)
        if d < best_dist:
            best_dist = d
            best = v
    return best if best_dist <= MAX_EDIT_DISTANCE else word


def correct_typo(query: str) -> str:
    """Correct obvious typos in Latin-script tokens against the product
    vocabulary. Devanagari tokens and Hinglish words are left unchanged."""
    corrected: list[str] = []
    for t in _tokenize(query):
        if _DEVANAGARI_RE.fullmatch(t):
            corrected.append(t)
            continue
        lower = t.lower()
        if lower in _PRODUCT_VOCAB or lower in _HINGLISH_WORDS or len(lower) <= MAX_EDIT_DISTANCE:
            corrected.append(t)
        else:
            fixed = correct_word(lower, _PRODUCT_VOCAB)
            corrected.append(fixed if fixed != lower else t)
    return " ".join(corrected)


def understand(query: str) -> Understanding:
    """Run full query understanding and return a contract ``Understanding``."""
    language = detect_language(query)
    intent, brand = classify_intent(query)
    corrected = correct_typo(query)
    return Understanding(
        corrected_query=corrected,
        language=language,
        intent=intent,
        brand=brand,
        source="rule-based",
    )
