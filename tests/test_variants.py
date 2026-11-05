"""Unit tests for variant generation determinism (data/make_variants.py)."""

from data.make_variants import VARIANT_TYPES, make_variants


def test_variants_deterministic():
    queries = ["phone cover", "shoes", "water bottle"]
    v1 = make_variants(queries, seed=42)
    v2 = make_variants(queries, seed=42)
    assert v1 == v2


def test_variants_all_types():
    queries = ["phone cover"]
    v = make_variants(queries, seed=42)
    for vt in VARIANT_TYPES:
        assert vt in v
        assert len(v[vt]) == 1
