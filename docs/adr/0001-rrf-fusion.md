# ADR-001: Reciprocal Rank Fusion for hybrid retrieval

## Status

Accepted

## Context

The search API combines BM25 (lexical) and dense vector retrieval results
into a single ranked list. Two candidate fusion strategies were considered:

1. **Reciprocal Rank Fusion (RRF):** `score = sum(1 / (k + rank))` over each
   retriever's top-N list, with tunable `k`.
2. **Weighted score fusion:** normalize each retriever's scores to [0, 1]
   and take a weighted sum.

## Decision

Use RRF with `k = 60` and `top = 100`, as implemented in
`services/retrieval/bharatsearch_retrieval/fusion.py`. Tune `k` against
`gold_dev` only; compare RRF against weighted score fusion on the dev set
before finalizing.

## Consequences

- RRF is rank-based and does not require score calibration across retrievers
  with different score distributions (BM25 scores are unbounded, dense
  cosine is in [-1, 1]).
- RRF is parameter-light: only `k` needs tuning.
- RRF can over-reward documents that appear in both lists at mediocre ranks;
  this is mitigated by tuning `k` and by the ranker's downstream re-ranking.
- The decision was recorded with the Section 3.4 acceptance test: BM25-only
  and hybrid runs exist for all variants, and the Hinglish gap is visible.

## Related

ADR-002 (LLM cascade policy), ADR-003 (tokenizer gate)
