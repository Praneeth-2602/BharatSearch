# ADR-002: LLM cascade policy for query understanding

## Status

Accepted

## Context

Query understanding rewrites noisy (Hinglish, typo, Devanagari) queries into
clean English for the English catalog. Calling the LLM for every query is
expensive and slow. The pipeline (Implementation Guide Section 5) has a cheap
path (normalization + candidate retrieval) and an LLM path (RAG with
candidate grounding).

## Decision

Use a cascade: call the LLM only when the query is not detected as clean
English, or when the BM25 top score for the normalized query is below a
threshold configured in `config/default.yaml`. Keep-or-drop decision for the
LLM step is made on `gold_dev`: keep the LLM only if the NDCG@10 gain on the
`hinglish_roman` and `codemixed` slices has a confidence interval above zero
and p95 added latency on a cache miss stays within budget.

## Consequences

- Most clean-English traffic bypasses the LLM, keeping p99 latency low.
- Non-English and typo queries get LLM grounding, closing the Hinglish gap.
- If the LLM step does not pass the gate, the cheap path ships and the
  decision is recorded in the write-up (honest negative result).
- Fallback to the normalized raw query (`source=fallback`) keeps search
  working even when the model server is down.

## Related

ADR-001 (RRF fusion), ADR-003 (tokenizer gate)
