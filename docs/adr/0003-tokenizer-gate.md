# ADR-003: Tokenizer gate (sarvam1 vs XLM-R vs byte-level)

## Status

Accepted

## Context

The catalog is English, so a custom Indic tokenizer has no useful role as a
BM25 analyzer unless Indic-language product fields are added. The tokenizer
can help on the query side: experiment T1 (Implementation Guide Section 5.5)
trains a 2-4 layer transformer bi-encoder from scratch on (variant query,
source English query) pairs and compares recall@1/recall@5 across tokenizers:
sarvam1, XLM-R, and byte-level.

## Decision

sarvam1 is the default choice and is wired into the product (as the fast-path
encoder or the language classifier's input) only if it matches or beats the
best baseline on the `hinglish_roman` and `devanagari` slices by paired
bootstrap, and the fertility benchmark supports it. Otherwise, publish the
comparison as a research note and use the best-performing tokenizer.

## Consequences

- The decision is evidence-driven, not preference-driven.
- If sarvam1 loses, the comparison is still published as a legitimate finding.
- The catalog remains English-only; no Indic product fields are added unless
  a future stretch adds LLM-translated titles indexed in `title_tok`.

## Related

ADR-002 (LLM cascade policy)
