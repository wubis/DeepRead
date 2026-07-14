# Hybrid Information Retrieval

## Lexical search

Keyword retrieval such as BM25 is precise when a query contains the same names and terminology as a relevant passage. It is inexpensive and its term-level scores are easy to inspect, but vocabulary mismatch can hide conceptually relevant material.

## Semantic search

Dense retrieval compares vector representations and can connect paraphrases or related concepts even when exact terms differ. It may, however, retrieve topically similar passages that do not contain the needed fact.

## Fusion

Hybrid retrieval combines complementary ranked lists. Reciprocal rank fusion is robust because it rewards documents that rank well in either channel without requiring raw scores to share a scale. Deduplication and provenance records keep the merged candidate pool useful and auditable.

