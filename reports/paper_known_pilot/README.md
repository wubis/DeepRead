# QASPER paper-known pilot

This pilot evaluates passage retrieval, evidence selection, and synthesis on a fixed
10-paper slice of the official QASPER v0.3 development split. The first 10 papers contain
30 questions, so the requested 30–50 question range is met without partially sampling an
additional paper.

- Corpus hash: `0d43d5b9b4149bae00f77deba7577652aabff11bcf7cb9969aa9cf0002421e13`
- Random seed: `17`
- Mode: paper-known
- Correct threshold: answer F1 ≥ 0.5

## Aggregate results

| Configuration | Correct | Answer F1 | Evidence F1 | Recall@20 | MRR@10 | Coverage | Read tokens | API tokens | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 only (offline) | 0/30 | 0.084 | 0.153 | 0.714 | 0.290 | 1.000 | 566.3 | 0.0 | 4.4 |
| Embeddings only (offline fallback) | 0/30 | 0.091 | 0.196 | 0.749 | 0.296 | 0.933 | 532.4 | 0.0 | 7.4 |
| Flat hybrid top-k (offline) | 0/30 | 0.091 | 0.175 | 0.786 | 0.362 | 0.967 | 565.3 | 0.0 | 8.2 |
| Hybrid + rerank flag (offline) | 0/30 | 0.091 | 0.175 | 0.786 | 0.362 | 0.967 | 565.3 | 0.0 | 8.1 |
| Hierarchical single-pass (offline) | 0/30 | 0.082 | 0.150 | 0.786 | 0.362 | 1.000 | 105.4 | 0.0 | 32.0 |
| Bounded EvidenceGraph (offline) | 0/30 | 0.082 | 0.150 | 0.786 | 0.362 | 1.000 | 105.4 | 0.0 | 33.5 |
| Hierarchical single-pass (OpenAI) | 1/30 | 0.131 | 0.390 | 0.819 | 0.736 | 0.294 | 243.7 | 8467.5 | 12772.2 |
| Bounded EvidenceGraph (OpenAI) | 0/30 | 0.151 | 0.378 | 0.819 | 0.699 | 0.408 | 539.0 | 10512.0 | 19759.4 |

Offline reranking is intentionally inert because no model is present; its row is a
configuration/provenance check and should match flat hybrid retrieval. Likewise, the
offline bounded loop stops after one round because deterministic coverage reaches 100%.

## OpenAI failure breakdown

### By matched answer type

| Answer type | Questions | Answer F1 | Evidence F1 | Recall@20 | Coverage |
|---|---:|---:|---:|---:|---:|
| abstractive | 5 | 0.153 | 0.233 | 0.550 | 0.400 |
| boolean | 2 | 0.000 | 0.200 | 0.750 | 0.500 |
| extractive | 23 | 0.163 | 0.425 | 0.884 | 0.402 |

### By gold evidence location

| Evidence location | Questions | Answer F1 | Evidence F1 | Recall@20 | Coverage |
|---|---:|---:|---:|---:|---:|
| figure_table | 1 | 0.059 | 0.000 | 0.000 | 0.000 |
| figure_table+paragraph | 4 | 0.178 | 0.417 | 0.833 | 0.417 |
| paragraph | 23 | 0.149 | 0.378 | 0.837 | 0.417 |
| paragraph+section_heading | 2 | 0.160 | 0.500 | 1.000 | 0.500 |

### Observed failure modes

- Boolean answers score zero even when relevant passages rank well because synthesis
  produces a qualified explanation instead of the benchmark's expected `Yes` or `No`.
- The table-only question has zero Recall@20, while mixed paragraph/table questions are
  substantially stronger; captions need their own retrieval treatment.
- Extractive questions reach high retrieval recall but low answer F1, showing that
  planning decomposition, evidence acceptance, and concise synthesis are the main
  bottlenecks after retrieval.
- Some gold passages are retrieved but rejected as insufficient, so reported coverage
  is much lower than retrieval recall.

## Calibration

| Run | Provider | Window | Coverage target | Answer F1 | Evidence F1 | Citation tokens | API tokens | Rounds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| calibration-openai-w1-c05 | openai | 1 | 0.500 | 0.072 | 0.700 | 123.0 | 10679.0 | 1.75 |
| calibration-openai-w1-c08 | openai | 1 | 0.800 | 0.120 | 0.710 | 135.2 | 12155.0 | 2.00 |
| calibration-openai-w1-c1 | openai | 1 | 1.000 | 0.027 | 0.417 | 125.5 | 10122.2 | 1.75 |
| calibration-openai-w3-c05 | openai | 3 | 0.500 | 0.176 | 0.825 | 145.5 | 9142.2 | 1.50 |
| calibration-openai-w3-c08 | openai | 3 | 0.800 | 0.092 | 0.592 | 206.5 | 12240.0 | 2.00 |
| calibration-openai-w3-c1 | openai | 3 | 1.000 | 0.126 | 0.500 | 193.8 | 12349.0 | 2.00 |
| calibration-window-1 | offline | 1 | 0.800 | 0.082 | 0.150 | 24.1 | 0.0 | 1.00 |
| calibration-window-2 | offline | 2 | 0.800 | 0.082 | 0.150 | 33.5 | 0.0 | 1.00 |
| calibration-window-3 | offline | 3 | 0.800 | 0.082 | 0.150 | 35.0 | 0.0 | 1.00 |
| calibration-window-5 | offline | 5 | 0.800 | 0.082 | 0.150 | 39.2 | 0.0 | 1.00 |

The provisional operating point is a 3-sentence evidence window with a 0.5 coverage
target. It produced the best observed answer and evidence F1 on the four-question
stratified calibration subset while using fewer API tokens than stricter targets. This
choice should be revalidated on a larger paired run because API outputs are stochastic.

## Provenance audit

All sampled annotations resolved: `0` unresolved references.
The audit covers paragraph, figure/table, and section-heading sources and verifies both
source offsets and mapped text. `FLOAT SELECTED:` annotation prefixes are normalized before
comparing them with stored figure/table captions.

| Question ID | Answer type | Source | Source path | Match | Offsets valid | Text valid |
|---|---|---|---|---|---|---|
| `b6f15fb627` | extractive | paragraph | `full_text.paragraphs[9][0]` | exact | True | True |
| `5eda469a8a` | abstractive | paragraph | `full_text.paragraphs[7][1]` | exact | True | True |
| `5eda469a8a` | abstractive | figure_table | `figures_and_tables[2]` | float_exact | True | True |
| `1f085b9bb7` | boolean | paragraph | `full_text.paragraphs[5][0]` | exact | True | True |
| `37861be6ae` | extractive | section_heading | `full_text.section_name[6]` | exact | True | True |
| `9ee07edc37` | extractive | figure_table | `figures_and_tables[4]` | float_exact | True | True |

## Representative traces

- [best-answer](traces/best-answer-18c5d366b1.json): what ner models were evaluated? (answer F1 0.412, evidence F1 0.333)
- [boolean-failure](traces/boolean-failure-1f085b9bb7.json): did they use a crowdsourcing platform for manual annotations? (answer F1 0.000, evidence F1 0.400)
- [figure-table-case](traces/figure-table-case-5eda469a8a.json): what language pairs are explored? (answer F1 0.269, evidence F1 0.667)
- [retrieval-miss](traces/retrieval-miss-a87a009c24.json): What predictive model do they build? (answer F1 0.043, evidence F1 0.000)

## Interpretation

This is a small pilot, not a statistically powered comparison. Paper recall is 1.0 by
construction in paper-known mode and should not be interpreted as document-retrieval
performance. Seed 17 controls deterministic components, but current OpenAI API calls do
not expose a seed in this pipeline, so paid calibration cells remain stochastic. The
results identify concrete next work: answer-type-aware synthesis, less
aggressive planning decomposition, evidence-assessor calibration, and explicit support for
negative boolean answers.

## Post-pilot priority fixes

The original pilot tables above are retained as the baseline. The first follow-up implements
the failure-driven changes without changing the 10-paper corpus, 30-question slice, corpus hash,
or random seed:

- atomic questions are constrained to one retrieval requirement, while explicitly compound
  questions can still decompose;
- synthesis uses answer-type-specific contracts, including short extractive outputs and a
  forced `Yes`/`No` schema for boolean questions;
- evidence assessment receives the original question and recognizes evidence supporting a
  negative boolean answer without requiring the literal word `No`;
- figure/table retrieval follows explicit references such as `Table 3` from highly ranked prose
  and fuses the caption conservatively;
- the bounded supervisor stops when retrieval candidates repeat, no evidence is added, or a
  later round adds no supported evidence.

### Deterministic 30-question regression

| Run | Correct | Answer F1 | Exact match | Boolean accuracy | Evidence F1 | Recall@20 | MRR@10 | Read tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original flat hybrid | 0/30 | 0.091 | 0.000 | 0.000 | 0.175 | 0.786 | 0.362 | 565.3 |
| Priority fixes | 2/30 | 0.158 | 0.067 | 1.000 | 0.175 | 0.786 | 0.362 | 560.9 |

This isolates the response-shape improvement: ranking and evidence metrics remain unchanged,
while both boolean questions become exact matches. On the benchmark/table case, the relevant
prose remains at ranks 1–2 and its cited table moves from rank 10 to rank 3 without changing
MRR or Recall@20.

### OpenAI diagnostic

On the same four-question calibration subset used for the selected window/coverage setting,
the first post-fix run increased answer F1 from `0.176` to `0.395`, moved from `0/4` to `2/4`
answers above the correctness threshold, reduced mean API tokens from `9,142` to `8,064`, and
reduced mean rounds from `1.50` to `1.25`. Evidence F1 decreased from `0.825` to `0.708`, so
concise synthesis improved faster than evidence selection.

That run exposed a boolean-schema edge case (`Unanswerable`). After replacing the generic
schema with a forced boolean contract, a targeted rerun returned the exact answer `No`
(`answer F1 = 1.0`, `yes/no accuracy = 1.0`). These paid checks are diagnostic rather than a
statistically powered comparison, and OpenAI output remains stochastic.

## Reader and cost-control follow-up

A second pass corrects the document-level redundancy penalty, opens up to two distinct evidence
spans per requirement, assesses only newly opened evidence on later rounds, separates benchmark
forced-choice answers from grounded abstention, and restricts model reranking to eight focused
candidate windows. Unused reads remain in the trace but are no longer reported as answer citations.

Passage-similarity diversity was calibrated at weights `0`, `0.01`, `0.02`, and `0.05`. Every
positive weight slightly reduced MRR on the paper-known slice, so the recorded default is `0.0`;
the implementation remains executable for corpus-wide experiments. With diversity disabled,
Recall@20 and MRR@10 exactly match the original flat-hybrid values (`0.786` and `0.362`).

The revised hierarchical reader raises conditional selection recall from `0.163` to `0.221` and
answer F1 from `0.147` to `0.161` in the deterministic 30-question regression. It reads 152 source
tokens per question, compared with 561 for flat reading. A later evaluator audit found that the
two apparent unanswerable items both contain mixed answerable/unanswerable annotations, so `0/2`
was not a valid consensus-unanswerable result.

### Four-question OpenAI regression

| Run | Correct | Answer F1 | Exact match | Boolean accuracy | Evidence F1 | MRR@10 | API tokens | Rounds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Calibration baseline | 0/4 | 0.176 | 0.000 | 0.000 | 0.825 | 1.000 | 9,142 | 1.50 |
| First priority pass | 2/4 | 0.395 | 0.000 | 0.000 | 0.708 | 1.000 | 8,064 | 1.25 |
| Reader and cost pass | 3/4 | 0.645 | 0.250 | 1.000 | 0.583† | 0.778 | 3,964 | 1.25 |

† The latest run scores only evidence cited by synthesized claims; earlier runs treated every
examined evidence span as a citation. The new run separately records read-evidence F1 (`0.625`)
and selection recall conditional on retrieval (`0.750`).

Reranking accounts for 8,509 of 15,857 total API tokens in the latest run, down from 26,142 of
32,255 in the first priority pass. A paired four-question bootstrap comparison estimates an
answer-F1 delta of `+0.469` (95% interval `[+0.210, +0.729]`) and an API-token delta of `-5,178`
per question (95% interval `[-7,183, -3,173]`). With only four paired questions, these intervals
are diagnostic rather than confirmatory.

The negative-boolean question remains the main evidence-selection failure. Expanding the model
reranker from eight to ten candidates moved its MRR only from `0.111` to `0.125`, did not recover
gold evidence, and increased tokens, so eight remains the calibrated cutoff.

## Answerability and negative-boolean follow-up

Evaluation schema v4 reports unanswerable accuracy only when every annotation marks a question
unanswerable. The 30-question pilot contains zero such questions and two answerability-disagreement
cases. This preserves maximum-over-annotator answer scoring without penalizing a valid answer merely
because another annotator abstained. Consensus abstention therefore still needs evaluation on a
larger held-out slice.

The runtime now gives boolean requirements separate routing keywords describing the underlying
activity that should appear even when the answer is No. One routing-based rescue candidate can be
added below the eight-item reranker cutoff. Evidence verdicts record explicit answer polarity, and
the supervisor can combine consistent partial negative evidence while requiring at least one
detailed method/results span. For universal or superlative questions, explicit counterevidence can
ground No even when another local result supports Yes.

Focused OpenAI validation produced exact `No` for both boolean pilot questions (`2/2`, yes/no
accuracy `1.0`). The crowdsourcing-platform case now cites only the gold `Test dataset` passage:
answer exact match `1.0`, citation precision `1.0`, citation recall `0.5`, and Evidence F1 `0.667`.
The BERT “among all algorithms” case cites the gold counterexample showing second-place performance
and also reaches Evidence F1 `0.667`.

The mixed-annotation question “What crowdsourcing platform is used?” now abstains instead of
claiming Amazon Mechanical Turk from related work, earning exact match against its unanswerable
annotation. This is evidence of safer attribution behavior, not a consensus-unanswerable result.
The detailed negative-evidence check required two rounds and 6,639 API tokens, so broader validation
should measure the quality/cost tradeoff before changing global budgets.
