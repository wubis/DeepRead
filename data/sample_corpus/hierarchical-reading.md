# Progressive Hierarchical Reading

## Evidence budgets

Flat retrieval commonly sends every top-ranked passage to a generator, whether or not each passage fills an evidence gap. Progressive reading first inspects low-cost views such as titles and summaries, then opens sections or passages only when their expected evidence gain justifies the additional token cost.

## Coverage-based stopping

An evidence checklist changes the stopping rule from similarity to completeness. The controller stops when requirements are supported or when its search, latency, or token budget is exhausted. Logging skipped and selected reading actions makes the efficiency claim inspectable.

## Full-document fallback

Full documents are useful when evidence depends on distant context, but they are expensive. A hierarchical reader reserves full-document access for cases where cheaper views cannot resolve an uncovered requirement.

