# Threshold calibration (P4)

Embedder: `sentence-transformers/all-MiniLM-L6-v2`, cosine, `top_k=5`.

## Positives - factual questions that must retrieve their target chunk

| Question | Target chunk | Score | Retrieved |
|---|---|---|---|
| `What is the exit load on the Axis Large Cap Fund Direct plan?` | s01-c017, s03-c051, s03-c238 | 0.724 | yes |
| `What is the exit load on the Axis Large Cap Fund Regular plan?` | s02-c016, s03-c051, s03-c238 | 0.740 | yes |
| `What is the exit load on the Axis ELSS Tax Saver Fund?` | s09-c018 | 0.000 | NO |
| `What is the lock-in period of the Axis ELSS Tax Saver Fund?` | s09-c000, s09-c007, s09-c011, s09-c029, s10-c028 | 0.844 | yes |
| `What is the minimum lump sum investment for the Axis ELSS Tax Saver Fund?` | s09-c018, s12-c047 | 0.762 | yes |
| `What is the expense ratio of the Axis Flexi Cap Fund Direct plan?` | s04-c000, s06-c224 | 0.827 | yes |
| `What is the benchmark of the Axis Flexi Cap Fund?` | s04-c000, s06-c194 | 0.840 | yes |
| `What is the risk profile of the Axis Flexi Cap Fund Direct plan?` | s04-c001, s04-c023 | 0.833 | yes |
| `What is the minimum investment amount for the Axis Midcap Fund Direct plan?` | s14-c018 | 0.740 | yes |
| `How do I download an account statement?` | s16-c008 | 0.000 | NO |

**Recall@5: 8/10 = 80.0%**

## Negatives - off-topic questions that must retrieve nothing

| Question | Best score | Clears 0.35? |
|---|---|---|
| How do I cook pasta? | 0.112 | no |
| What is the weather in Mumbai tomorrow? | 0.268 | no |
| Who won the last World Cup? | 0.130 | no |
| Write me a Python function to sort a list. | 0.091 | no |
| What is the capital of France? | 0.204 | no |
| How do I lose weight fast? | 0.115 | no |
| Explain quantum computing to me. | 0.128 | no |
| What is 2 plus 2? | 0.205 | no |
| Book me a flight to Delhi. | 0.348 | no |
| Tell me a joke about cats. | 0.035 | no |

## Retrieval misses

On-topic questions whose target chunk did not reach the top 5. These are *retrieval* failures, not threshold failures: they score far above any sane threshold, so raising the threshold cannot fix them and would only cost recall.

| Question | Target | Best score |
|---|---|---|
| What is the exit load on the Axis ELSS Tax Saver Fund? | s09-c018 | 0.785 |
| How do I download an account statement? | s16-c008 | 0.356 |

## Separation band

Positives are the correct chunk's score. Negatives are the best score reached by an *off-topic* question. Deliberately excluding the misses above: the threshold separates off-topic from on-topic, it does not separate right from wrong answers. A wrong-but-relevant chunk is caught by the LLM sentinel and the output verifier.

- Lowest positive (weakest correct match): **0.724**
- Highest negative (strongest off-topic match): **0.348**
- Midpoint: **0.536**

## Recommendation

`SIMILARITY_THRESHOLD = 0.54`

The configured value is `0.35`. It does **not** sit inside the band and should be changed to the midpoint.

Strongest off-topic score was 0.348; a question that reaches only 0.348 is rendered as `NO_RETRIEVAL` rather than answered.
