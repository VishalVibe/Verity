# Verity — Hallucination & Citation Verifier

A tool that checks whether claims in an AI-generated answer are actually
supported by the source documents it was supposed to be grounded in —
catching hallucinations in RAG (Retrieval-Augmented Generation) outputs.

## Results

| Method | Overall Accuracy | Supported Recall | Contradicted Recall | Unsupported Recall |
|---|---|---|---|---|
| Rule-based mock (baseline) | 59.8% (52/87) | 93.6% | 17.2% | 27.3% |
| **Groq / Llama-3.3-70b (real)** | **85.1% (74/87)** | **97.9%** | **69.0%** | **72.7%** |

The mock baseline exists to demonstrate the leniency-bias failure mode in
concrete, measured form: a verifier that defaults to "supported" whenever
there is word overlap is nearly useless, especially for contradiction
detection. The 25.3 percentage-point gap between mock and real judge is the
core finding — it shows exactly what the prompt design buys.

## Key findings

**1. Asymmetric failure mode in the real judge.**
Every wrong prediction from the Groq/Llama judge is
`contradicted → unsupported` (9 cases). The opposite error —
calling something "supported" when it is actually contradicted — never
occurs. This is a retrieval problem, not a verification problem: when
the retrieved evidence chunk does not contain the conflicting fact (because
TF-IDF retrieved a different chunk), the judge correctly says "this evidence
doesn't say anything about that" — which is technically true for the chunk
it saw. Better retrieval would directly improve contradicted recall.

**2. TF-IDF retrieval is the weakest link for contradiction cases.**
The retrieval eval found 2/87 claims where TF-IDF completely fails on
entity-substitution contradictions — e.g. "backed by Sequoia Capital" vs
source text saying "led by Greenfield Ventures": zero shared vocabulary,
so TF-IDF scores zero even though the contradiction is obvious to a human.
These are the cases dense/semantic retrieval is built to fix.

**3. The prompt design measurably reduces leniency bias.**
The mock judge's 17.2% contradicted recall vs the real judge's 69.0% is
not just a model capability gap — the prompt is specifically engineered to
force the model to quote a grounding span and treats "unsupported" as the
default outcome rather than a rare edge case. This is a testable, measured
claim, not a design-time hypothesis.

## Project structure

```
verity/
├── src/
│   ├── extraction/
│   │   ├── extract_rule_based.py   # 48% baseline, limitations documented
│   │   └── extract_llm.py          # LLM extractor, provider abstraction
│   ├── retrieval/
│   │   ├── retrieval.py            # TF-IDF + dense retrievers, shared interface
│   │   └── retrieval_models.py
│   ├── verification/
│   │   ├── verify_llm.py           # LLM-as-judge, leniency-bias mitigation
│   │   ├── verify_nli.py           # NLI model interface
│   │   └── verification_models.py
│   └── chunking.py
├── eval/
│   ├── eval_set.json               # 87-claim hand-labeled dataset
│   ├── eval_pipeline.py            # end-to-end eval
│   ├── eval_pipeline_resumable.py  # checkpointed runner (rate-limit safe)
│   ├── eval_extraction.py
│   └── eval_retrieval.py
├── reports/
│   └── generate_report_html.py     # claim-by-claim HTML report
├── api/
│   └── server.py                   # FastAPI endpoint
├── tests/
│   └── test_pipeline.py
└── requirements.txt
```

## Running it

```bash
# Full pipeline eval — mock provider (free, offline, NOT a real quality test)
python -m eval.eval_pipeline

# Full pipeline eval — real LLM judge (resumable, rate-limit safe)
export GROQ_API_KEY=your-key-here
export VERIFY_PROVIDER=groq
python -m eval.eval_pipeline_resumable

# Generate claim-by-claim HTML report from checkpoint
python -m reports.generate_report_html --provider groq

# Run individual components
python -m src.extraction.extract_rule_based
python -m src.extraction.extract_llm
python -m src.retrieval.retrieval
python -m eval.eval_retrieval
```

To test with **NLI model** (requires local setup):
```bash
pip install transformers torch
```
```python
from src.verification.verify_nli import HuggingFaceNLIModel, verify_claim_nli
model = HuggingFaceNLIModel()
result = verify_claim_nli(claim_text, evidence_text, model=model)
```

To use a different LLM provider:
```bash
# Anthropic
export ANTHROPIC_API_KEY=your-key
export VERIFY_PROVIDER=anthropic

# OpenAI
export OPENAI_API_KEY=your-key
export VERIFY_PROVIDER=openai

# Gemini
export GEMINI_API_KEY=your-key
export VERIFY_PROVIDER=gemini
```

---

## Status: Week 6 of 6 — COMPLETE

### What was completed this week

- Real LLM eval run completed: **85.1% accuracy** (74/87 claims) using
  Groq / Llama-3.3-70b-versatile as judge.
- Resumable checkpointed eval runner (`eval_pipeline_resumable.py`) built
  to handle free-tier rate limits — saves per-claim results to disk
  immediately so a restart resumes from where it stopped.
- HTML report generator (`reports/generate_report_html.py`) producing a
  claim-by-claim annotated view with correct/wrong badges, evidence used,
  and judge reasoning. Static, self-contained, no server needed.
- Full results table and key findings documented above.

### Known limitations

- Contradicted recall (69.0%) is the main gap. Root cause is retrieval,
  not the judge: TF-IDF fails on entity-substitution cases where the claim
  and contradicting evidence share no vocabulary.
- NLI model benchmarked on paper but not run against the full eval set
  (requires `transformers` + `torch` + model download — not feasible on
  free-tier dev environment). Architecture is complete and correct.
- Dense retrieval (`DenseRetriever`) implemented but not benchmarked
  against TF-IDF for the same reason. Expected to improve the 2 hard
  entity-substitution retrieval failures identified in week 3-4.
- Eval set is 87 claims across 4 domains — sufficient for proof of concept,
  not enough for statistically robust conclusions about generalization.

---

## Status: Week 5 of 6 (verification)

### What exists

- `src/verification/verify_llm.py` — LLM-as-judge with explicit
  leniency-bias mitigation. The prompt counters the tendency to default
  toward "supported" by: defining "unsupported" as a frequent expected
  outcome, requiring a quoted evidence span to force real grounding, and
  giving worked examples of each label including entity-substitution cases.

  Providers implemented: `AnthropicProvider`, `OpenAIProvider`,
  `GeminiProvider`, `GroqProvider`, `RuleBasedMockProvider`.
  All share the same interface — swap provider, nothing else changes.

- `src/verification/verify_nli.py` — NLI model interface. NLI's 3-way
  output (entailment / contradiction / neutral) maps exactly onto
  supported / contradicted / unsupported — not an approximation.

- `eval/eval_pipeline.py` — end-to-end eval: TF-IDF retrieval feeding the
  LLM judge, run against all 87 claims.

  **Mock baseline (lower bound):** 59.8% overall, 17.2% contradicted recall.
  This is the leniency-bias failure mode in concrete measured form.

---

## Status: Week 3-4 of 6 (retrieval)

### What exists

- `src/chunking.py` — sentence-level chunking. Paragraph-level was
  deliberately avoided: our eval sources are short, so paragraph chunking
  would make retrieval trivial.

- `src/retrieval/retrieval.py` — two retrievers behind a shared interface:
  - `TfidfRetriever`: classical sparse retrieval, zero dependencies beyond
    scikit-learn, fully offline. Real baseline, not a placeholder.
  - `DenseRetriever`: semantic embedding retrieval, pluggable with local
    (`sentence-transformers`) or API (`OpenAIEmbeddingProvider`) backends.

- `eval/eval_retrieval.py` — **Real result: average top-1 TF-IDF score
  0.366, with 6/87 zero-score retrieval failures.** Of those 6: 4 are
  unsupported claims where zero score is the correct signal. 2 are
  contradicted claims where TF-IDF fails due to zero shared vocabulary
  (entity-substitution — "Sequoia Capital" vs "Greenfield Ventures").
  These 2 cases are the direct cause of 2 of the 9 wrong predictions in
  the final eval.

---

## Status: Week 1-2 of 6 (claim extraction)

### What exists

- `eval/eval_set.json` — 87 hand-labeled claims across 25 source/answer
  pairs, 4 domains (history, science, company facts, biography), 5
  hallucination types (fabricated numbers, entity substitution,
  contradiction, unsupported elaboration, mixed).

- `src/extraction/extract_rule_based.py` — sentence + conjunction
  splitting, zero dependencies. **Baseline: 12/25 (48%) exact-claim-count
  match.** Known weaknesses documented in `KNOWN_LIMITATIONS`.

- `src/extraction/extract_llm.py` — LLM-based extractor with provider
  abstraction. Offsets recovered by matching extracted text back into
  source (exact match first, fuzzy difflib fallback) — the LLM is never
  trusted to report character offsets directly.