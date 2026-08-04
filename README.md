# VibeAI

Exploring how an AI can build and refine a representation of the "vibe" a user perceives from photos.

A vibe representation (natural language, structured scores, or embeddings) is evaluated against six criteria: plausibility, interpretability, richness, robustness, discriminability, and personalization. This repo currently focuses on the natural-language track and its **plausibility** evaluation, built via a decompose-then-verify pipeline: generate a vibe representation → decompose it into atomic vibe claims → judge the claims with an LLM.

## Project layout

```
preprocess.py            image ingestion: normalize, resize, dedupe -> data/main_processed/

vibeai/
  prompts/                versioned prompt strings (representation, decomposition, judges)
  llm/
    client.py              OpenAI Responses API wrapper, with disk caching
    budget.py              daily token-budget tracking, stops a run before it blows the account's TPD cap
  pipeline/
    represent.py          image -> natural-language vibe representation
    decompose.py          vibe representation -> list of atomic vibe claims
    evaluate.py            image -> representation -> decomposition -> judged MetricResult, in one call
  metrics/
    base.py               Metric base class (measure() -> score + reason; optional extract_submetrics() for prompt-level rollups)
    decomposition_quality.py  LLM-judge metric: Completeness, Claim Independence, Atom Quality
  eval/
    test_cases.py         dataclasses passed between pipeline and metrics
    dataset.py            loads images from data/main_processed/
    concurrency.py         bounded-concurrency helper for running many async LLM calls at once
    parsing.py            JSON extraction from LLM output
    results.py            logs each test run's scores to results/*.json (image-level, run-scoped)
    prompt_results.py      aggregates per-image MetricResults into a PromptEvalResult for one prompt version

tests/                    pytest suites (deepeval-style: pipeline output -> metric -> assert)
                           conftest.py: --n-images, --image-dir, --representation-prompt-version,
                           --decomposition-prompt-version, --concurrency CLI options for batch eval tests
conftest.py               (empty) makes the vibeai package importable by pytest - do not delete
```

## Setup

```bash
uv sync
```

Requires an `OPENAI_API_KEY` in `.env`.

## Running

```bash
uv run pytest tests/ -v -s
```

By default, batch eval tests (e.g. `test_decomposition_quality_batch`) run on a small sample of `data/main_processed/` for a quick check. Control the sample size with `--n-images`:

```bash
uv run pytest tests/test_decomposition_quality.py --n-images=20 -s    # 20-image sample
uv run pytest tests/test_decomposition_quality.py --n-images=all -s   # every image in data/main_processed/
```

LLM calls are cached under `.cache/llm/`, keyed by `(model, prompt, image)` - re-running the same images/prompt versions is free and doesn't count against the daily token budget. Note that `--n-images=N` for `N < len(dataset)` samples randomly (fixed seed), so raising `N` across separate runs is *mostly* but not guaranteed cache-hit; use `--n-images=all` directly if you want a single run with no resampling.

Other batch eval options, all defaulting to the baseline setup:

```bash
uv run pytest tests/test_decomposition_quality.py --image-dir=data/holdout_set -s                        # source images (default: data/main_processed/)
uv run pytest tests/test_decomposition_quality.py --representation-prompt-version=v2 -s                  # representation prompt version (default: baseline)
uv run pytest tests/test_decomposition_quality.py --decomposition-prompt-version=v2 -s                   # decomposition prompt version (default: baseline)
uv run pytest tests/test_decomposition_quality.py --concurrency=10 -s                                    # max concurrent evaluations (default: 30)
```

### Results

`test_decomposition_quality_batch` writes prompt-level results under `results/`, via `vibeai.eval.prompt_results.aggregate_prompt_results`:

- `results/<metric_name>/<run_name>.json` - a `PromptEvalResult` summary for one metric + prompt-version combo: `mean_score`, `std_score`, `pass_rate`, per-submetric means (via `Metric.extract_submetrics`), `failures` (below threshold), and `errors` (images whose pipeline call raised instead of scoring). Small enough to diff between prompt versions to see whether a change helped.
- `results/<metric_name>/<run_name>.per_image.jsonl` - one line per image (`image_path`, `score`, `passed`, and `details`: representation, atoms, full judge verdict) - the full detail behind any regression in the summary above.

`vibeai.eval.results.result_log` (a simpler run-scoped image-level logger, saved to `results/<run>.json` at session end) is also available for ad-hoc tests, but isn't used by the batch eval test above.

## Adding a prompt version

Add an entry to the relevant dict in `vibeai/prompts/` (e.g. `representation.py`), then pass `prompt_version="your_version"` when calling the pipeline or reference it in a test.

## Adding a metric

Subclass `vibeai.metrics.base.Metric`, implement `measure(test_case) -> MetricResult`. See `metrics/decomposition_quality.py` for an example backed by an LLM-as-judge prompt. Optionally override `extract_submetrics(result) -> dict[str, float]` (0-1 normalized) to expose named sub-scores in prompt-level aggregation - see `DecompositionQualityMetric.extract_submetrics` for its Completeness/Claim Independence/Atom Quality breakdown.
