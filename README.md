# VibeAI

Exploring how an AI can build and refine a representation of the "vibe" a user perceives from photos.

A vibe representation (natural language, structured scores, or embeddings) is evaluated against six criteria: plausibility, interpretability, richness, robustness, discriminability, and personalization. This repo currently focuses on the natural-language track and its **plausibility** evaluation, built via a decompose-then-verify pipeline: generate a vibe representation → decompose it into atomic vibe claims → judge the claims with an LLM.

## Project layout

```
preprocess.py            image ingestion: normalize, resize, dedupe -> data/main_processed/

vibeai/
  prompts/                versioned prompt strings (representation, decomposition, judges)
  llm/client.py           OpenAI Responses API wrapper, with disk caching
  pipeline/
    represent.py          image -> natural-language vibe representation
    decompose.py          vibe representation -> list of atomic vibe claims
  metrics/
    base.py               Metric base class (measure() -> score + reason)
    decomposition_quality.py  LLM-judge metric: Completeness, Claim Independence, Atom Quality
  eval/
    test_cases.py         dataclasses passed between pipeline and metrics
    dataset.py            loads images from data/main_processed/
    parsing.py            JSON extraction from LLM output
    results.py            logs each test run's scores to results/*.json

tests/                    pytest suites (deepeval-style: pipeline output -> metric -> assert)
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

Each test run's scores and judge output are saved to `results/<run>.json`. LLM calls are cached under `.cache/llm/`, keyed by (model, prompt, image) - safe to delete if you want a clean rerun.

## Adding a prompt version

Add an entry to the relevant dict in `vibeai/prompts/` (e.g. `representation.py`), then pass `prompt_version="your_version"` when calling the pipeline or reference it in a test.

## Adding a metric

Subclass `vibeai.metrics.base.Metric`, implement `measure(test_case) -> MetricResult`. See `metrics/decomposition_quality.py` for an example backed by an LLM-as-judge prompt.
