# VibeAI

Exploring how an AI can build and refine a representation of the "vibe" a user perceives from photos, via a decompose-then-verify pipeline: generate a vibe representation → decompose it into atomic vibe claims → judge the claims with an LLM.

## Setup

```bash
uv sync
```

Requires an `OPENAI_API_KEY` in `.env`.

## Commands

```bash
# run all tests
uv run pytest tests/ -v -s

# batch eval on a sample / all images
uv run pytest tests/test_decomposition_quality.py --n-images=20 -s
uv run pytest tests/test_plausibility.py --n-images=all -s

# other batch eval options (defaults shown)
uv run pytest tests/test_decomposition_quality.py --image-dir=data/main_processed -s
uv run pytest tests/test_decomposition_quality.py --representation-prompt-version=baseline -s
uv run pytest tests/test_decomposition_quality.py --decomposition-prompt-version=baseline -s
uv run pytest tests/test_decomposition_quality.py --concurrency=30 -s

# human annotation webapp
uv run uvicorn vibeai.webapp.server:app --reload   # then open http://localhost:8000

# LLM/human agreement (Cohen's kappa)
uv run -m vibeai.eval.human_alignment <run> --metric <metric> --annotator <name>
```

LLM calls are cached under `.cache/llm/`, keyed by `(model, prompt, image)`. Batch results are written under `results/<metric_name>/<run_name>.json` (summary) and `.per_image.jsonl` (per-image detail).

## Extending

- **Prompt version**: add an entry to the relevant dict in `vibeai/prompts/` (e.g. `representation.py`), then pass `prompt_version="your_version"`.
- **Metric**: subclass `vibeai.metrics.base.Metric`, implement `measure(test_case) -> MetricResult`; see `metrics/decomposition_quality.py` or `metrics/plausibility.py` for examples.

See `vibeai/` module docstrings and `tests/` for further detail on project layout.
