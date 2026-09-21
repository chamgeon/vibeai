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

# richness: how much of each holdout image's vibe pool a representation covered.
# The image set comes from the pool file, not --image-dir.
uv run pytest tests/test_richness.py -s
uv run pytest tests/test_richness.py --pool=annotations/richness_holdout.vibes.json -s

# other batch eval options (defaults shown)
uv run pytest tests/test_decomposition_quality.py --image-dir=data/main_processed -s
uv run pytest tests/test_decomposition_quality.py --representation-prompt-version=v1 -s
uv run pytest tests/test_decomposition_quality.py --decomposition-prompt-version=v2 -s
uv run pytest tests/test_decomposition_quality.py --concurrency=30 -s

# models: the two pipeline calls and the judge are set independently
# (representation/decomposition default to DEFAULT_MODEL, the judge to DEFAULT_EVAL_MODEL)
uv run pytest tests/test_decomposition_quality.py --representation-model=gpt-5.6-luna -s
uv run pytest tests/test_decomposition_quality.py --decomposition-model=gpt-5.6-luna -s
uv run pytest tests/test_decomposition_quality.py --eval-model=gpt-5 -s

# build the vibe pool for a holdout set (4 branches: 2 models x 2 representation prompts)
uv run -m vibeai.pipeline.pool_construction --image-dir data/richness_holdout
uv run -m vibeai.pipeline.pool_construction --n-images 5 --models gpt-5.6-luna claude-opus-5

# human annotation webapps + read-only results viewers
# (decomposition quality, plausibility, richness — cross-linked from every start screen)
uv run uvicorn vibeai.webapp.server:app --reload   # then open http://localhost:8000

# LLM/human agreement (Cohen's kappa)
uv run -m vibeai.eval.human_alignment <run> --metric <metric> --annotator <name>
```

Models are routed by id: ids starting with `claude-` go to the Anthropic Messages API (needs `ANTHROPIC_API_KEY`), everything else to the OpenAI Responses API. The daily token budget and per-call usage log track the OpenAI account's TPD limit only, so Anthropic calls are cached and retried but not metered.

LLM calls are cached under `.cache/llm/`, keyed by `(model, prompt, image)`. Batch results are written under `results/<metric_name>/<run_name>.json` (summary) and `.per_image.jsonl` (per-image detail).

## Extending

- **Prompt version**: add an entry to the relevant dict in `vibeai/prompts/` (e.g. `representation.py`), then pass `prompt_version="your_version"`.
- **Metric**: subclass `vibeai.metrics.base.Metric`, implement `measure(test_case) -> MetricResult`; see `metrics/decomposition_quality.py` or `metrics/plausibility.py` for examples.

See `vibeai/` module docstrings and `tests/` for further detail on project layout.
