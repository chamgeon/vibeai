# VibeAI

Exploring how an AI can build and refine a representation of the "vibe" a user perceives from photos, via a decompose-then-verify pipeline: generate a vibe representation → decompose it into atomic vibe claims → judge the claims with an LLM.

`vibeai.pipeline.workflow` closes that pipeline into a generator-evaluator loop: the judges' findings go back to the generator as feedback, and it revises until both accept or the round budget runs out.

## Metrics

| metric | question | reference |
| --- | --- | --- |
| decomposition quality | do the atoms faithfully and completely decompose the representation? | the representation |
| plausibility | is each atom true of the image? | the image |
| richness | how much of what is really there did the representation say? | a vibe pool built from four branches |
| breadth | of the vibe aspects this image supports, how many did it touch at all? | the image |

Breadth is the pool-free stand-in for richness. The pool is the answer key richness scores against, so the workflow's richness gate cannot use it without optimising against its own benchmark; breadth asks the same coverage question of the image, over the ten vibe aspects in `prompts/breadth_eval.py`.

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

# score the workflow's output with the same judges/models as the prompt runs:
# --representation-prompt-version=workflow reuses results/workflow/*.representations.json,
# and runs the workflow for any image that doesn't have one yet.
uv run pytest tests/test_plausibility.py --representation-prompt-version=workflow --n-images=all -s
uv run pytest tests/test_richness.py --representation-prompt-version=workflow --n-images=all -s

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
# (DEFAULT_MODEL / DEFAULT_DECOMPOSITION_MODEL / DEFAULT_EVAL_MODEL)
uv run pytest tests/test_decomposition_quality.py --representation-model=gpt-5 -s
uv run pytest tests/test_decomposition_quality.py --decomposition-model=gpt-5 -s
uv run pytest tests/test_decomposition_quality.py --eval-model=gpt-5 -s

# build the vibe pool for a holdout set (4 branches: 2 models x 2 representation prompts)
uv run -m vibeai.pipeline.pool_construction --image-dir data/richness_holdout
uv run -m vibeai.pipeline.pool_construction --n-images 5 --models gpt-5 claude-opus-5

# generator-evaluator workflow: loop one image until both judges accept it.
# Every call in the loop uses DEFAULT_MODEL.
uv run -m vibeai.pipeline.workflow data/richness_holdout/2025.jpg
uv run -m vibeai.pipeline.workflow <image> --max-rounds 5 --plausible-rate 0.9 --out trace.json

# ...and over an image set. --pool takes the image set from a vibe pool file, so the run
# covers exactly the images richness can then be scored on.
uv run -m vibeai.pipeline.workflow_batch --pool
uv run -m vibeai.pipeline.workflow_batch --image-dir data/main_processed --n-images 20 --concurrency 10
# writes results/workflow/<run>.json (summary), .per_image.jsonl (full traces) and
# .representations.json (image -> final representation, the input to a scoring pass)

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
