RICHNESS_EVAL_PROMPT = """# Vibe Coverage Judgement
You are judging how well a target vibe set covers a pool of candidate vibes.
Both the target vibes and the pool vibes were extracted from vibe representations of the same image. A vibe representation is a natural language description of the mood, atmosphere, aesthetic character, or experiential feeling the image evokes.
You will see only the target vibe set and the vibe pool, not the image.

## Distinctness of vibe
A vibe is distinct with respect to a set of vibes if it contributes affective meaning not already conveyed by that set.
Put another way, a distinct vibe narrows the range of images the set could be describing.

A pool vibe that is redundant with respect to the target set is **covered** by the target set.
A pool vibe that is distinct with respect to the target set is **not covered** by the target set.

## Task
For each vibe in the pool, judge whether it is distinct from or redundant with the target set.

## Procedure
Apply the witness test to each pool vibe separately.

**Witness test**
Try to describe an image that fits every vibe in the target set but clearly does not fit the pool vibe.
- If you can describe such an image → the pool vibe narrows down the feeling further → distinct.
- If every image fitting the target set would also fit the pool vibe → redundant.

The witness image must be one an ordinary viewer would readily accept, not a contrived edge case built on a wording technicality.

## Rules
1. Judge each pool vibe against the target set only. Never compare pool vibes with each other, and never let the verdict for one pool vibe influence another. Two near-identical pool vibes should receive the same verdict.
2. The target set is fixed. Do not add pool vibes to it as you go.
3. If the target set is empty, every pool vibe is distinct.
4. If a pool vibe is in tension with the target set, it is distinct, not invalid.
5. Judge against the target set as a whole. Two or more target vibes can jointly entail a pool vibe.
6. Judge every pool vibe, in the given order, even if it appears word for word in the target set.

## Examples
target: ["cozy", "calm"]
pool vibe: "peaceful"
witness_test: any cozy, calm image already reads as peaceful.
verdict: redundant

target: ["cozy", "calm"]
pool vibe: "nostalgic"
witness_test: an image of a bright new minimalist apartment in soft daylight is cozy and calm, with nothing nostalgic about it.
verdict: distinct

target: ["sunny", "laid-back", "beachy"]
pool vibe: "slow summer afternoon"
witness_test: an image of a bright morning beach scene is sunny, laid-back, and beachy, but not an afternoon.
verdict: distinct

target: ["quietly productive", "focused", "unhurried"]
pool vibe: "concentrated"
witness_test: 'focused' already carries the 'concentrated' vibe.
verdict: redundant

target: ["high-energy", "crowded dance floor energy"]
pool vibe: "chaotic"
witness_test: an image of a tightly choreographed, packed club floor is high-energy and crowded, but controlled rather than chaotic.
verdict: distinct

target: ["cluttered", "warm", "personal"]
pool vibe: "lived-in"
witness_test: cluttered, warm, and personal traces jointly entail lived-in.
verdict: redundant

## Output Format
Return only the following JSON object. No prose, no markdown code fences.
`judgements` must contain exactly one object per pool vibe, in the same order as the input pool.

{{
  "target": ["<target vibes>"],
  "judgements": [
    {{
      "candidate": "<pool vibe>",
      "witness_test": "<witness test>",
      "verdict": "distinct" | "redundant"
    }}
  ]
}}

## Target vibe set
{target}

## Vibe pool
{pool}
"""