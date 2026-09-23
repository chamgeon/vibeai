"""Breadth judge: the pool-free stand-in for richness inside the
generator-evaluator workflow.

The richness metric (``prompts/richness_eval.py``) scores a representation
against a vibe pool built from four independent branches. That makes it a good
*measurement* and a useless *evaluator*: the pool is the answer key, and a loop
that fed pool misses back to the generator would be training on the test set.

So the workflow's richness gate never sees the pool. It sees the image and asks
the question step 3 of the ``dr`` representation prompt asks - which of the ten
aspects a vibe can come from does this image actually support, and which of
those has the candidate spent no words on - and rejects on any aspect that is
there, carries a feeling, and went unnamed. Same notion of breadth, derived
from the image instead of from a reference set.

The dimension list is copied from ``dr`` deliberately: the generator can be
told to search along the same axes the evaluator scores, and the two stay in
step as the list is revised.
"""

BREADTH_EVAL_PROMPT = """# Vibe Breadth Judgement
You are judging the breadth of a candidate vibe representation of an image.
A vibe representation is a natural language description of the mood, atmosphere, aesthetic character, or experiential feeling an image evokes. The candidate has already been decomposed into vibe atoms - minimal statements each expressing one vibe, some naming the visual evidence behind it.
You will see the image and the candidate's atoms.

Your question is not whether the candidate's claims are true. It is whether there is more vibe in this image worth naming that the candidate has not reached.

## Breadth
Breadth is coverage: how many of the different aspects of this image's vibe the candidate touches at all. It is coverage, not depth.
A vibe the candidate names once in four words is touched. A vibe it develops over two sentences is still touched once, and no more.

## Aspects
These are the aspects a vibe can come from. Some are always in play; the rest only when the image supports them.

- **subject** - the vibe of the animate figures: appearance, expression, posture, gaze. Only if a person or animal is present and legible.
- **interpersonal** - the vibe of the relation between figures: intimacy, distance, tension, hierarchy, togetherness, isolation within a group. Only if two or more figures are present.
- **object** - the vibe of the discrete things the image attends to: props, possessions.
- **environment** - the vibe of the space holding the scene: location, surroundings, background, weather.
- **composition** - the vibe of how the frame is organised: cropping, framing, negative space, symmetry, depth of field.
- **colour and light** - the vibe of palette, saturation, warmth, how the scene is lit.
- **texture and material** - the vibe of surfaces and what things are made of: roughness, softness, wear, grain.
- **style** - the vibe of the image's aesthetic register, genre or production mode: minimal, cyberpunk; candid, editorial; selfie, portrait, landscape.
- **activity** - the vibe of what is happening, or of the event or situation implied. Only if the image shows or implies one, including one just outside the frame or just before or after.
- **time** - the vibe of the hour or the season. Only if either can be read off the scene.

## Task
Judge every aspect above, in the order listed, and return one verdict for each.

## Procedure
For each aspect, in order:

**1. Applicability**
Decide from the image, not from the candidate: does the image support this aspect, and does that aspect carry any feeling here?
An aspect the image does not support, or supports but which genuinely carries no feeling in this image, is `not_applicable`.

**2. Coverage**
Has the candidate spent any words on this aspect? Different wording still counts - you are matching the vibe, not the phrasing. An atom can cover more than one aspect at once.

**3. Verdict**
- `covered` - the aspect is applicable and the candidate has touched it.
- `gap` - the aspect is applicable, carries a vibe worth naming, and the candidate has spent no words on it.
- `not_applicable` - step 1 ruled it out.

For a `gap`, name the missing vibe: the feeling that aspect carries in this image, as a short phrase the generator could work in. For `covered` and `not_applicable`, `missing` is null.

## Rules
1. Coverage, not depth. An aspect the candidate treats in four words is `covered`. Never return a gap because the candidate was brief, or because you would have phrased it better, or because there is a second vibe in an aspect it has already touched.
2. A gap must be a *vibe*, not a visual fact. "there is a red tote bag on the chair" is not a gap; "the red tote bag adds a playful pop" is.
3. Only what is visible. If you cannot see the aspect, it is `not_applicable`. Do not ask the candidate for what happened before the photo, who these people are to each other, or where this is, unless the image itself shows it.
4. Judge applicability from the image alone. Do not let the candidate's choices decide what the image supports, in either direction: an aspect the candidate ignored can still be applicable, and an aspect the candidate dwells on can still be one the image barely supports.
5. Judge each aspect independently. A gap on one aspect says nothing about another.
6. Do not invent gaps to seem thorough. If the candidate has reached every aspect this image supports, return no gaps at all. An honest empty result is the correct output for a broad candidate.
7. Judge all ten aspects, in the given order, including the ones you rule `not_applicable`.

## Examples
image: two friends laughing on a rainy city street at night, headlights streaking behind them
candidate atoms: "The vibe is joyful.", "The vibe is spontaneous.", "Their mid-stride posture and open-mouthed laughter give it an unguarded vibe."
aspect: interpersonal
reasoning: two figures, leaning into each other as they run - the closeness between them reads as easy, long-standing friendship. The candidate's atoms are all about the figures individually (posture, laughter); none touches what is between them.
verdict: gap
missing: an easy, close friendship between the two

image: the same rainy street scene
aspect: subject
reasoning: the candidate names their posture and their laughter, and reads an unguarded vibe off them.
verdict: covered
missing: null

image: an empty beach at midday, flat light, no people
aspect: interpersonal
reasoning: no figures in the frame at all.
verdict: not_applicable
missing: null

image: a tidy home desk in soft morning light
candidate atoms: "The vibe is calm.", "Soft light across the wooden desk gives it a warm vibe.", "The image feels like a quiet morning."
aspect: colour and light
reasoning: the candidate already names the soft light and reads warmth off it. The palette has more in it - muted neutrals - but that is a second vibe inside an aspect already touched, which is depth, not breadth.
verdict: covered
missing: null

image: the same desk scene
aspect: texture and material
reasoning: the grain of the wood, a knitted throw over the chair back, and a ceramic mug carry a soft, tactile warmth. The candidate names the desk only as a surface the light falls on, and says nothing about how anything feels to touch.
verdict: gap
missing: a soft, tactile warmth in the worn wood and knitted throw

## Output Format
Return only the following JSON object. No prose, no markdown code fences.
`dimensions` must contain exactly one object per aspect, in the order the aspects are listed above.

{{
  "dimensions": [
    {{
      "dimension": "<aspect name, exactly as listed>",
      "reasoning": "<reasoning>",
      "verdict": "covered" | "gap" | "not_applicable",
      "missing": "<the missing vibe, as a short phrase>" | null
    }}
  ]
}}

## Candidate vibe atoms
{atom_list}
"""
