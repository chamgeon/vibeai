"""The generator's revision prompt: previous representation + evaluator
feedback -> a new representation.

Used for every round of the generator-evaluator workflow after the first (the
first is an ordinary ``prompts/representation.py`` call). Its output is plain
prose in the same shape as ``prompt_v1``'s, so the decomposition step and both
judges run unchanged on a revision.

The prompt deliberately never names the evaluators' rules. The generator is
told what failed and what is missing, not that plausibility is gated at 90% or
that the aspects come from a fixed list of ten - a generator that knew the
rubric would optimise against the judge instead of looking harder at the image.
"""

REFINE_PROMPT = """# Image Vibe Representation - Revision

You are an expert at describing the vibe of an image: the mood, atmosphere, aesthetic character, and experiential feeling it evokes, in natural language.

You wrote the vibe representation below. It was checked against the image and did not pass. Your task is to revise it.

## What a vibe representation is
Flowing natural language - a short paragraph or a few connected sentences - weaving together two kinds of statements:

1. **Evidence-backed statements**: claims that name specific visual evidence and link it to a vibe.
2. **Vibe-only statements**: standalone claims about the feeling of the image, without stated visual evidence.

It is not a description of the image. Every visual cue you mention is there to carry a vibe. The vibe must be grounded in the image whether or not it is explicitly evidenced, and any evidence you state must be a main driver of the vibe, not a secondary or incidental one. Do not speculate about facts that are not visually evident.

## Your previous representation
{representation}

## What the check found
{feedback}

## How to revise
Work from the previous representation. Do not start over: everything the check did not object to was accepted, and should survive into the revision largely as written.

**For each claim that failed**: look at the image again and decide which of these is true.
- The vibe is there but you attached it to the wrong cue → keep the vibe, name the cue that actually produces it.
- The cue is there but the vibe is not, or the image works against it → drop the claim. Do not argue with the finding or restate it in safer words.
- You asserted something not visible in the image → drop it.

**For each vibe reported missing**: work it in, if you can see it yourself. A few words naming it is enough. Do not elaborate an aspect you have already touched - that buys nothing. Anything you add has to hold up to the same scrutiny as the rest: an aspect you cannot actually see in the image is not one to reach for, and a wider representation that overclaims is worse than a narrow one, not better.

Keep the revision about the length of the previous representation. Refinement is a trade: what comes out is depth, what goes in is breadth.

The revision stands alone as the representation. It must not refer to the previous version, to the check, or to this revision process, and must not read as a list of fixes worked through in order.

## Output Format
Return only the revised vibe description as natural language prose. No headers, preamble, lists."""
