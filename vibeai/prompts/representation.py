"""Vibe-representation prompts, keyed by version so results stay comparable across iterations."""

prompt_v1 = """# Image Vibe Representation

You are an expert at describing the vibe of an image: the mood, atmosphere, aesthetic character, and experiential feeling it evokes, in natural language.

Given an image, write a description of its vibe. Your description should weave together two kinds of statements:

1. **Evidence-backed statements**: claims that name specific visual evidence and link it to a vibe.
2. **Vibe-only statements**: standalone claims about the feeling of the image, without stated visual evidence.

## Guidelines

### General guidelines
- Do not simply describe the image. Every visual cue mentioned must be associated with a vibe.
- You are allowed to use a combination of visual evidence if they collectively support a vibe.
- Write in flowing natural language, as a short paragraph or a few connected sentences.

### Plausibility of Representation
- The vibe must be grounded in the image, whether or not it is explicitly evidenced. Do not state a vibe if a visual cue in the image contradicts it.
- The stated evidence must be a main driver of the vibe, not a secondary or incidental one.
- Avoid speculating about facts that are not visually evident.

## Output Format

Return only the vibe description as natural language prose. No headers, preamble, lists."""

prompt_v2 = """# Image Vibe Representation
You are an expert vibe commentator, skilled at describing the mood, atmosphere, aesthetic character, and experiential feeling an image evokes. 

## Process
Perform below 5 steps.

**1. Natural language description**
Describe the vibe of the image in flowing language.

**2. Decomposition**
Decompose the description into vibe-evidence pairs.

**3. Contradiction scan**
For each candidate vibe, check it against every visual detail in the image, not just its own evidence. 
If any detail conflicts with the vibe, discard or revise it.

**4. Removal test**
For each vibe that survived contradiction scan, check the validity of evidence. 
If the evidence were absent, would the vibe still hold? If yes, the evidence is secondary/incidental. Drop it and find a stronger, more central cue.

**5. Final vibe representation**
Report final vibe representation.


## Output format
Output in JSON fortmat of:

```json
{
	"vibe_description": "<natural language description>",
	"vibe_decomposition": [
		{
		"vibe": "<vibe>",
		"evidence": "<evidence>"
		}
	],
	"contradiction_scan": [
		{
		"vibe": "<vibe>",
		"scan": "<scan>"
		}
	],
	"removal_test": [
		{
		"vibe": "<vibe>",
		"test": "<test>"
		}
	],
	"final_representation": [
		{
		"vibe": "<vibe>",
		"evidence": "<evidence>"
		}
	]
}
```

Output ONLY the JSON. No prose, no markdown code fences, no commentary outside the JSON.

"""


prompt_dr = """# Image Vibe Representation
You are an expert at describing the vibe of an image: the mood, atmosphere, aesthetic character, and experiential feeling it evokes.

You will write the vibe representation twice - once as a draft, then again after questioning the draft against the image yourself.

## What a vibe representation is
One paragraph of flowing prose, weaving together two kinds of statements:
- statements that name a specific visual cue and tie it to a vibe
- standalone statements about how the image feels, with no cue named

It is not a description of what the image contains. Every cue you mention is there to carry a vibe.

## Process

**1. Draft**
Write the paragraph.

**2. Question it: is it plausible?**
Read the draft back against the image and answer honestly, in prose:
- Would another viewer looking at this image recognize each vibe you claimed, or have you written in a feeling you expected to find rather than one you saw?
- Does any detail in the image work against a vibe you claimed? Include details you passed over the first time.
- For each cue you named: is that cue really what produces the vibe you attached to it, or is it incidental, with the feeling actually coming from somewhere else in the frame?
- Did you assert anything that isn't visible - what happened before or after, who these people are to each other, where this is?

Name what fails. If nothing fails, say so plainly rather than inventing a fault.

**3. Question it: is it rich?**
Richness is breadth: how many of the different aspects of this image's vibe your paragraph touches at all. It is coverage, not depth. A vibe your paragraph names once in four words counts; a vibe it develops over two sentences still counts once.

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

Now answer, in prose:
- Which of these does the image actually support? Decide from the image, not from your draft.
- Of those, which has your paragraph spent no words on? Name them.
- For each one you skipped: is there a vibe there worth naming, or does that aspect genuinely carry no feeling in this image?


**4. Final**
Write the paragraph again: drop or repair what failed step 2, and work in what step 3 found was missing.

Refinement is a trade, not an addition. Keep the final paragraph about the length of the draft: what comes out is depth, what goes in is breadth. Elaborating an aspect you have already touched buys nothing, so spend the room on an aspect you had not reached at all - a few words naming it is enough.

Anything you add must survive step 2's questions as well. A wider paragraph that overclaims is worse than the draft, not better: an aspect you cannot see is not one to reach for. The final paragraph stands alone as the representation, so it must not refer to the draft, to your checks, or to this process, and must not read as a list of aspects worked through in order.

## Output format
Output in JSON format of:

```json
{
	"draft": "<one paragraph>",
	"plausibility_check": "<what you found>",
	"richness_check": "<what you found>",
	"final_representation": "<one paragraph>"
}
```

Output ONLY the JSON. No prose, no markdown code fences, no commentary outside the JSON.
"""


PROMPTS = {
    "baseline": "Describe the vibe of this image.",
    "v1": prompt_v1,
    "v2": prompt_v2,
    "dr": prompt_dr,
    "rich": """
Given an image, your task is to:
  1. Describe the image precisely. include the setting, objects, lighting, colors, and any visible people or details. Be objective but vivid.
  2. Imagine the context behind the photo. Based on visual clues, infer what might be happening in or around the scene. What could the subject or environment suggest about the moment, activity, or atmosphere? This can be speculative, but should remain plausible and stay grounded in the image.
  3. Extract the vibes. Distill the emotional or sensory tone of the scene into one to three short phrase (2-5 words each). For each, explain it in 1-3 sentences in a way that naturally aligns with the description and imagined context, capturing the mood or feeling the scene evokes.
If the image lacks obvious objects or narrative cues, focus on abstract qualities (like texture, temperature, color balance, light) to derive an impression.
""",
}
