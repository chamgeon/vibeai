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


PROMPTS = {
    "baseline": "Describe the vibe of this image.",
    "v1": prompt_v1,
    "rich": """
Given an image, your task is to:
  1. Describe the image precisely. include the setting, objects, lighting, colors, and any visible people or details. Be objective but vivid.
  2. Imagine the context behind the photo. Based on visual clues, infer what might be happening in or around the scene. What could the subject or environment suggest about the moment, activity, or atmosphere? This can be speculative, but should remain plausible and stay grounded in the image.
  3. Extract the vibes. Distill the emotional or sensory tone of the scene into one to three short phrase (2-5 words each). For each, explain it in 1-3 sentences in a way that naturally aligns with the description and imagined context, capturing the mood or feeling the scene evokes.
If the image lacks obvious objects or narrative cues, focus on abstract qualities (like texture, temperature, color balance, light) to derive an impression.
""",
}
