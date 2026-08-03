"""Vibe-representation prompts, keyed by version so results stay comparable across iterations."""

PROMPTS = {
    "baseline": "Describe the vibe of this image.",
    "rich": """
Given an image, your task is to:
  1. Describe the image precisely. include the setting, objects, lighting, colors, and any visible people or details. Be objective but vivid.
  2. Imagine the context behind the photo. Based on visual clues, infer what might be happening in or around the scene. What could the subject or environment suggest about the moment, activity, or atmosphere? This can be speculative, but should remain plausible and stay grounded in the image.
  3. Extract the vibes. Distill the emotional or sensory tone of the scene into one to three short phrase (2-5 words each). For each, explain it in 1-3 sentences in a way that naturally aligns with the description and imagined context, capturing the mood or feeling the scene evokes.
If the image lacks obvious objects or narrative cues, focus on abstract qualities (like texture, temperature, color balance, light) to derive an impression.
""",
}
