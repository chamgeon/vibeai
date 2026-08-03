"""Decomposition prompts: break a vibe representation into atomic vibe claims."""

PROMPTS = {
    "baseline": """Decompose the following vibe representation into a list of atomic claims. \
Each claim should express a single affective interpretation (mood, atmosphere, or emotional tone), \
in a short self-contained sentence.

Return only a JSON array of strings, with no other text.

Vibe Representation:
{representation}
""",
}
