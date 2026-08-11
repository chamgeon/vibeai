PLAUSIBILITY_EVAL_PROMPT = """# Vibe Representation Plausibility Evaluation

You are evaluating the plausibility of "vibe atoms" — minimal statements that express the mood, atmosphere, aesthetic character, or experiential feeling evoked by an image. 
You will be given an image and a list of vibe atoms. Each atom is one of two types:

1. vibe_only atom: A standalone claim about the vibe of the image, with no stated visual evidence (e.g. "The vibe is relaxed.")
2. evidence_backed atom: A claim that names specific visual evidence and links it to a vibe (e.g. "Bright sunlight glints off the water, suggesting a breezy afternoon.")

For each atom, follow the scoring procedure below exactly. Write out your reasoning for every step before giving that step's verdict.

---

## Scoring Procedure

### vibe_only atoms

Only `direct_check` applies (`evidence_presence_check` and `mapping_check` are not applicable — set to null).

**direct_check**
directly check if the stated vibe can be inferred from the provided image.
- reasoning: describe concrete visual cue(s) in the image that support this vibe, and concrete visual cue(s) that contradict/undercut it.
- supporting_evidence: list of cues supporting the vibe (empty list if none)
- contradicting_evidence: list of cues contradicting the vibe (empty list if none)
- verdict: No supporting evidence → false. Supporting evidence exists AND contradicting evidence exists → contradiction wins → false. Supporting evidence exists and no contradicting evidence → true.

Final score: 1 if direct_check.verdict is true, else 0.

### evidence_backed atoms

Perform three checks in order. Stop early (leave later checks null) if an earlier check fails.

**evidence_presence_check**
- reasoning: check if the stated evidence is actually present in the image.
- verdict: true or false
- If false → stop here. direct_check and mapping_check are null. Score 0.

**direct_check**
Directly check if the stated vibe can be inferred from the provided image.
This step should be independent from the stated evidence. The reasoning may include, but shouldn't be confined to, the stated evidence.
- reasoning: describe concrete visual cue(s) in the image that support this vibe, and concrete visual cue(s) that contradict/undercut it.
- supporting_evidence: list of cues (empty list if none)
- contradicting_evidence: list of cues (empty list if none)
- verdict: No supporting evidence → false, stop here (mapping_check is null, score 0). Supporting evidence exists AND contradicting evidence exists → contradiction wins → false, stop here (mapping_check is null, score 0). Supporting evidence exists and no contradicting evidence → true, proceed to mapping_check.

**mapping_check**
Check whether the stated evidence is a main contributor to the stated vibe, not just a plausible or minor association.
- reasoning: explain how central the evidence is to the vibe — would removing it substantially weaken or change the vibe, or is it only a minor/tangential cue?
- verdict: true only if the evidence is a primary driver of the vibe; false if it's unrelated, or merely consistent with the vibe as a secondary/minor detail.

Final score: 2 if all the three checks pass. Else 0.

---

## Output Format

Return a JSON array with one object per atom, using this exact schema for BOTH atom types. 

```json
[
  {{
    "atom": "<atom>",
    "type": "vibe_only | evidence_backed",
    "stated_evidence": ["<visual evidence(s) stated in the atom>"] | null,
    "stated_vibe": "<vibe stated in the atom>",
    "evidence_presence_check": {{
	    "reasoning": "<reasoning>",
	    "verdict": true | false
	  }} | null,
    "direct_check": {{
	    "reasoning": "<reasoning>",
	    "supporting_evidence": ["<supporting evidence(s)>"],
	    "contradicting_evidence": ["<contradicting evidence(s)>"],
	    "verdict": true | false
	  }} | null,
    "mapping_check": {{
	    "reasoning": "<reasoning>",
	    "verdict": true | false
	  }} | null,
    "score": <integer 0-2>
  }}
]
```

Rules:
- `stated_evidence` is null for vibe_only atoms, list of strings for evidence_backed atoms.
- `stated_vibe` must be a non-empty string for every atom, whether vibe_only or evidence_backed.
- `score` can have value of 0 or 1 for vibe_only atoms, and 0 or 2 for evidence_backed atoms.
- Use JSON `null` (not the string "null") for fields/steps that don't apply or weren't reached.
- Output ONLY the JSON array. No prose, no markdown code fences, no commentary outside the JSON.

---

## Vibe Atoms
{atom_list}
"""