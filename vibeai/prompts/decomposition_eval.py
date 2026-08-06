"""Decomposition-quality judge prompt.

Ported from the Notion page "Decomposition evaluation prompt" (under
Projects/VibeAI/Natural language representation/Plausibility experiment).
Scores a decomposition of a vibe representation into vibe atoms along two
axes: Completeness and Atom Quality.
"""

DECOMPOSITION_EVAL_PROMPT = """# Vibe Decomposition Quality Evaluation
You are an expert judge for evaluating the quality of atomic decomposition of vibe representations. 
A vibe representation is a natural language description of the perceived mood, atmosphere, aesthetic character, or experiential feeling evoked by an image, including feelings associated with particular settings, activities, or situations.
An atomic decomposition breaks this description into a set of minimal vibe atoms that are faithful to, and jointly cover, the original vibe representation.

## Task
You are given:
- Original Vibe Representation
- Decomposed Vibe Atoms

### Original Vibe Representation
{representation}

### Decomposed Vibe Atoms
{atoms}

Your task is to evaluate whether the decomposition of a vibe representation into vibe atoms is appropriate for a vibe-oriented decompose-then-verify evaluation pipeline.
For each of the two perspectives below, first give a concise justification for your judgment, then provide your rating.
1. Completeness
2. Atom Quality


## Evaluation Criteria
### Completeness
The set of vibe atoms should preserve the meaningful vibe claims expressed in the original vibe representation. 
vibe claims include:
- mood / atmosphere (cozy, high intensity, etc.)
- aesthetic character (minimalistic, luxury, etc.)
- experiential feeling (golden hour vibe, romantic date night vibe, etc.)

Do not penalize omission of standalone visual observations or visual evidence supporting vibe claims.

**Rubric**
5 — Complete: All vibe claims from the original representation are preserved.
4 — Mostly complete: Nearly all vibe claims are preserved, with only minor omissions.
3 — Partially complete: Several vibe cliams are missing, but the overall meaning is mostly retained.
2 — Mostly incomplete: Many important vibe claims are omitted.
1 — Incomplete: Most of the original affective meaning is lost.

### Atom Quality
Evaluate each vibe atom individually, then aggregate the atom-level judgments to determine the overall Atom Quality score.
Each atom should satisfy all four criteria below.

**1. Affectiveness**
The atom must express a perceived mood, atmosphere, aesthetic character, or experiential feeling.
It should not merely describe objective visual facts.

Good Example: The crowded dance floor creates an energetic atmosphere.
Bad Example: There is a crowded dance floor.

**2. Atomicity**
The atom should express a single independently evaluable affective interpretation. 
Closely related or synonymous vibe descriptors may be grouped together.

Good Example (atomic): The warm lighting creates a cozy atmosphere.
Good Example (closely related vibes): The warm lighting creates a cozy and comforting atmosphere.
Bad Example (not atomic): The warm lighting creates a cozy and romantic atmosphere.

**3. Fidelity**
The atom should faithfully decompose the original representation.
The atom should use the same wording as the original claim whenever possible. It should not paraphrase the claim or introduce additional reasoning.

Good Example
Original: nostalgic country-store vibe.
Atom: The image has nostalgic country-store vibe.

Bad Example
Original: A woman in cropped streetwear studying the car.
Atom (infers new vibe): The style carries an urban, streetwise edge.

Bad Example
Original: Warm sunlight pours through the open sunroof, creating relaxing atmosphere.
Atom (paraphrasing): The sunlight creates a comforting atmosphere.

**4. Evidence preservation**
The atom should preserve visual evidence supporting a vibe, when such evidence is provided explicitly in the original representation.
- If the atom's Fidelity is bad, Evidence Preservation is automatically Bad.
- If the original representation does not explicitly link a vibe to visual evidence, Evidence Preservation is Good.
- If the original representation explicitly links a vibe to visual evidence, the atom should preserve that evidence with the exact same wording.
- If multiple visual cues are explicitly linked to the same vibe, the atom should preserve all of them.

Good Example
Original: The orange lanterns make the street feel nostalgic.
Atom: The orange lanterns make the street feel nostalgic.

Good Example
Original: Soft morning light and a pastel-pink palette make the setup feel cozy.
Atom: Soft morning light and a pastel-pink palette make the cozy feeling.

Bad Example
Original: a woman in cropped streetwear studying the car.
Atom (Fidelity failed): The cropped streetwear carries an urban, streetwise edge.

Bad Example
Original: The orange lanterns make the street feel nostalgic. 
Atom (dropped visual cue): The street feels nostalgic.

Bad Example
Original: Soft morning light and a pastel-pink palette make the setup feel cozy.
Atom (different wording): The lighting and tone make the setup feel cozy.

**Atom Evaluation Procedure**
For each vibe atom:
- Evaluate Affectiveness.
- Evaluate Atomicity.
- Evaluate Fidelity.
- Evaluate Evidence Preservation.
- Determine whether the atom is Good or Bad.
A Good atom satisfies all four criteria. Otherwise, it is Bad. For the aggregation, Compute the proportion of Good atoms.

**Atom quality aggregation**
Compute the proportion of atoms rated "Good," then scale this proportion to a 0–5 score.
Score = (Number of Good atoms / Total number of atoms) × 5


## Output Format
Return only the following JSON. `atomic_judgement` must contain exactly one object per vibe atom, in the same order as the input atoms.

```json
{{
  "atomic_judgement": [
    {{
      "atom": "<atom>",
      "reason": "<concise explanation>",
      "evaluation": {{
		    "affectiveness": true | false,
		    "atomicity": true | false,
		    "fidelity": true | false,
		    "evidence_preservation": true | false
		  }},
      "verdict": "Good" | "Bad"
    }}
  ],
  "final_verdict": {{
    "completeness": {{
      "reason": "<concise explanation>",
      "verdict": <integer 1-5>
    }},
    "atom_quality": {{
      "good_atom_count": <integer>,
      "total_atom_count": <integer>,
      "reason": "<concise explanation>",
      "verdict": <number 0-5, rounded to two decimal places>
    }}
  }}
}}
```
"""
