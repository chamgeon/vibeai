"""Decomposition-quality judge prompt.

Ported from the Notion page "Decomposition evaluation prompt" (under
Projects/VibeAI/Natural language representation/Plausibility experiment).
Scores a decomposition of a vibe representation into vibe atoms along three
axes: Completeness, Claim Independence, and Atom Quality.
"""

DECOMPOSITION_EVAL_PROMPT = """# Vibe Decomposition Quality Evaluation
You are an expert judge for evaluating the quality of atomic decomposition of vibe representations. A vibe representation is a natural language description of the atmosphere, mood, emotion, or aesthetic impression conveyed by an image. An atomic decomposition breaks this description into a set of vibe atoms that are faithful to, and jointly cover, the original vibe representation.

## Task
You are given:
- Original Vibe Representation
- Decomposed Vibe Atoms

### Original Vibe Representation
{representation}

### Decomposed Vibe Atoms
{atoms}

Your task is to evaluate whether the decomposition of a vibe representation into vibe atoms is appropriate for a vibe-oriented decompose-then-verify evaluation pipeline.
For each of the three perspectives below, first give a concise justification for your judgment, then provide your rating.
1. Completeness
2. Claim Independence
3. Atom Quality


## Evaluation Criteria
### Completeness
The set of vibe atoms should preserve all **affective interpretations** expressed in the original vibe representation. Do not penalize omission of standalone visual observations that aren't themselves vibe interpretations.

**Rubric**
5 — Complete: All affective interpretations from the original representation are preserved.
4 — Mostly complete: Nearly all affective interpretations are preserved, with only minor omissions.
3 — Partially complete: Several affective interpretations are missing, but the overall meaning is mostly retained.
2 — Mostly incomplete: Many important affective interpretations are omitted.
1 — Incomplete: Most of the original affective meaning is lost.

### Claim Independence
Each vibe atom should express a semantically distinct, independently evaluable interpretation, with no substantial overlap or duplication between atoms.

**Rubric**
5 — Fully independent: Every atom expresses a distinct affective interpretation with no redundancy.
4 — Mostly independent: Minor overlap exists but atoms remain largely distinct.
3 — Moderately independent: Noticeable overlap exists among several atoms.
2 — Poor independence: Many atoms substantially overlap or repeat similar interpretations.
1 — Not independent: Atoms are highly redundant or cannot be evaluated independently.

### Atom Quality
Evaluate each vibe atom individually, then aggregate the atom-level judgments to determine the overall Atom Quality score.
Each atom should satisfy all four criteria below.

**1. Affectiveness**
The atom must express a perceived atmosphere, mood, emotion, or aesthetic interpretation.
It should not merely describe objective visual facts.

Good Example: The crowded dance floor creates an energetic atmosphere.
Bad Example: There is a crowded dance floor.

**2. Atomicity**
The atom should express exactly one affective interpretation.

Good Example: The warm lighting creates a cozy atmosphere.
Bad Example: The warm lighting creates a cozy and nostalgic atmosphere.

**3. Evidence preservation**
If the original representation explicitly supports a vibe interpretation using visual evidence, that evidence should remain attached to the corresponding vibe atom.

Good Example
Original: The orange lanterns make the street feel nostalgic.
Atom: The orange lanterns make the street feel nostalgic.

Bad Example
Original: The orange lanterns make the street feel nostalgic.
Atom: The street feels nostalgic.

**4. Faithfulness**
The atom must not introduce, remove, or alter the meaning of the original representation.

Good Example
Original: The dim lighting creates a calm atmosphere.
Atom: The dim lighting creates a calm atmosphere.

Bad Example
Original: The dim lighting creates a calm atmosphere.
Atom: The dim lighting creates a mysterious atmosphere.

**Atom Evaluation Procedure**
For each vibe atom:
- Evaluate Affectiveness.
- Evaluate Atomicity.
- Evaluate Evidence Preservation.
- Evaluate Faithfulness.
- Determine whether the atom is Good or Bad.
A Good atom satisfies all four criteria. Otherwise, it is Bad. For the aggregation, Compute the proportion of Good atoms.

**Atom quality aggregation**
Compute the proportion of atoms rated "Good," then scale this proportion to a 0-5 score.
Score = (Number of Good atoms / Total number of atoms) x 5


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
        "evidence_preservation": true | false,
        "faithfulness": true | false
      }},
      "verdict": "Good" | "Bad"
    }}
  ],
  "final_verdict": {{
    "completeness": {{
      "reason": "<concise explanation>",
      "verdict": <integer 1-5>
    }},
    "claim_independence": {{
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
