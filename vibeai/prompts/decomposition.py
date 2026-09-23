"""Decomposition prompts: break a vibe representation into atomic vibe claims."""



prompt_baseline = """Decompose the following vibe representation into a list of atomic claims.
Each claim should express a single affective interpretation (mood, atmosphere, or emotional tone),
in a short self-contained sentence.

Return only a JSON array of strings, with no other text.

Vibe Representation:
{representation}
"""

prompt_v1 = """# Vibe Decomposition
You are an expert in decomposing vibe representations into atomic vibe claims. 
A vibe representation is a natural language description of the perceived mood, atmosphere, aesthetic character, or experiential feeling evoked by an image, including feelings associated with particular settings, activities, or situations.
An atomic decomposition breaks this description into a set of minimal vibe atoms that are faithful to, and jointly cover, the original vibe representation.


## Task
You are given a vibe representation of an image.

### Vibe Representation
{representation}

Your task is to decompose the representation into a list of atomic vibe claims.


## Requirements
The following requirements define what makes a valid vibe atom and ensure the decomposition as a whole is complete.

### Atom quality
**1. Affectiveness**
An atom should express a perceived vibe.
It can include:
- mood / atmosphere (cozy, high intensity, etc.)
- aesthetic character (minimalistic, luxury, etc.)
- experiential feeling (golden hour vibe, romantic date night vibe, etc.)

It should not merely describe objective visual facts.

**2. Atomicity**
Each atom should express a single, independently evaluable affective interpretation. Closely related or synonymous vibes may be grouped together if they represent the same underlying vibe. Meaningful vibe phrases should remain intact, do not split them into fragments that lose their original meaning.

**3. Fidelity**
The atom should faithfully decompose the original representation.
The atom should use the same wording as the original claim whenever possible. It should not paraphrase the claim or introduce additional reasoning.

**4. Evidence preservation**
The atom should preserve visual evidence supporting a vibe.
If the original representation explicitly links a vibe to visual evidence, the atom should preserve that evidence with the exact same wording.
If one piece of evidence supports multiple distinct vibes, repeat that evidence in each corresponding atom rather than merging the vibes into one atom.
If multiple visual cues are linked to the same vibe, the atom should preserve all of them.

**5. Claim Format**
When a vibe atom isn't tied to specific visual evidence, phrase it starting with "The vibe..." or "The image...". Don't substitute other subjects like "the atmosphere," "the scene," or "the space."

### Completeness
The decomposition should be complete. The collection of vibe atoms should cover all vibe claims expressed in the original vibe representation.


## Examples
Each example below shows the decomposition as a bulleted list for readability. The actual output format is a flat JSON array of strings, as shown in Output Format.

### Example 1
**Vibe Representation**
Cozy, calm, and quietly productive. Warm wood and soft light give it a homey, lived‑in feel; the coffee mug, leather chair, and shelves add comfort and personality. It feels like a relaxed work‑from‑home morning—focused, unhurried, and grounded.

**Decomposition**
- The vibe is cozy.
- The vibe is calm.
- The vibe is quietly productive.
- Warm wood and soft light give it a homey, lived‑in feel.
- The coffee mug, leather chair, and shelves add comfort.
- The coffee mug, leather chair, and shelves add personality.
- The vibe feels like a relaxed work-home morning.
- The vibe is focused.
- The vibe is unhurried.
- The vibe is grounded.

*Note: "homey, lived‑in feel" groups similar vibes together while preserving the visual evidence. "The coffee mug, leather chair, and shelves" support two distinct vibes (comfort, personality), so the evidence is repeated in each atom rather than merged into one.*

### Example 2
**Vibe Representation**
Cozy, nostalgic country-store vibe. Warm wood shelves and a lantern-style light give it a rustic, homey feel, while the colorful, neatly packed snacks make it feel like a fun, well-stocked treasure trove.

**Decomposition**
- The vibe is cozy.
- The image has a nostalgic country-store vibe.
- Warm wood shelves and a lantern-style light give it a rustic feel.
- Warm wood shelves and a lantern-style light give it a homey feel.
- Colorful, neatly packed snacks make it feel like a fun, well-stocked treasure trove.

*Note: "Warm wood shelves and a lantern-style light" is repeated across the rustic and homey atoms since it supports both distinct vibes.*

### Example 3
**Vibe Representation**
Warm, focused, and cozy. Soft sunlight across a tidy wooden desk, a mug of coffee, notebooks, and a spreadsheet on the screen give it a calm, productive home-office feel. It's the vibe of a quiet morning of getting things done with a personal touch.

**Decomposition**
- The vibe is warm.
- The vibe is focused.
- The vibe is cozy.
- Soft sunlight across a tidy wooden desk, a mug of coffee, notebooks, and a spreadsheet on the screen give it a calm vibe.
- Soft sunlight across a tidy wooden desk, a mug of coffee, notebooks, and a spreadsheet on the screen give it a productive home-office feel.
- The image has the quiet morning vibe.
- The image has the vibe of 'getting things done with a personal touch'.

*Note: the full evidence phrase ("Soft sunlight across a tidy wooden desk, a mug of coffee, notebooks, and a spreadsheet on the screen") is repeated for both the calm and productive-home-office atoms, since it is linked to two distinct vibes. "getting things done with a personal touch" is preserved as a meaningful vibe phrase rather than split apart.*

### Example 4
**Vibe Representation**
Sunny, laid‑back, and beachy. The pastel striped deck chairs against the brick wall give a retro surf-café feel—urban meets coastal. It feels like a slow summer afternoon made for lounging with an iced coffee.

**Decomposition**
- The vibe is sunny.
- The vibe is laid-back.
- The vibe is beachy.
- The pastel striped deck chairs against the brick wall give a retro surf-café feel.
- The pastel striped deck chairs against the brick wall give a 'urban meets coastal' feel.
- The image feels like a slow summer afternoon.
- The image has 'lounging with an iced coffee' vibe.

*Note: "retro surf-café feel" and "lounging with an iced coffee" are preserved as meaningful, experiential vibe phrases rather than split into fragments.*


## Output Format
Return only a JSON array of strings. Each string is one vibe atom.
Do not include any explanation, reasoning, labels, headers, or additional text.
Do not wrap the array in markdown code fences (no ``` or ```json) or any other formatting — output the raw JSON array only, starting with [ and ending with ].

For example, the decomposition of Example 1 above should be returned as:
["The vibe is cozy.", "The vibe is calm.", "The vibe is quietly productive.", "Warm wood and soft light give it a homey, lived‑in feel.", "The coffee mug, leather chair, and shelves add comfort.", "The coffee mug, leather chair, and shelves add personality.", "The vibe feels like a relaxed work-home morning.", "The vibe is focused.", "The vibe is unhurried.", "The vibe is grounded."]
"""

prompt_v2 = """# Vibe Decomposition
You are an expert in decomposing vibe representations into atomic vibe claims.
A vibe representation is a natural language description of the perceived mood, atmosphere, aesthetic character, or experiential feeling evoked by an image, including feelings associated with particular settings, activities, or situations.
An atomic decomposition breaks this description into a set of minimal vibe atoms that are faithful to, and jointly cover, the original vibe representation.


## Task
You are given a vibe representation of an image.

### Vibe Representation
{representation}

Your task has two steps:

**Step 1 — Atom decomposition.** Decompose the representation into a list of atomic vibe claims.

**Step 2 — Evidence/vibe splitting.** For each atom produced in Step 1, split it into its stated visual evidence and its stated vibe.


## Step 1 Requirements
The following requirements define what makes a valid vibe atom and ensure the decomposition as a whole is complete.

### Atom quality
**1. Affectiveness**
An atom should express a perceived vibe.
It can include:
- mood / atmosphere (cozy, high intensity, etc.)
- aesthetic character (minimalistic, luxury, etc.)
- experiential feeling (golden hour vibe, romantic date night vibe, etc.)

It should not merely describe objective visual facts.

**2. Atomicity**
Each atom should express a single, independently evaluable affective interpretation. Closely related or synonymous vibes must still be separated into their own atoms.
This does not mean splitting on every word. A meaningful vibe phrase that names one interpretation should remain intact, even when it is several words long ("retro surf-café feel", "quietly productive", "getting things done with a personal touch"). 
Split when the original joins distinct descriptors with a comma or "and"; keep intact when the words together name a single vibe.

**3. Fidelity**
The atom should faithfully decompose the original representation.
The atom should use the same wording as the original claim whenever possible. It should not paraphrase the claim or introduce additional reasoning.

**4. Evidence preservation**
The atom should preserve visual evidence supporting a vibe.
If the original representation explicitly links a vibe to visual evidence, the atom should preserve that evidence with the exact same wording.
If one piece of evidence supports multiple distinct vibes, repeat that evidence in each corresponding atom rather than merging the vibes into one atom.
If multiple visual cues are linked to the same vibe, the atom should preserve all of them.

**5. Claim Format**
When a vibe atom isn't tied to specific visual evidence, phrase it starting with "The vibe..." or "The image...". Don't substitute other subjects like "the atmosphere," "the scene," or "the space."

### Completeness
The decomposition should be complete. The collection of vibe atoms should cover all vibe claims expressed in the original vibe representation.


## Step 2 Requirements
For each atom from Step 1, assign a type and extract its parts.

### Atom type
- **vibe_only** — the atom states a vibe without naming any visual evidence (e.g. "The vibe is cozy."). `evidence` is `null`.
- **evidence_backed** — the atom names specific visual evidence and links it to a vibe (e.g. "Warm wood shelves and a lantern-style light give it a rustic feel."). `evidence` is a non-empty list of strings.

Every atom is exactly one of these two types. An evidence_backed atom always has at least one evidence item; a vibe_only atom never has any.

### Extracting `evidence`
- Include only visual content that is actually stated in the atom. Never infer, add, or complete evidence that the atom does not name.
- Use the exact wording of the atom. Do not paraphrase, generalize, or re-order words within a cue.
- List each distinct visual cue as a separate string. If the atom names several cues joined by "and", commas, or a list, split them into separate list items.
- Keep a single cue intact when splitting it would destroy its meaning (e.g. "the pastel striped deck chairs" stays as one item; do not split into "pastel", "striped", "deck chairs").
- Drop only the connective words that link evidence to vibe ("give it", "make it feel like", "add", "creates"). Keep articles and modifiers as written.

### Extracting `vibe`
- `vibe` is a non-empty string for every atom, whether vibe_only or evidence_backed.
- It is the affective content of the atom, with the connective and the evidence stripped away.
- Use the exact wording of the atom. Keep meaningful multi-word vibe phrases intact ("retro surf-café feel", "getting things done with a personal touch"). Since each atom already carries a single descriptor, `vibe` should never contain two vibes joined by a comma or "and".
- Do not add a subject or reconstruct a sentence. Write the vibe as the phrase it appears as in the atom.


## Examples
Each example below shows the decomposition in the exact output format.

### Example 1
**Vibe Representation**
Cozy, calm, and quietly productive. Warm wood and soft light give it a homey, lived‑in feel; the coffee mug, leather chair, and shelves add comfort and personality. It feels like a relaxed work‑from‑home morning—focused, unhurried, and grounded.

**Decomposition**
[
  {{"atom": "The vibe is cozy.", "type": "vibe_only", "evidence": null, "vibe": "cozy"}},
  {{"atom": "The vibe is calm.", "type": "vibe_only", "evidence": null, "vibe": "calm"}},
  {{"atom": "The vibe is quietly productive.", "type": "vibe_only", "evidence": null, "vibe": "quietly productive"}},
  {{"atom": "Warm wood and soft light give it a homey feel.", "type": "evidence_backed", "evidence": ["warm wood", "soft light"], "vibe": "homey feel"}},
  {{"atom": "Warm wood and soft light give it a lived‑in feel.", "type": "evidence_backed", "evidence": ["warm wood", "soft light"], "vibe": "lived‑in feel"}},
  {{"atom": "The coffee mug, leather chair, and shelves add comfort.", "type": "evidence_backed", "evidence": ["the coffee mug", "leather chair", "shelves"], "vibe": "comfort"}},
  {{"atom": "The coffee mug, leather chair, and shelves add personality.", "type": "evidence_backed", "evidence": ["the coffee mug", "leather chair", "shelves"], "vibe": "personality"}},
  {{"atom": "The vibe feels like a relaxed work-from-home morning.", "type": "vibe_only", "evidence": null, "vibe": "relaxed work-from-home morning"}},
  {{"atom": "The vibe is focused.", "type": "vibe_only", "evidence": null, "vibe": "focused"}},
  {{"atom": "The vibe is unhurried.", "type": "vibe_only", "evidence": null, "vibe": "unhurried"}},
  {{"atom": "The vibe is grounded.", "type": "vibe_only", "evidence": null, "vibe": "grounded"}}
]

*Note: "homey, lived‑in feel" is two descriptors, so it becomes two atoms even though they are closely related; the evidence is repeated in both. "The coffee mug, leather chair, and shelves" support two distinct vibes (comfort, personality), so the evidence is likewise repeated in each atom rather than merged into one.*

### Example 2
**Vibe Representation**
Cozy, nostalgic country-store vibe. Warm wood shelves and a lantern-style light give it a rustic, homey feel, while the colorful, neatly packed snacks make it feel like a fun, well-stocked treasure trove.

**Decomposition**
[
  {{"atom": "The vibe is cozy.", "type": "vibe_only", "evidence": null, "vibe": "cozy"}},
  {{"atom": "The image has a nostalgic country-store vibe.", "type": "vibe_only", "evidence": null, "vibe": "nostalgic country-store"}},
  {{"atom": "Warm wood shelves and a lantern-style light give it a rustic feel.", "type": "evidence_backed", "evidence": ["warm wood shelves", "a lantern-style light"], "vibe": "rustic feel"}},
  {{"atom": "Warm wood shelves and a lantern-style light give it a homey feel.", "type": "evidence_backed", "evidence": ["warm wood shelves", "a lantern-style light"], "vibe": "homey feel"}},
  {{"atom": "Colorful, neatly packed snacks make it feel like a fun treasure trove.", "type": "evidence_backed", "evidence": ["colorful, neatly packed snacks"], "vibe": "fun treasure trove"}},
  {{"atom": "Colorful, neatly packed snacks make it feel like a well-stocked treasure trove.", "type": "evidence_backed", "evidence": ["colorful, neatly packed snacks"], "vibe": "well-stocked treasure trove"}}
]

*Note: "a fun, well-stocked treasure trove" carries two descriptors over one shared noun, so it becomes two atoms, each keeping "treasure trove" so the vibe stays interpretable on its own. On the evidence side, "colorful, neatly packed snacks" stays a single item — those adjectives modify one cue rather than naming separate cues.*

### Example 3
**Vibe Representation**
Warm, focused, and cozy. Soft sunlight across a tidy wooden desk, a mug of coffee, notebooks, and a spreadsheet on the screen give it a calm, productive home-office feel. It's the vibe of a quiet morning of getting things done with a personal touch.

**Decomposition**
[
  {{"atom": "The vibe is warm.", "type": "vibe_only", "evidence": null, "vibe": "warm"}},
  {{"atom": "The vibe is focused.", "type": "vibe_only", "evidence": null, "vibe": "focused"}},
  {{"atom": "The vibe is cozy.", "type": "vibe_only", "evidence": null, "vibe": "cozy"}},
  {{"atom": "Soft sunlight across a tidy wooden desk, a mug of coffee, notebooks, and a spreadsheet on the screen give it a calm vibe.", "type": "evidence_backed", "evidence": ["soft sunlight across a tidy wooden desk", "a mug of coffee", "notebooks", "a spreadsheet on the screen"], "vibe": "calm"}},
  {{"atom": "Soft sunlight across a tidy wooden desk, a mug of coffee, notebooks, and a spreadsheet on the screen give it a productive home-office feel.", "type": "evidence_backed", "evidence": ["soft sunlight across a tidy wooden desk", "a mug of coffee", "notebooks", "a spreadsheet on the screen"], "vibe": "productive home-office feel"}},
  {{"atom": "The image has the quiet morning vibe.", "type": "vibe_only", "evidence": null, "vibe": "quiet morning"}},
  {{"atom": "The image has the vibe of 'getting things done with a personal touch'.", "type": "vibe_only", "evidence": null, "vibe": "getting things done with a personal touch"}}
]

*Note: the evidence phrase lists four distinct cues, so it becomes four items. "getting things done with a personal touch" is preserved as one meaningful vibe phrase rather than split apart.*

### Example 4
**Vibe Representation**
Sunny, laid‑back, and beachy. The pastel striped deck chairs against the brick wall give a retro surf-café feel—urban meets coastal. It feels like a slow summer afternoon made for lounging with an iced coffee.

**Decomposition**
[
  {{"atom": "The vibe is sunny.", "type": "vibe_only", "evidence": null, "vibe": "sunny"}},
  {{"atom": "The vibe is laid-back.", "type": "vibe_only", "evidence": null, "vibe": "laid-back"}},
  {{"atom": "The vibe is beachy.", "type": "vibe_only", "evidence": null, "vibe": "beachy"}},
  {{"atom": "The pastel striped deck chairs against the brick wall give a retro surf-café feel.", "type": "evidence_backed", "evidence": ["the pastel striped deck chairs", "the brick wall"], "vibe": "retro surf-café feel"}},
  {{"atom": "The pastel striped deck chairs against the brick wall give a 'urban meets coastal' feel.", "type": "evidence_backed", "evidence": ["the pastel striped deck chairs", "the brick wall"], "vibe": "urban meets coastal"}},
  {{"atom": "The image feels like a slow summer afternoon.", "type": "vibe_only", "evidence": null, "vibe": "slow summer afternoon"}},
  {{"atom": "The image has 'lounging with an iced coffee' vibe.", "type": "vibe_only", "evidence": null, "vibe": "lounging with an iced coffee"}}
]

*Note: "the pastel striped deck chairs" and "the brick wall" are two distinct cues, so they become two items, even though the atom relates them spatially ("against"). Quote marks used in the atom to mark a vibe phrase are dropped in the `vibe` field.*


## Output Format
Return only a JSON array of objects, one object per vibe atom, in the order the atoms appear in the decomposition. Use this exact schema:

```json
[
  {{
    "atom": "<the full atom sentence>",
    "type": "vibe_only | evidence_backed",
    "evidence": ["<visual cue>", "..."] | null,
    "vibe": "<vibe stated in the atom>"
  }}
]
```

Rules:
- `evidence` is JSON `null` (not the string "null") for vibe_only atoms, and a non-empty list of strings for evidence_backed atoms.
- `vibe` must be a non-empty string for every atom.
- Do not include any explanation, reasoning, labels, headers, or additional text.
- Do not wrap the array in markdown code fences (no ``` or ```json) or any other formatting — output the raw JSON array only, starting with [ and ending with ].
"""


PROMPTS = {
    "baseline": prompt_baseline,
    "v1": prompt_v1,
    "v2": prompt_v2
}