"""Decomposition prompts: break a vibe representation into atomic vibe claims."""

PROMPTS = {
    "baseline": """Decompose the following vibe representation into a list of atomic claims. \
Each claim should express a single affective interpretation (mood, atmosphere, or emotional tone), \
in a short self-contained sentence.

Return only a JSON array of strings, with no other text.

Vibe Representation:
{representation}
""",
    "v1": """# Vibe Decomposition
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
}
