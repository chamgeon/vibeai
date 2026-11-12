RICHNESS_COMPARISON_PROMPT = """# Vibe Richness Comparison
You are comparing two vibe sets written for the same image, and deciding which one is richer.
A vibe is a short phrase naming a mood, atmosphere, aesthetic character, or experiential feeling the image evokes. Each set was extracted from a different vibe representation of the image. You will see the image, Set A, and Set B.

## Richness
A vibe set is rich when it captures more of the image's vibe: more of the distinct affective facets the image actually carries, and each of them with more specificity and nuance.

Ask of each set: if someone who had never seen the image read only these vibes, how much of the image's feeling would they get - how many of its different sides, and how precisely?

What makes a set richer:
- **Range**: it reaches facets of the image's vibe the other set does not touch (e.g. the other set only conveys the calm, while this one also conveys the loneliness and the nostalgia the image carries).
- **Specificity**: where both sets touch the same facet, it names that facet more precisely (e.g. "the hush of a city just after rain" over "quiet").

## What you must ignore
1. **Plausibility.** Treat every vibe in both sets as true of the image. Do not reward or penalise a set for whether you agree with its vibes. Another judge checks that separately.
2. **Count and length.** A set is not richer because it lists more vibes or uses longer phrases. Vibes that say the same thing in different words count once. Three vibes covering three distinct facets beat eight near-synonyms covering one.
3. **Position and labels.** The order of the sets and their names "A" and "B" carry no information.

## Procedure
1. Look at the image and list the distinct facets of its vibe: the different sides of the feeling it carries.
2. For each set, say which of those facets it captures, which it misses, and how specifically it captures each one. Merge near-synonyms within a set before you count anything.
3. Compare the two sets on range first, then on specificity, and decide which is richer overall.

Answer "tie" only when neither set captures the image's vibe more fully or more precisely than the other - not as a way to avoid a close call. If one set is even slightly richer, pick it.

## Output Format
Return only the following JSON object. No prose, no markdown code fences.

{{
  "image_facets": ["<distinct facet of the image's vibe>"],
  "set_a_analysis": "<which facets Set A captures, which it misses, how specifically>",
  "set_b_analysis": "<which facets Set B captures, which it misses, how specifically>",
  "comparison": "<range first, then specificity>",
  "winner": "A" | "B" | "tie"
}}

## Set A
{vibes_a}

## Set B
{vibes_b}
"""
