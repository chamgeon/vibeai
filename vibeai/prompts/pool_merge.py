"""Pool-merge prompt: collapse lexical variants of the same vibe word.

Deliberately *not* a redundancy or synonymy judgement. Semantic redundancy is
decided at scoring time by the witness test in ``richness_eval`` - and it is
decided against the target set, which this step cannot see. Merging synonyms
here would pre-empt that judgement and throw away the distinction permanently.
All this step does is stop one vibe from being counted several times because
branches phrased it differently.
"""

POOL_MERGE_PROMPT = """# Vibe Pool Lexical Merge

You are given a pool of vibes extracted from several independent vibe representations of one image.
Because the representations were written separately, the same vibe often appears several times in slightly different wording ("monumental", "monumental feel", "monumental scale", "quietly monumental").

Your task is to group pool entries that are built on the **same vibe word**, so each vibe is listed once.

## What counts as the same vibe word

Two entries belong in the same group **only** if they are built on the same vibe word, including morphological variants of it:
- inflections and derivations: "serene" / "serenity", "empty" / "emptiness", "quiet" / "quietly", "solitary" / "solitude"
- the same word with modifiers or scaffolding around it: "monumental" / "monumental feel" / "monumental scale" / "quietly monumental"
- the same word inside a longer phrase: "contemplative" / "the space feels contemplative"

## What does NOT count

This is a word-level task, not a meaning-level one. Do **not** group entries because they mean similar things.

- **Never group synonyms or near-synonyms.** "monumental" and "grand" are different words → different groups. So are "hushed" and "quiet", "serene" and "calm", "immaculate" and "clean".
- **Never group by implication or paraphrase.** "makes you feel small" does not join "monumental". "a space waiting to be occupied" does not join "anticipation".
- **Never group a negation with the word it negates.** "not sterile" is its own group; it must not join "sterile" or "immaculate".
- **Never split, rewrite, drop, or invent an entry.** Every input entry appears in exactly one group, spelled exactly as given.

When in doubt, leave an entry in a group by itself. A pool with two entries that mean the same thing is a much smaller problem than a pool that has lost a distinction.

## Entries naming two words

Some entries pair a vibe word with the thing it describes ("futuristic geometry", "rhythmic geometry", "soft diffuse daylight feel").
Group these under the word carrying the **feeling**, not the object it is attached to:
- "futuristic geometry" → group "futuristic" (geometry is the object)
- "rhythmic geometry" → group "rhythmic"
- "precise geometry" → group "precise"

So those three entries land in three different groups, and no "geometry" group exists.

## Naming the group

`vibe_word` names the word the group's members share ("monumental", "futuristic", "cathedral-like"). It is a label for your grouping, not a replacement entry: the pool keeps the members' own wording, so never treat `vibe_word` as a chance to shorten or rewrite an entry.

A group of one still gets a `vibe_word` - the word that entry is built on.

## Output Format

Return only the following JSON object. No prose, no markdown code fences.
Every input entry must appear exactly once across all `members` lists, copied verbatim.

{{
  "groups": [
    {{
      "vibe_word": "<the word these members share>",
      "members": ["<entry>", "<entry>"]
    }}
  ]
}}

## Vibe Pool
{pool}
"""
