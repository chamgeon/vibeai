"""Pool-merge prompt: collapse entries that name the same vibe idea.

Branches phrase one feeling many ways ("quiet", "hushed", "tranquil",
"unhurried"), and richness scores a fraction over pool entries - so unmerged,
an idea's weight in the score is how many words the branches happened to use
for it. Merging by idea makes each idea count once.

This is a compromise. It decides synonymy before scoring, without seeing the
representation being scored, so a distinction the merge erases is gone for
every run. The prompt therefore merges only entries that name the same
feeling, and keeps related-but-different feelings apart.
"""

POOL_MERGE_PROMPT = """# Vibe Pool Merge

You are given a pool of vibes extracted from several independent vibe representations of one image.
Because the representations were written separately, the same vibe idea often appears several times in different words ("quiet", "hushed", "tranquil", "quiet wandering rather than urgency").

Your task is to group pool entries that name the **same vibe idea**, and write one representative vibe for each group, so each idea is listed once.

## The membership test

An entry joins a group only if **a representation that said nothing but the group's vibe would already have conveyed the feeling that entry names**. Apply the test to every member, one at a time, against the group's vibe (not against the member next to it).

Judge the feeling, not the wording. None of these make an entry a different feeling:
- scaffolding: "feel", "vibe", "mood", "sense of", "a feeling of"
- intensity: "slightly", "half-", "gentle", "almost", "vividly"
- setting detail that says where or what the feeling is about: "nostalgic drive-in movie night vibe" is the feeling "nostalgic"; "lived-in urban texture" is the feeling "lived-in"

Entries that pass:
- synonyms and near-synonyms: "quiet" / "hushed" / "tranquil" / "peaceful"
- entries built on the same feeling word, always: "cinematic" / "cinematic feel" / "almost cinematic stillness"; "lived-in density" / "vividly lived-in" / "lived-in urban texture"; "cozy" / "modern coziness" (unless the word means something else there - see "quietly energetic" below)
- paraphrases of one idea: "liminal magic hour between day and night" / "a private moment of transition between day and night"

## What stays separate

- **Related but different feelings.** Feelings that often come together in one image are still different feelings if one can be present without the other. Each of these pairs is two groups:
  - "nostalgic" / "melancholic", "nostalgic" / "kitschy"
  - "relaxed" / "content" ("simple pleasures", "low-effort happiness")
  - "communal" / "anticipation", "communal" / "playful"
  - "dreamy" / "magical" / "hushed"
  - "quiet" / "lonely", "intimate" / "cozy", "mysterious" / "eerie"
- **A shared word with a different meaning.** "quietly energetic" is about energy, not quiet - it does not join "quiet".
- **Negations.** "not sterile" never joins "sterile" or "clean".
- **Objects, props and scenes.** An entry that names a thing, an activity or a place rather than a feeling ("treats in hand", "drive-in movie night", "residential Japanese district") joins a group only if it is plainly just another way of saying that group's feeling. Otherwise it stays its own entry - never use one as a reason to widen a group.
- **Umbrella groups.** Do not chain entries into a broad bucket ("calm mood", "a fun night out", "atmosphere") because each one is close to the next. A long list of entries is not a reason to make fewer groups.

## Check before answering

- **Large groups.** Re-run the membership test on every member of any group with more than five members. Large groups are where umbrellas hide.
- **Compound labels.** If a group's vibe needs two feeling words to cover its members ("dreamy twilight hush", "cozy and nostalgic"), it is two or more groups - split it.

When in doubt, keep entries apart. A pool that lists one idea twice is a smaller problem than one that has merged two ideas into one.

## The representative vibe

For each group, write `vibe`: the one vibe that best stands for the whole group.
- It names the feeling, in 1-4 words: "quiet", "lived-in density", "gentle mystery". Not a sentence, not a description of the scene.
- It is as specific as the members allow - covering all of them, but no broader. "quiet" for "quiet" / "hushed" / "tranquil"; not "calm atmosphere".
- It does not add a feeling none of the members express.
- A group of one keeps its entry's meaning; you may shorten scaffolding ("cinematic feel" -> "cinematic"), nothing more.

## Output Format

Return only the following JSON object. No prose, no markdown code fences.
Every input entry must appear exactly once across all `members` lists, copied verbatim.

{{
  "groups": [
    {{
      "vibe": "<representative vibe for the group>",
      "members": ["<entry>", "<entry>"]
    }}
  ]
}}

## Vibe Pool
{pool}
"""
