# h3-prompt-writing (official skill, summarized)

> Source: https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing
> Full guide texts are in `base-en.txt` (T2VA/I2VA/FL2VA/L2VA) and `ref-en.txt` (Ref2VA).

## Purpose

Writes MiniMax H3 video-generation prompts for five modes:
T2VA, I2VA, FL2VA, L2VA (base modes) and Ref2VA (full-reference mode).

## Workflow

1. Identify the input mode from the user's input (text only / text + first frame /
   text + last frame / text + first & last frame / text + omni references).
2. For base modes, read `references/base-en.txt` and follow its final prompt structure:
   - T2VA: no alignment instruction, starts directly with the three core fields.
   - I2VA / FL2VA / L2VA: alignment instruction line first, one blank line, then fields.
3. For Ref2VA, read `references/ref-en.txt` and follow its six-section rewrite format:
   subject_definitions, summary, retention_analysis, detailed_description,
   overall_soundscape, non_diegetic_music.
4. Preserve exact field names, section order, labels, and timing notation from the guides.

## Base modes — three core fields (in order)

1. `integrated_multimodal_description` — visuals, actions, shots, speakers,
   dialogue, singing, diegetic audio along the timeline.
2. `overall_soundscape` — ambient / physical / non-verbal human sounds (1-4 sentences).
3. `non_diegetic_music` — audience-only background music (1-3 sentences, or N/A).

## Ref2VA mode — six sections (in order)

subject_definitions → summary → retention_analysis → detailed_description →
overall_soundscape → non_diegetic_music

Reference labels: `<Subject N>` / `<Picture N>` / `<Video N>` / `<Audio N>`,
kept consistent across all sections.

## Output rules

- Write sections in English; preserve dialogue, lyrics, and visible scene text
  in their original language inside `<d>[lang] ...</d>`.
- Describe each shot by composition, subjects, environment, actions, camera,
  sound, and the exact point where referenced content appears.
- Avoid plot summaries, unresolved reference labels, and timing that does not
  match the requested duration.
