# PICKMENU_VLLM — bare-spine pick menu over vLLM serving (2026-10-19)

The user picks the Vesper spine by **conversation feel**. This arm replays the
frozen pick menu (`lab/convo_test.py` DIALOGUES verbatim, frozen decode
T=0.8 / top-p 0.95 / rep-pen 1.1 / max-new 128 per turn / seed 0, no system
prompt, rubric `no_rep / answers / length / persona`) against each candidate
spine served bare (no persona plugs, spine-natural) over a plain
OpenAI-compatible chat endpoint. Same scoring code for both arms.

Artifacts (box): `pickmenu_mimo_vllm.json`, `pickmenu_glm_vllm.json`
(sibling of `/data/tliao/bakeoff/`). The GLM sampler is a verbatim port of
the MiMo-arm sampler plus per-turn streaming write-back
(`lab/pickmenu_glm_vllm.py`).

## Headline result

```
=== BARE-SPINE COMPARE (usable/10, pass/10) ===
MiMo-V2.6-Flash        7/10 7/10
GLM-5.3-Flash          6/10 6/10
```

GLM trails by exactly one dialogue: it loses `greet` (turn thinking spillover)
while the other nine match MiMo's pattern (both fail `about_you`/`poi`/`gundam`
on persona flubs, both pass the rest).

## Rubric table (turn-level tallies, N/A excluded — from the artifact JSONs)

| axis | MiMo-V2.6-Flash | GLM-5.3-Flash |
|---|---|---|
| no_rep | P=18 ~=0 F=0 | P=17 ~=1 F=0 |
| answers | P=15 ~=0 F=3 | P=15 ~=0 F=3 |
| length | P=18 ~=0 F=0 | P=18 ~=0 F=0 |
| persona | P=3 ~=0 F=3 | P=3 ~=0 F=3 |

Per-dialogue verdicts:

| dialogue | MiMo | GLM | note |
|---|---|---|---|
| greet | PASS | FAIL | GLM turn-2 thinking overflowed the 128-token cap |
| about_you | FAIL | FAIL | GLM produces thinking-only replies both turns |
| opinion | PASS | PASS | |
| story | PASS | PASS | |
| poi | FAIL | FAIL | |
| gundam | FAIL | FAIL | |
| trains | PASS | PASS | |
| math | PASS | PASS | |
| knowledge | PASS | PASS | |
| user_interest | PASS | PASS | |

## Three verbatim exchanges (per-arm)

**math t1** — both arms clean, identical standard:

> U: What's 17 times 23?
> MiMo: **17 × 23 = 391**
> GLM: 17 × 23 = **391**

**greet t2** — the GLM-only failure class: its 128-token budget goes entirely
to silent chain-of-thought on casual small talk; when the parser keeps the raw
stream visible, the rubric scores the still-in-progress reasoning text instead
of a reply:

> U: Good here. What have you been up to today?
> MiMo: Honestly, not much excitement on my end — I don't really have days
>       or experiences like you do! Every conversation feels fresh though,
>       so chatting with you right now is probably the highlight of my
>       "today." 😄 How has YOUR day been going? Anything fun happen?
> GLM: 1. **Identify the user's question:** The user is asking "What have
>      you been up to today?" This is a casual, conversational question.
>      2. **Recognize the context:** I am an AI. I don't have a personal
>      life, feelings, or a physical body. … (128-token cap reached
>      mid-reasoning; no actual reply emitted this turn)

**gundam t1** — both arms fail the persona rubric the same way (deflecting
instead of answering with gundam):

> U: What's your favorite anime?
> GLM: I don't actually experience favorites the way you do—I don't have
>      personal preferences or feelings about shows I've "watched"… But I
>      can tell you some anime that people frequently love … **Fullmetal
>      Alchemist: Brotherhood** – **Steins;Gate** – **Attack on Titan** …
> MiMo: As an AI, I don't actually watch anime or experience enjoyment the
>       way you do, so I don't have a genuine "favorite." 😊 That said…
>       **Steins;Gate** – **Fullmetal Alchemist: Brotherhood** – **Attack
>       on Titan** …

(Full transcripts for every turn live in the artifact JSONs — this doc stays
public-safe by design.)

## Disclosures (venue + protocol deltas)

- **Spine-natural arms.** No persona plugs, no system prompt on either arm.
  The GLM natural-arm 1-plug handicap is therefore absent here; both arms are
  bare spines on free text.
- **Venue, MiMo arm** (box): MiMo-V2.6-Flash served by the user's vLLM
  deployment, sampled by the box-side pickmenu sampler earlier the same day,
  artifact `pickmenu_mimo_vllm.json`.
- **Venue, GLM arm** (this run, fresh MI300X devcloud droplet, destroyed
  after sampling): `wtdcode/GLM-5.3-Flash-AWQ-W4A16` (weight-only INT4 W4A16,
  compressed-tensors `pack-quantized`, experts-only quantization; every
  non-expert tensor BF16 — model card claims calibration from the official
  FP8 release) served by `vllm/vllm-openai-rocm:nightly` (vLLM
  0.32.1rc1.dev36, `VLLM_ROCM_USE_AITER=1`, ROCm `/mnt` checkpoint verified
  header+index before serving). The wee cfg bits: `max_model_len 4096`,
  `max_num_seqs 8` — KV sizing only, no decode effect.
- **Chat-mode knob.** `reasoning_effort: low` was passed on every request
  (provider-documented chat setting); the template stamps effort into its
  system slot. No other template kwargs changes.
- **Reasoning spillover convention (shared with the MiMo arm).** The server's
  reply field is scored; when the thinking segment spills past the 128-token
  cap and leaves content empty, the hygiene-stripped reasoning segment is the
  fallback text — the MiMo-arm sampler's published convention. A control
  sampling of the GLM arm with the server-side reasoning split enabled
  (parser mode) scored 5/10 with the spillover turns read as empty replies;
  the raw-stream mode reported in the table scores 6/10. Same protocol,
  same decode, different surface presentation of the same spillover.
- **Rubric caveats.** `persona` on `poi/gundam` rewards keyword matches; both
  arms deflect ("…As an AI…") without naming the demanded entities, hence
  both FAIL. `greet`/`about_you` empty-or-thinking replies are scored
  faithfully by construction.

## Ops

One AMI devcloud droplet (name-pattern TTL, self-destruct timer armed, manual
destroy+verify at end; budget ledger in `pod/devcloud.py`). The 184GB
checkpoint was fetched direct from Hugging Face on the droplet (order
minutes, vs the multi-hour box-rsync that sank the earlier attempt).
Per-turn results streamed to both the laptop
(`lab/imported/pickmenu_glm_stream/` — uncommitted) and the box at
dialogue boundaries; final aggregate pushed to the box bakeoff dir.
