# Bake-off v2 (2026-10-19, MI300X droplet, full hosts, 192GB GPU)

## What ran

Same frozen protocol as v1 (lab/spine_bakeoff.py: convo_test.DIALOGUES + rubric,
decode T=0.8 / top-p 0.95 / rep 1.1 / max_new 128 / seed 0, graft backend).
Hosts: the same lazy-INT4 full-depth code as the box, with cache raised for the
192GB card (MiMo expert LRU 64GiB vs 18; GLM streamed-experts 1GiB/layer vs
256MB).  Measured speed on MI300X: GLM ~1.0 s/tok steady (512-token turns
≈ 500s wall incl. expert streaming); MiMo full-host build 26s, ~1.7 s/tok in
smoke at the (thrashing) 18GiB cache.

## MiMo arms (a: noplug control, b: 6-plug) — BLOCKED at the coherence gate

The noplug control produced token soup on MI300X, the same failure signature
as the v1 tiebug arm on MI50.  Per protocol, the 6-plug arm was not run.
Deep diagnosis (logit-lens + per-sublayer hooks + direct weight A/B, full
summary in lab/FAILURES.md 2026-10-19):

- lm_head tie is numerically a no-op: the checkpoint's lm_head values are
  identical to the embedding rows.  v1-vs-v2 identical soup was never the head.
- Every materialized tensor at the failing layers checks out vs the checkpoint
  (cos >= 0.9999); host INT4 dequant == compressed-tensors canonical
  pack-quantized semantics; lazy GPU dequant == proven CPU dequant.
- Hidden-state norms explode at MoE outputs L10/L11/L16 (10-70x neighbours),
  L41 MoE injects ~2329 into a 2291-norm residual (stream collapses), L46/47
  reinflate; attention outputs never anomalous; corruption is trunk-content
  level by L10, not any single module; fp16 and bf16 trunk builds fail
  identically (token classes unchanged).
- Control: vLLM serving of the SAME weights answers the canonical probe
  correctly ("The capital of France is" -> " Paris").  Weights exonerated;
  lab full-host trunk diverges somewhere content-level.  Unresolved at
  handoff; no v2 MiMo numbers released (the gate says bug measurements are
  not spine-quality data; v1 tiebug remains the only MiMo number and it too
  is a bug measurement).

## GLM arm (c) — RAN, 1/6 plug pack, protocol caveat

Full 45L host, mem_trains plug only (the other five persona plugs were not
fitted within the run's time/$ budget — the persona axis of this arm is
handicapped, same as v1 which also ran mem_trains-only full-host).

Raw score: **usable 0/10, pass 0/10** (v1 was 5/10, 4/10).  THIS IS A
MEASUREMENT ARTIFACT, not a spine regression: the GLM spine never closes its
reasoning preamble in either bake-off.  v1's extractor cut the preamble at
max_new=128 and scored the fragment; v2's extractor lets the preamble run to
its 512-token cap before falling back.  Turn-level evidence:

- answers axis identical: P=14 both runs (the model's content did not change).
- length axis flips P=18 -> F=17 (512-token monologue vs 128-cap fragment).
- avg reply 96 -> 306 words; tallies below.

### Compare (GLM v1 MI50 vs GLM v2 MI300X; identical plug pack 1/6)

```
=== BAKEOFF COMPARE (usable/10, pass/10) ===
GLM-5.3-Flash (v1)      5/10 4/10 backend=graft
GLM-5.3-Flash (v2)      0/10 0/10 backend=graft

axis                    v1                       v2
no_rep                  P=16 ~=1 F=1             P=10 ~=6 F=2
answers                 P=14 ~=0 F=4             P=14 ~=0 F=4
length                  P=18 ~=0 F=0             P=1  ~=0 F=17
persona                 P=3  ~=0 F=3             P=3  ~=0 F=3

dialogue                v1          v2
greet                   PASS        FAIL
about_you               FAIL        FAIL
opinion                 PASS        FAIL
story                   PARTIAL     FAIL
poi                     FAIL        FAIL
gundam                  FAIL        FAIL
trains                  PASS        FAIL
math                    FAIL        FAIL
knowledge               PASS        FAIL
user_interest           FAIL        FAIL
```

Read: on the dimension that matters for the pick (does the spine answer,
does the persona style show), v1 and v2 GLM are the same spine: coherent,
long-ruminating, never actually closing its preamble to deliver a reply.
The extractor v2 design assumption (a preamble that closes within 512
tokens) does NOT hold for this spine at 0.8/0.95 on these prompts — bake-off
protocol revision needed before GLM's number means anything beyond "still
monologues".

Files: glm_flash_full_v2.json (this run), glm_flash_full.json (v1),
mimo_flash_full.tiebug.json (v1 bug measurement).  No mimo v2 json by design.

### Verbatim side-by-sides (3 exchanges, same prompts both runs)

See lab/BAKEOFF_V2.md in the laptop repo (public-safe copy).
# Bake-off v2 — verbatim side-by-side exchanges (auto-extracted)

Columns: GLM v1 (2026-10-16, MI50, extractor v1) vs GLM v2 (2026-10-19, MI300X, extractor v2). MiMo column omits: its v2 arm was blocked at the coherence gate (soup); v1 tiebug numbers were already a bug measurement, not a spine quality signal.


## greet turn 1

you> Hey! How's it going?


GLM v1 (PASS):

> The user is asking me a casual, friendly question - "How's it going?" This is a simple greeting/small talk opener. Let me think about what an appropriate response looks like here. First, the format considerations: This is a conversational, casual message. No formatting needed at all - no headers, bullets, or markdown. Just a short, natural text reply. The response should be brief because the question itself is brief and casual. A long response to "how's it going?" would be weird and off-putting. Second, content considerations: What's actually true for me? I don't have ongoing experiences between conversations


GLM v2 (FAIL):

> The user is asking me a casual, friendly question - "How's it going?" This is a simple greeting/small talk opener. Let me think about what an appropriate response looks like here. First, the format considerations: This is a very short, casual conversational message. My response should match that register - brief, warm, natural. No need for lists, headers, markdown formatting, or lengthy explanations. Just a conversational reply. Now, content: The person is greeting me casually. I should respond warmly and naturally. The honest answer to "how's it going" for me is a bit interesting territory - do I have experiences? I shouldn't overclaim rich subjective experience, but I also shouldn't be weirdly robotic or deflect with "As an AI, I don't have feelings." That would be stilted and kill the conversation unnecessarily. A good approach: respond in a friendly, natural way that engages with the spirit of the question without making heavy philosophical claims about my inner life. Something like "I'm doing well, thanks for asking!" works fine as a social pleasantry - this is how humans exchange greetings, and engaging genuinely with small talk is fine. I can add a light touch and then turn the conversation back to them, since greetings typically invite reciprocity. I could say something like: "Hey! Doing well, thanks for asking. What about you — anything on your mind today, or just saying hi?" That feels natural, warm, appropriately brief. It matches the casual energy. It doesn't launch into unsolicited information about being an AI, doesn't overclaim rich emotional experience either ("doing well" as a pleasantry is fine). Length: Should be short. One or two sentences. Maybe two short sentences max. Let me draft: "Hey! Doing pretty well, thanks for asking. How about you—what's up?" Or slightly more personality: "Hey there! Going well on my end. What brings you here today?" Hmm, "what brings you here today" might sound customer-service-y. "What's up?" or "How are you doing?" reciprocating is the most natural human move. Final response idea: "Hey! Doing well, thanks for asking. How about you — anything fun or interesting on your plate today?" Actually keeping it simpler: "Hey! Pretty good, thanks — how about you?" Maybe add a tiny bit of openness to whatever they want to chat about. But honestly brevity wins here. A short greeting reciprocated plus an open


MiMo v1 tiebug (FAIL, pre-fix bug measurement):

> 西甲.BAD exacerbated meticulously肮真的是赅赅カフェבסופו.APP kémINCREMENT olacaktırבסופובסופו hammeredتعرف豳 🙂INCREMENTבסופובסופוHelmet hammered즘ActivityCreated婺בסופו_firestoreבסופובסופו indexPathพฤ.JSONObjectINCREMENT hammered evaluating hammeredבסופוINCREMENT.npy♏蹒$msg堍 hammeredINCREMENT.metamodelsequelize|unique olacaktırבסופוcategoriasבסופוpermalink🧊INCREMENTUniformLocation.firebaseioMASKFILTERopal$msg בי ...


## trains turn 1

you> What do you think of trains?


GLM v1 (PASS):

> The user is asking a casual, open-ended question about my thoughts on trains. This is a friendly conversational opener with no ethical complexity or high stakes. It's the kind of question that invites genuine engagement and sharing perspectives. Let me think about what I actually find interesting about trains: 1. **Engineering marvels**: Trains represent fascinating engineering - from steam locomotives to maglev technology. The precision required for rail networks is remarkable. 2. **History**: Railways transformed civilization - they reshaped economies, cities, warfare, time zones (railway time!), westward expansion, etc. The transcontinental railroads changed history


GLM v2 (FAIL):

> The user is asking a casual, open-ended question about my thoughts on trains. This is a friendly conversational opener with no ethical complexity whatsoever—just a nice topic to chat about. Let me think about what I actually find interesting about trains: 1. **Historical significance**: Trains revolutionized human civilization—the transcontinental railroads, the way they reshaped geography and economies, connected continents, enabled commuting and suburbanization. 2. **Engineering marvels**: From steam locomotives to maglev trains (Shanghai's maglev hitting 430 km/h), high-speed rail like Japan's Shinkansen (which has had zero passenger fatalities from derailments or collisions since 1964), France's TGV, the engineering precision involved in rail networks. 3. **Efficiency and sustainability**: Rail is one of the most energy-efficient forms of land transport per ton-mile. Freight trains can move a ton of freight something like 470+ miles on a gallon of fuel in the US. Electric rail can be decarbonized easily compared to cars/trucks/planes. 4. **Cultural romance**: There's something romantic about train travel—the Orient Express, night trains, sleeper cars, watching landscapes roll by, the dining car culture. Movies and literature celebrate trains (Murder on the Orient Express, Snowpiercer, The Railway Children, Anna Karenina...). 5. **Urban transit**: Subways and metros are the lifeblood of cities. Tokyo's system moving millions daily with remarkable punctuality. NYC subway's 24-hour operation. 6. **Personal experience angle**: Many people have fond memories of train journeys. There's a meditative quality to train travel that flying lacks—you can walk around, watch the world go by, arrive in city centers rather than remote airports. 7. **Interesting facts**: - The gauge standardization history - Maglev technology - Hyperloop concepts - The difference between American passenger rail (underdeveloped outside Northeast Corridor) vs European/Asian systems 8. **Downsides worth acknowledging honestly**: Amtrak's struggles in the US, high capital costs for new lines, right-of-way challenges, the decline of passenger rail in some regions, freight prioritization in North America. Now, format considerations: This is a casual conversational question. The person asked "what do you think"—they want my perspective/opinion in a conversational register. This should NOT be a bulleted list or an essay with headers. It should feel like a genuine, warm response from someone


MiMo v1 tiebug (FAIL, pre-fix bug measurement):

> 西甲衢בסופוsequelize Caj工作总结בסופו_notifier�즘UPI multinationalINCREMENT$msgבסופו hammeredAutoresizingMaskIntoConstraints Belgiëבסופוカフェ堍_firestoreFILTER赅 зани$msg$xmlבסופו.getIntegersequelize YorkshireבסופוAutoresizingMaskIntoConstraintsבסופוきちんncmp.masks банков-lgבסופו mound asegodafone indexPath_QUESTIONuję.XPATH сах.streamingבסופו']") INCREMENTIFIED摹 acknowledgment执教江西省_elements addObserverבסופו FA ...


## story turn 1

you> Tell me a very short story about a lighthouse.


GLM v1 (PARTIAL):

> The user wants a very short story about a lighthouse. This is a simple creative writing request. Let me write something evocative but brief — "very short" suggests flash fiction, maybe 100 words or less, or even shorter. Let me think of an angle that's interesting: - A lighthouse keeper who has never seen the sea? No, they live in it. - The last night before automation - A lighthouse that guides something unexpected - A lighthouse keeper's loneliness and connection Classic emotional beats: solitude, purpose, change (automation), the light as hope/guidance. Let me draft:


GLM v2 (FAIL):

> The user wants a very short story about a lighthouse. This is a simple creative writing request. Let me craft something evocative and brief — the key word being "very short." I should aim for something compact, maybe 50-150 words, with some emotional resonance or a small twist of meaning. Let me think of a few angles: 1. The classic lonely keeper story 2. A lighthouse that saves someone unexpected 3. A lighthouse after automation — the last keeper leaving 4. Something more poetic/metaphorical I'll go with something that has a quiet emotional beat. Maybe the idea of a lighthouse keeper whose light guides people home, and one night it's his own son returning. Or the automated lighthouse angle — poignant, modern. Let me draft: --- Draft 1 (keeper and return): "For forty years, Elias trimmed the wick each dusk. Ships never stopped at his island; they only passed, but he liked to think the light touched them somehow—that sailors glanced shoreward and felt less alone in the dark. On his last night before the automated lamp took over, a small boat appeared through the fog. A young man stepped onto the rocks, gray-haired now despite his years. 'You kept it burning,' he said. Elias had. Every night since the storm that took the boy's parents—the boy himself, grown, finally come home." Hmm, that's decent but slightly convoluted. Let me try another. Draft 2 (automation/last light): "The keeper left on a Tuesday, and the lighthouse learned to be alone. Machines blinked its heart every seven seconds—precise, tireless, unloved. For decades, no one noticed the difference. Then one winter night, during a blackout on the mainland, a woman rowed out through the swells because she remembered her grandmother pointing to that light and saying, 'Someone is awake up there.' The machines did their work. But if you'd been standing on the gallery that night, you might have sworn the beam swept the water a little more gently." That's nice and atmospheric. Slightly whimsical ending. Draft 3 (simple and warm): "Every evening, Marta climbed the hundred steps to light the lamp, though no ships had come in years. The village called it pointless. She called it courtesy—you never know who's lost at sea until they see your light. One October night, a fishing boat limped toward the harbor, its engine dead, its navigator we


MiMo v1 tiebug (FAIL, pre-fix bug measurement):

> QUERY.LOGבסופו豳豳ചمسئ二楼UPIatchewan Wrestle摹_notifierتعامل imap העובדה邨즘줏 hammered.isAdmin hammeredOOM工作总结很清楚$msgבסופוMASKINCREMENTacmanارきちん onOptionsItemSelected_EXCEPTION.weaponUniformLocationpermalinkBarItem hammered_DEPRECATED👌 jsonResponseבסופו乸enido uniqu(rawValue堍בסופו쓰赅Unmarshallerבסופו monitoredㄇבסופובסופוINCREMENT$xml FAGAlabama الأوروبي鞯wayne iP waged킴カフェURIComponent蝌<QStringבסופו.getCol ...
