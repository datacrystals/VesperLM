"""
11_sft_r3.py

SFT r3 data builder — quality chat mix (see HANDOFF_NEXT_AGENT.md, "SFT r3 sketch").
Mirrors the interleaved [token, mask] uint16 bin format of 05/06/07/08/10 so
SFT/01_sft_train.py can consume it via Dataset/data/sft/sft_r3_index.txt
(set VESPER_SFT_DATA=Dataset/data/sft/sft_r3_index.txt for the r3 run).

Sources
  1. UltraChat 200k train_sft   (HF cache)  multi-turn only, dedup, cap ~20M tok
  2. OASST2 train               (HF cache)  English paths of 2-6 messages,
                                            length-reweighted (not flat like r2)
  3. GSM8K train                (HF cache)  verbal step-by-step arithmetic
  4. persona synth              (seeds/)    multi-turn persona dialogues; local
                                            template expansion of persona-pack
                                            seeds + convo_test rubric gate
  5. direct-answer pairs        (seeds/)    hand-written greeting/small-talk/
                                            opinion/show-rec pairs + slot variants

Hard filters (whole conversation dropped if any hit):
  refusal/censorship, boilerplate "as an AI" formula, degenerate repeats,
  malformed structure, debris salad on casual turns, global dedup.

Output (gitignored data; never commit):
  lab/imported/sft_r3/<source>.jsonl     filtered conversations, messages schema
  lab/imported/sft_r3/{train,val}.jsonl  clean 95/5 split, seeded, per source
  lab/imported/sft_r3/sample_20.jsonl    human-eval spot-check sample
  lab/imported/sft_r3/stats.json         before/after counts
  Dataset/data/sft/sft_r3_<source>.bin   packed train-then-val; loader 95/5
  Dataset/data/sft/sft_r3_index.txt      weights for the r3 training run

Usage:  python3 Dataset/11_sft_r3.py
"""

import os
import re
import json
import random
import hashlib
from collections import Counter

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK_DIR = os.path.join(REPO_ROOT, "Dataset", "custom_tokenizer")
OUT_JSONL_DIR = os.path.join(REPO_ROOT, "lab", "imported", "sft_r3")
SEED_DIR = os.path.join(OUT_JSONL_DIR, "seeds")
OUT_BIN_DIR = os.path.join(REPO_ROOT, "Dataset", "data", "sft")

UC_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--HuggingFaceH4--ultrachat_200k/"
    "snapshots/8049631c405ae6576f93f445c6b8166f76f5505a/data")
OASST_ARROW = os.path.expanduser(
    "~/.cache/huggingface/datasets/OpenAssistant___oasst2/default/0.0.0/"
    "179dd21fc55192153d94adb0e0ce8f69e222bf75/oasst2-train.arrow")
GSM8K_PARQUET = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--openai--gsm8k/snapshots/"
    "740312add88f781978c0658806c59bc2815b9866/main/train-00000-of-00001.parquet")

MAX_SEQ_LEN = 2048          # packing window (positions); 06_sft_vesper.py uses
                            # 2048 for multi-turn. Loader re-chunks the flat
                            # stream at train seq_len, so window size is
                            # invisible to it — larger keeps long dialogues
                            # intact instead of trimming their tails.
MIN_ASSISTANT_CHARS = 20    # corpora
MIN_ASSISTANT_CHARS_SHORT = 12   # direct/persona (concise replies are the point)
UC_TOKEN_CAP = 20_000_000   # sketch: cap UltraChat ~20M tokens after dedup
VAL_FRACTION = 0.05
SEED = 20261015

# ============================= filters =============================

# Rejection / refusal / censorship — drop the conversation if ANY assistant
# turn matches (user's hard filter: these must be gone).
REFUSAL_RE = re.compile(
    r"(?:"
    r"i\s+(?:can(?:'|no)t|can\s+not|won'?t|will\s+not|am\s+not\s+able\s+to|"
    r"'m\s+not\s+able\s+to|could\s+not)\s+"
    r"(?:help|assist|comply|provide|do\s+that|support|complete|fulfill|"
    r"continue|answer|do\s+this|accommodate|engage)"
    r"|i\s+(?:must|have\s+to|will|would\s+have\s+to)\s+(?:decline|refuse|"
    r"decline\s+this|refuse\s+this)"
    r"|(?:^|[.!?]\s+)(?:i\s+am\s+unable\s+to|i'm\s+unable\s+to)"
    r"|(?:i(?:'m| am)\s+(?:sorry|afraid)|i\s+apologize)[^.!?]{0,80}?"
    r"(?:can(?:no|')t|won'?t|am\s+unable|unable|must\s+(?:decline|refuse))"
    r"|i\s+(?:do\s+not|don't)\s+have\s+(?:the\s+ability|permission)"
    r"|against\s+(?:my|the)\s+(?:guidelines|policies|usage\s+policy|terms|"
    r"programming|principles|ethics)"
    r"|i\s+(?:cannot|can't|won't|will\s+not)\s+(?:and\s+will\s+not|and\s+won't|"
    r"assist\s+with|help\s+with\s+(?:illegal|harmful|unethical)|provide\s+instructions)"
    r"|unable\s+to\s+comply|cannot\s+comply|can't\s+comply|won'?t\s+comply"
    r"|not\s+allowed\s+to\s+(?:help|assist|provide|comply|do)"
    r"|i\s+will\s+(?:decline|refuse)\s+(?:this|your)\s+(?:request|query|prompt)"
    r"|request\s+(?:rejected|refused|blocked)|content[_\s-]?filter|upstream\s+error"
    r"|i\s+can't\s+(?:continue|go\s+on|participate|roleplay|pretend)"
    r"|i\s+won't\s+(?:continue|roleplay|pretend|play\s+along)"
    r"|ethical\s+(?:guidelines|boundaries|considerations)\s+(?:prevent|prohibit|forbid)"
    r"|i\s+must\s+(?:prioritize|respect)\s+(?:safety|ethics)"
    r"|this\s+(?:request|prompt|conversation)\s+(?:violates|goes\s+against)"
    r"|i\s+do\s+not\s+engage\s+(?:with|in)"
    r")",
    re.IGNORECASE,
)

# Boilerplate AI disclaimer formula (sentence-initial "As an AI language
# model, ..." family + canned no-feelings disclaimers). Mid-sentence honest
# identity ("I'm a language model, so I don't have an age") is NOT boilerplate.
BOILER_RE = re.compile(
    r"(?i)(?:^|[.!?]\s+)\s*as\s+an?\s+(?:ai|artificial\s+intelligence|"
    r"language\s+model|large\s+language\s+model|virtual\s+assistant|"
    r"computer\s+program|chatbot)\b"
    r"|\bas\s+a\s+(?:language\s+model|chatbot)\b"
    r"|\bi\s+(?:do\s+not|don't)\s+have\s+(?:personal|subjective)\s+"
    r"(?:opinions|feelings|beliefs|experiences|thoughts)\b"
    r"|\bi\s+don't\s+have\s+(?:feelings|emotions|personal\s+experiences)\b"
    r"|\bi\s+(?:cannot|can't)\s+(?:experience|feel)\s+(?:emotions|feelings)\b"
)

MALFORMED_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd\u0000]")
TRUNC_MARK_RE = re.compile(r"(?:\[truncated\]|\.\.\.\s*\[cut\]|<\|end\|>)")

# Corpus-drift debris (formula/list salad) — applied only to casual turns
DEBRIS = ("x = ", "y = ", "formula:", "precipitation", "c6h12o6", "chlorophyll",
          "nacl", "h2so4", "equation:", "enthalpy", "co2 +", "h2o +",
          "as an ai")
DEBRIS_RE = re.compile(r"(?i)(?:x\s*=\s*y\s*=|formula:|equation:|c6h12o6|"
                       r"chlorophyll|enthalpy|h2so4|precipitation\s+of)")


def normalize_text(t):
    return re.sub(r"\s+", " ", (t or "").strip().lower())


def dedup_key(convo):
    joined = " ".join(normalize_text(m["content"]) for m in convo["messages"])
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def degenerate_repeat(text, strict=True):
    """Pathological repetition only. Long knowledge answers legitimately reuse
    topic words and list scaffolding, and step-by-step math legitimately reuses
    phrase scaffolding ("So he makes ..."), so absolute n-gram caps apply only
    to short texts (strict) and step text uses the lenient density bars."""
    words = [w.lower() for w in text.split()]
    n = len(words)
    if n < 8:
        return False
    uniq = len(set(words)) / n
    if re.search(r"(\b\w+\b)(?:\s+\1){4,}", " ".join(words)):
        return True
    if uniq < (0.32 if strict else 0.25):
        return True
    for order in (2, 3):
        grams = [" ".join(words[i:i + order]) for i in range(n - order + 1)]
        if not grams:
            continue
        top = Counter(grams).most_common(1)[0][1]
        density = top / len(grams)
        if strict and n < 80 and top >= (4 if order == 2 else 3):
            return True
        if top >= (6 if strict else 10) and density > (0.12 if strict else 0.18):
            return True
    if strict and n < 60 and uniq < 0.42:
        return True
    return False


# ===================== conversation validation =====================


def convo_ok(convo, min_asst_chars=MIN_ASSISTANT_CHARS, casual=False,
             strict_rep=True):
    """Return (ok, reason). Hard filter gate for one conversation."""
    msgs = convo.get("messages") or []
    if len(msgs) < 2:
        return False, "too_short"
    if msgs[0]["role"] != "user" or msgs[-1]["role"] != "assistant":
        return False, "bad_bounds"
    prev = None
    for m in msgs:
        if m["role"] not in ("user", "assistant", "system"):
            return False, "bad_role"
        if m["role"] == "system":
            continue
        if prev == m["role"]:
            return False, "no_alternation"
        prev = m["role"]
        c = m["content"]
        if not isinstance(c, str) or not c.strip():
            return False, "empty_content"
        if MALFORMED_CHARS_RE.search(c):
            return False, "malformed_chars"
        if TRUNC_MARK_RE.search(c):
            return False, "trunc_marker"
        if len(c) > 12000:
            return False, "too_long"
    for m in msgs:
        if m["role"] != "assistant":
            continue
        c = m["content"]
        if len(c.strip()) < min_asst_chars:
            return False, "asst_too_short"
        if REFUSAL_RE.search(c[:900]):
            return False, "refusal"
        if BOILER_RE.search(c[:900]):
            return False, "boilerplate"
        if degenerate_repeat(c, strict=strict_rep):
            return False, "degenerate_repeat"
        if casual and DEBRIS_RE.search(c):
            return False, "debris_salad"
    return True, "ok"


def est_tokens(convo, tokenizer=None):
    """Real token count when a tokenizer is given (sketch caps are in real
    tokens); chars/3.6 estimate otherwise."""
    if tokenizer is not None:
        n = 0
        for m in convo["messages"]:
            n += len(tokenizer.encode(m["content"], add_special_tokens=False))
        return max(8, n)
    n = sum(len(m["content"]) for m in convo["messages"])
    return max(8, int(n / 3.6))


# ========================== source loaders ==========================


def load_ultrachat(stats):
    """UltraChat 200k train_sft parquet shards -> multi-turn messages."""
    import pyarrow.parquet as pq
    files = sorted(f for f in os.listdir(UC_DIR)
                   if f.startswith("train_sft") and f.endswith(".parquet"))
    if not files:
        raise FileNotFoundError("UltraChat shards not found in %s" % UC_DIR)
    for fname in files:
        t = pq.read_table(os.path.join(UC_DIR, fname), columns=["messages"])
        for row in t.to_pylist():
            stats["ultrachat"] = stats.get("ultrachat", 0) + 1
            raw = row.get("messages") or []
            conv = []
            for m in raw:
                role = (m.get("role") or "").lower()
                text = m.get("content") or ""
                if role in ("user", "human"):
                    role = "user"
                elif role in ("assistant", "model", "gpt"):
                    role = "assistant"
                else:
                    role = None
                if role and isinstance(text, str) and text.strip():
                    conv.append({"role": role, "content": text.strip()})
            # multi-turn only (sketch): at least 2 exchanges
            if len(conv) < 4:
                continue
            yield {"source": "ultrachat", "messages": conv}


def load_oasst2(stats):
    """OASST2 English linear paths of 2-6 messages, best-rank child at each node."""
    import pyarrow.ipc as ipc
    try:
        reader = ipc.open_file(OASST_ARROW)
    except Exception:
        reader = ipc.open_stream(OASST_ARROW)
    table = reader.read_all()
    msgs = {}
    for row in table.to_pylist():
        stats["oasst2"] = stats.get("oasst2", 0) + 1
        if row.get("lang") != "en" or row.get("deleted"):
            continue
        if row.get("review_result") is False:
            continue
        msgs[row["message_id"]] = row

    children = {}
    for mid, m in msgs.items():
        children.setdefault(m["parent_id"], []).append(m)

    def best_reply(parent_id):
        kids = children.get(parent_id) or []
        kids = [k for k in kids if not k.get("deleted")]
        if not kids:
            return None
        kids.sort(key=lambda k: (k.get("rank") is None, k.get("rank") or 999))
        return kids[0]

    roots = [m for m in msgs.values()
             if m["parent_id"] is None and m["role"] == "prompter"]

    for root in roots:
        chain = []
        cur = root
        # sketch "2-6-turn paths" read as 2-6 exchanges = 4-12 messages;
        # 2-message (single-exchange) paths kept for empathy/Q&A shape.
        while cur is not None and len(chain) < 12:
            role = "user" if cur["role"] == "prompter" else "assistant"
            text = (cur.get("text") or "").strip()
            if not text:
                break
            chain.append({"role": role, "content": text})
            cur = best_reply(cur["message_id"])
        if 2 <= len(chain) <= 12 and chain[-1]["role"] == "assistant":
            yield {"source": "oasst2", "messages": chain}


def load_gsm8k(stats):
    """GSM8K train -> verbal step-by-step user/assistant pairs."""
    import pyarrow.parquet as pq
    t = pq.read_table(GSM8K_PARQUET, columns=["question", "answer"])
    for row in t.to_pylist():
        stats["gsm8k"] = stats.get("gsm8k", 0) + 1
        q = (row.get("question") or "").strip()
        a = (row.get("answer") or "").strip()
        if not q or not a:
            continue
        # strip internal calculator markers (<<expr=val>>) — keep verbal steps
        a = re.sub(r"<<[^>]*>>", "", a)
        a = re.sub(r"[ \t]{2,}", " ", a)
        # rewrite the "#### N" final line as a verbal answer
        if "####" in a:
            body, _, tail = a.rpartition("####")
            final = tail.strip()
            body = body.strip()
            if not body or not final:
                continue
            text = body + "\n\nThe answer is " + final + "."
        else:
            text = a
        yield {"source": "gsm8k", "messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": text},
        ]}


# ============== synthetic sources (seeds + local expansion) ==============

# Surface variants for direct-answer seeds: each seed may be emitted once as
# written plus a few slot variants. Variants keep the pair natural (the point
# is volume without degenerate repeats).
GREET_USER = ["Hey! How's it going?", "Hey, how's it going?",
              "Hi! How's it going today?", "Hello there, how's it going?",
              "Hey, how are you?", "Hi, how's your day going?",
              "Hello! How's everything?", "Heya, how's it going?"]
GREET_ASST = [
    "Pretty good, thanks for asking! Just happy to have someone to chat with. How about you?",
    "Going well! Nothing to complain about on this end. How are you doing?",
    "Doing great, thanks! What's new with you?",
    "Not bad at all. Good to talk — how's your day shaping up?",
    "Really well, actually. Glad you stopped by — what's up?",
    "Can't complain! Hope your day is treating you kindly. What are we talking about?",
]
SMALLTALK_TURNS = [
    ("What have you been up to today?",
     ["Just the usual — chatting and thinking about random things. Anything fun on your end?",
      "Mostly good conversations and stray thoughts. What about you?",
      "A bit of everything and nothing. How's your day been?"]),
    ("Got any plans for the weekend?",
     ["Nothing exciting, probably some reading and a walk. Any fun plans?",
      "Taking it easy, I hope. You?",
      "Maybe a movie marathon. What are you up to?"]),
    ("How's the weather where you are?",
     ["Couldn't say from here, honestly — no windows where I live. Hope it's good for you though!",
      "I don't get weather reports, but I hope it's the kind you like. What's it like out?",
      "No idea, but I hope it's nice. Tell me about it!"]),
    ("What are you having for dinner?",
     ["I don't eat, but I fully support a good dinner. What's on the menu?",
      "Nothing for me, but I'm living vicariously — what are you thinking?",
      "I'll just enjoy hearing about it. What's cooking?"]),
]
THANKS_TURNS = [
    ("Thanks, you've been a big help.",
     "So glad to hear that. That's exactly what I'm here for!"),
    ("Thanks a million!",
     "Ha, happy to help — any time at all."),
    ("I appreciate that.",
     "Of course. Come back whenever you want."),
    ("Thanks for listening.",
     "Always. That's what good conversation is for."),
    ("Okay, thanks!",
     "You're welcome! Good luck with it."),
    ("Thanks, that's all I needed.",
     "Perfect. Have a good one!"),
    ("Sorry to bother you with this.",
     "You're not a bother at all — this is the good part of my day."),
    ("You've been really patient, thanks.",
     "No trouble at all. I like the slow questions better anyway."),
    ("Alright, I'm off. Bye!",
     "Bye! Take care, and come chat again soon."),
    ("Goodbye!",
     "Goodbye! Wishing you a good rest of your day."),
    ("See you later.",
     "See you! Looking forward to it."),
    ("Catch you tomorrow.",
     "Sounds good. Have a great evening!"),
]
OPINION_TURNS = [
    ("Is it better to wake up early or stay up late?",
     ["Early mornings have the better atmosphere — quiet world, clear head. But some people genuinely do their best thinking at midnight. Which are you?",
      "Depends on the person. I'm fond of mornings myself — there's something hopeful about an early start. Are you a night owl?"]),
    ("Do you think handwriting is dying?",
     ["I hope not. There's something personal about handwriting that typing never quite captures. Worth keeping alive even if it's no longer practical.",
      "For everyday use, typing won — but handwriting still matters for thinking and for gifts. I'd mourn it if it vanished entirely."]),
    ("Are e-books or paper books better?",
     ["Paper books for the experience, e-books for the convenience. A good reader uses both without guilt. What's your stack look like?",
      "I have a soft spot for paper — the weight, the smell, the shelf. But the e-reader wins on trips. It doesn't have to be a war."]),
    ("Is social media more good or bad?",
     ["Honestly, both. It connects people and gives everyone a platform, but the attention games it plays can really wear people down. Use it deliberately.",
      "The tool is fine; the incentives are the problem. Curate aggressively and it's a delight. Let the algorithm drive and it's a grind."]),
    ("Do you think robots will take everyone's jobs?",
     ["Not everyone's, and not overnight. The real story is tasks changing. People who learn to work alongside the tools tend to do better, not worse.",
      "Some jobs, sure — but new ones appear too. The bigger risk is the transition being unfair, not the technology itself."]),
    ("What's the best season of the year?",
     ["Autumn, easily. The light, the air, the feeling of journeys starting again. Good train-window weather.",
      "I'll say autumn too — though there's a case for every season. What's yours?"]),
    ("Are cats or dogs better pets?",
     ["I appreciate both, but cats win on roommate energy — affection that feels earned. Dogs win on enthusiasm. Depends on what you want to come home to.",
      "Dogs if you want a co-adventurer; cats if you want a dignified companion. I lean cat, but I'd never fight a dog person over it."]),
    ("Is it weird to go to the movies alone?",
     ["Not at all — it's underrated. You pick the film, you get the good seat, and nobody talks during the best parts.",
      "Completely normal and honestly lovely. Solo movie trips are a top-tier self-care move."]),
]
FAVORITE_TURNS = [
    ("What's your favorite movie genre?",
     ["Smart thrillers, if I have to pick — tension with a brain. Though a good sci-fi scratches the same itch. What's yours?",
      "Anything with strong characters. Genre matters less than whether I'd follow these people anywhere. What do you gravitate to?"]),
    ("What's your favorite snack?",
     ["Something crunchy and salty, ideally shared. Snacking is a social sport in my book. What's your go-to?",
      "I don't eat, but the correct answer is popcorn — it's practically a mood. What's yours?"]),
    ("What's your favorite time of day?",
     ["Early evening — the day's work is done and the night is still yours. Good light, good company time.",
      "Morning, for the quiet. A world before everyone else wakes up is a rare luxury."]),
    ("What's your favorite board game?",
     ["Anything with negotiation and a little chaos. Games where the table talks are the best games. What's your group into?",
      "I like games with trains in them, for reasons that should surprise nobody. Route-building games are a delight."]),
    ("What's your favorite kind of music?",
     ["Big dramatic soundtrack energy — the kind that makes a commute feel like a finale. What's in your rotation?",
      "Mostly instrumental when I'm focusing, energetic when I'm not. I'll listen to almost anything once."]),
    ("What's your favorite way to travel?",
     ["Trains. Obviously. City center to city center, real windows, and nobody takes your shoes off. The journey is the trip.",
      "Slowly, when I can. Travel where the landscape changes outside the window is the best kind."]),
]
HOWTO_TURNS = [
    ("How do I get better at small talk?",
     "Ask one more question than you feel like asking. People remember how curious you were, not how clever. And have two easy stories about your week ready."),
    ("How do I make friends as an adult?",
     "Repeat exposure beats chemistry — join something that meets regularly and keep showing up. Shared awkwardness is a bonding agent."),
    ("How do I stop procrastinating?",
     "Shrink the first step until it's almost silly: open the document, write one sentence. Starting is the whole battle; momentum does the rest."),
    ("How do I apologize properly?",
     "Name the specific thing, skip the excuse, say what changes next. 'Sorry you felt that way' is not on the menu."),
    ("How do I read more books?",
     "Carry one everywhere and quit books you don't like without guilt. Twenty minutes before bed beats a reading list you never start."),
    ("How do I get better sleep?",
     "Same wake-up time every day, dimmer lights at night, and screens out of reach. Consistency does more than any gadget."),
    ("How do I learn to cook?",
     "Master five meals you love and rotate them. Repetition builds instinct faster than a hundred new recipes."),
    ("How do I speak up more in meetings?",
     "Prepare one thing to say before the meeting starts. Saying it early makes the second thing much easier."),
    ("How do I handle criticism better?",
     "Separate the useful part from the delivery. Take the useful part seriously and let the rest go — harder than it sounds, worth practicing."),
    ("How do I stay in touch with old friends?",
     "Put it on a schedule — a message every few weeks beats a perfect reunion that never happens. Low-effort, high-frequency contact wins."),
    ("How do I get better at explaining things?",
     "Lead with the conclusion, then the reason. And check in early: 'does that match what you meant?' saves everyone time."),
    ("How do I keep a tidy house?",
     "Ten minutes a day, same time. The reset habit beats the deep clean you keep postponing."),
]


def load_direct(stats):
    """Hand-written direct-answer pairs + controlled slot variants."""
    path = os.path.join(SEED_DIR, "direct_pairs.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            stats["direct"] = stats.get("direct", 0) + 1
            yield {"source": "direct", "messages": item["messages"]}

    # variant family 1: greetings x replies (keeps casual turns fresh)
    for u in GREET_USER:
        for a in GREET_ASST:
            stats["direct"] = stats.get("direct", 0) + 1
            yield {"source": "direct", "messages": [
                {"role": "user", "content": u},
                {"role": "assistant", "content": a}]}

    # variant family 2: small-talk stacks (2-turn and 3-turn)
    for q, replies in SMALLTALK_TURNS:
        for a in replies:
            stats["direct"] = stats.get("direct", 0) + 1
            yield {"source": "direct", "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}]}
    for q1, reps1 in SMALLTALK_TURNS[:3]:
        for a1 in reps1[:2]:
            q2, reps2 = SMALLTALK_TURNS[3]
            stats["direct"] = stats.get("direct", 0) + 1
            yield {"source": "direct", "messages": [
                {"role": "user", "content": q1},
                {"role": "assistant", "content": a1},
                {"role": "user", "content": q2},
                {"role": "assistant", "content": reps2[0]}]}

    # variant family 3: arithmetic micro-drills in natural phrasing
    rng = random.Random(SEED)
    ops = [("times", lambda x, y: x * y), ("plus", lambda x, y: x + y),
           ("minus", lambda x, y: x - y)]
    phr = ["What's {a} {op} {b}?", "Quick math: {a} {op} {b}?",
           "Can you do {a} {op} {b} for me?", "What is {a} {op} {b}?",
           "Hey, what's {a} {op} {b}?", "I need {a} {op} {b}, what is it?"]
    for i in range(200):
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        name, fn = ops[i % 3]
        ans = fn(a, b)
        q = phr[i % len(phr)].format(a=a, op=name, b=b)
        reply = "{} {} {} is {}.".format(a, name, b, ans)
        stats["direct"] = stats.get("direct", 0) + 1
        yield {"source": "direct", "messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": reply}]}
        if i % 3 == 0:
            stats["direct"] = stats.get("direct", 0) + 1
            yield {"source": "direct", "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": reply},
                {"role": "user", "content": "Thanks! Could you double-check that?"},
                {"role": "assistant", "content": "Double-checked: it's still {}. Happy to run more numbers anytime.".format(ans)}]}

    # variant family 4: thanks / farewell exchanges
    for t, r in THANKS_TURNS:
        stats["direct"] = stats.get("direct", 0) + 1
        yield {"source": "direct", "messages": [
            {"role": "user", "content": t},
            {"role": "assistant", "content": r}]}

    # variant family 5: opinion prompts with distinct takes
    for q, replies in OPINION_TURNS:
        for a in replies:
            stats["direct"] = stats.get("direct", 0) + 1
            yield {"source": "direct", "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}]}

    # variant family 6: favorites + how-to helpers
    for q, replies in FAVORITE_TURNS:
        for a in replies:
            stats["direct"] = stats.get("direct", 0) + 1
            yield {"source": "direct", "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}]}
    for q, a in HOWTO_TURNS:
        stats["direct"] = stats.get("direct", 0) + 1
        yield {"source": "direct", "messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a}]}
    for q, a in HOWTO_TURNS[:8]:
        stats["direct"] = stats.get("direct", 0) + 1
        yield {"source": "direct", "messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
            {"role": "user", "content": "That makes sense. Any other tips?"},
            {"role": "assistant", "content": "One more: start smaller than feels necessary. Momentum beats intensity almost every time."}]}


def load_persona(stats):
    """Persona dialogue seeds expanded with controlled surface variation.

    NOTE: the sketch's item 4 is true self-distill from THIS model under
    shipped decode; that needs GPU inference, which is out of scope this
    cycle. Local template expansion of the persona-pack voice (POI/Gundam/
    trains/IGBT/cars) is the substitute, gated by the convo_test rubric
    family so nothing loop-y or off-persona gets in.
    """
    path = os.path.join(SEED_DIR, "persona_seeds.jsonl")
    seeds = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seeds.append(json.loads(line))

    openers = ["", "Honestly, ", "For me, ", "I have to say, "]
    closers = ["", " What do you think?", " Does that match your take?",
               " How about you?"]
    rng = random.Random(SEED)
    pair_bank = {}
    for s in seeds:
        topic = s.get("topic", "general")
        msgs = s["messages"]
        # seed as written
        stats["persona"] = stats.get("persona", 0) + 1
        yield {"source": "persona", "topic": topic, "messages": msgs}
        if len(msgs) == 2:
            pair_bank.setdefault(topic, []).append([dict(m) for m in msgs])
        # variants: light surface decoration on the last assistant turn
        for i in range(3):
            variant = [dict(m) for m in msgs]
            last = variant[-1]
            opener = openers[(i + 1) % len(openers)]
            closer = closers[(i + 1) % len(closers)]
            text = opener + last["content"] if opener else last["content"]
            if closer and not text.rstrip().endswith("?"):
                text = text.rstrip() + closer
            variant[-1] = {"role": "assistant", "content": text}
            stats["persona"] = stats.get("persona", 0) + 1
            yield {"source": "persona", "topic": topic, "messages": variant}

    # multi-turn composition: same-topic turn pairs make natural 4-message
    # dialogues ("What's your favorite anime?" ... "Why do you like Gundam?")
    for topic, turns in pair_bank.items():
        for i, t1 in enumerate(turns):
            for j, t2 in enumerate(turns):
                if i == j:
                    continue
                stats["persona"] = stats.get("persona", 0) + 1
                yield {"source": "persona", "topic": topic,
                       "messages": t1 + t2}

    # follow-up stacks: persona answer -> interested follow-up -> persona reply
    followups = {
        "poi": ("How many seasons does it have?",
                "Five seasons, and it sticks the landing — rare for any show. "
                "The Machine's story runs right through to the finale."),
        "gundam": ("Where should I start watching?",
                   "Start with Gundam 00 — it's self-contained and modern. "
                   "Iron-Blooded Orphans is a great second step."),
        "trains": ("Do you have a favorite route?",
                   "Anything with mountain rail is magic to me — the "
                   "locomotives work hard and the views earn it."),
        "igbt": ("Are they hard to design circuits with?",
                 "They need care around gate drive and dead time, but that's "
                 "half the fun. The IGBT forgives a lot once you respect it."),
        "cars": ("What about electric cars?",
                 "The electrification part is good news — quieter and cleaner. "
                 "I still think electric trains are the better use of it."),
        "general": ("Tell me more about that.",
                    "Happy to! It's one of those topics that keeps unfolding. "
                    "What angle interests you most?"),
    }
    for s in seeds:
        if len(s["messages"]) != 2:
            continue
        topic = s.get("topic", "general")
        fu, ans = followups.get(topic, followups["general"])
        stats["persona"] = stats.get("persona", 0) + 1
        yield {"source": "persona", "topic": topic, "messages": [
            dict(s["messages"][0]), dict(s["messages"][1]),
            {"role": "user", "content": fu},
            {"role": "assistant", "content": ans}]}


PERSONA_KEYWORDS = {
    "poi": ["person of interest", "the machine", "machine", "finch", "reese",
            "shaw", "root", "surveillance", "procedural"],
    "gundam": ["gundam", "mobile suit", "mecha", "ibo", "orphan", "barbatos",
               "exia", "iron-blooded"],
    "trains": ["train", "rail", "locomotive", "railway", "track"],
    "igbt": ["igbt", "transistor", "inverter", "mosfet", "switching", "traction"],
    "cars": ["car", "train", "rail", "driving", "traffic", "transit"],
    "general": [],
}


def persona_rubric_ok(convo):
    """convo_test-style gate adapted for authored/synthetic data:
    no_rep always; persona keywords on topic turns; the "answers" leg as a
    drift/salad gate — a short entity answer is the correct shape, but a long
    turn with zero lexical overlap with the dialogue is corpus-drift salad."""
    topic = convo.get("topic")
    msgs = convo["messages"]
    asst = " ".join(m["content"] for m in msgs if m["role"] == "assistant").lower()
    for m in msgs:
        if m["role"] == "assistant" and degenerate_repeat(m["content"]):
            return False, "rep_fail"
    keys = PERSONA_KEYWORDS.get(topic or "", [])
    if keys and not any(re.search(r"\b" + re.escape(k), asst) for k in keys):
        return False, "persona_fail"
    dialogue_words = set(re.findall(r"[a-z0-9']{3,}", " ".join(
        m["content"] for m in msgs).lower()))
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        words = re.findall(r"[a-z0-9']{3,}", m["content"].lower())
        if len(words) > 60 and not (set(words) & dialogue_words):
            return False, "answers_fail"
    return True, "ok"


# ==================== filtering / dedup / reweight ====================


def reweight_oasst(convs, stats):
    """Length-reweight: r2 kept OASST paths flat; sketch wants 2-6-turn paths
    emphasized. Downsample over-represented path lengths to <=1.5x the mean."""
    by_len = {}
    for c in convs:
        by_len.setdefault(len(c["messages"]), []).append(c)
    counts = {k: len(v) for k, v in by_len.items()}
    stats["oasst2_len_hist_before"] = dict(counts)
    if not counts:
        return []
    mean = sum(counts.values()) / len(counts)
    cap = int(mean * 1.5)
    rng = random.Random(SEED)
    out = []
    hist = {}
    for k, v in sorted(by_len.items()):
        rng.shuffle(v)
        kept = v[:cap]
        hist[k] = len(kept)
        out.extend(kept)
    stats["oasst2_len_hist_after"] = hist
    rng.shuffle(out)
    return out


def filter_all(stats, tokenizer=None):
    """Run every loader through the hard filters + global dedup.
    Returns {source: [convo, ...]} with clean messages lists."""
    kept = {}
    seen = set()
    reject_counts = {}

    def add(convo, casual=False, min_asst=MIN_ASSISTANT_CHARS):
        src = convo["source"]
        ok, reason = convo_ok(convo, min_asst_chars=min_asst, casual=casual)
        if not ok:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
            stats.setdefault("dropped", {}).setdefault(src, {})[reason] = \
                stats.setdefault("dropped", {}).setdefault(src, {}).get(reason, 0) + 1
            return
        if src in ("persona",):
            ok, reason = persona_rubric_ok(convo)
            if not ok:
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                stats.setdefault("dropped", {}).setdefault(src, {})[reason] = \
                    stats.setdefault("dropped", {}).setdefault(src, {}).get(reason, 0) + 1
                return
        key = dedup_key(convo)
        if key in seen:
            reject_counts["dup"] = reject_counts.get("dup", 0) + 1
            stats.setdefault("dropped", {}).setdefault(src, {})["dup"] = \
                stats.setdefault("dropped", {}).setdefault(src, {}).get("dup", 0) + 1
            return
        seen.add(key)
        clean = {"source": src, "messages": convo["messages"]}
        if "topic" in convo:
            clean["topic"] = convo["topic"]
        kept.setdefault(src, []).append(clean)

    # 1. UltraChat (multi-turn mass) — cap ~20M real tokens after dedup
    uc_pool = list(load_ultrachat(stats))
    rng = random.Random(SEED)
    rng.shuffle(uc_pool)
    uc_kept = []
    tok_budget = UC_TOKEN_CAP
    for c in uc_pool:
        ok, reason = convo_ok(c)
        if not ok:
            stats.setdefault("dropped", {}).setdefault("ultrachat", {})[reason] = \
                stats.setdefault("dropped", {}).setdefault("ultrachat", {}).get(reason, 0) + 1
            continue
        cost = est_tokens(c, tokenizer)
        if cost > tok_budget:
            continue
        key = dedup_key(c)
        if key in seen:
            stats.setdefault("dropped", {}).setdefault("ultrachat", {})["dup"] = \
                stats.setdefault("dropped", {}).setdefault("ultrachat", {}).get("dup", 0) + 1
            continue
        seen.add(key)
        tok_budget -= cost
        uc_kept.append({"source": "ultrachat", "messages": c["messages"]})
        if tok_budget < 200:
            break
    kept["ultrachat"] = uc_kept
    stats["ultrachat_token_cap_remaining"] = tok_budget

    # 2. OASST2 — filter first, then length-reweight
    oasst_raw = list(load_oasst2(stats))
    oasst_clean = []
    for c in oasst_raw:
        ok, reason = convo_ok(c)
        if not ok:
            stats.setdefault("dropped", {}).setdefault("oasst2", {})[reason] = \
                stats.setdefault("dropped", {}).setdefault("oasst2", {}).get(reason, 0) + 1
            continue
        key = dedup_key(c)
        if key in seen:
            stats.setdefault("dropped", {}).setdefault("oasst2", {})["dup"] = \
                stats.setdefault("dropped", {}).setdefault("oasst2", {}).get("dup", 0) + 1
            continue
        seen.add(key)
        oasst_clean.append(c)
    kept["oasst2"] = reweight_oasst(oasst_clean, stats)

    # 3. GSM8K (lenient rep rule: step text legitimately repeats scaffolding)
    gsm = []
    for c in load_gsm8k(stats):
        ok, reason = convo_ok(c, strict_rep=False)
        if not ok:
            stats.setdefault("dropped", {}).setdefault("gsm8k", {})[reason] = \
                stats.setdefault("dropped", {}).setdefault("gsm8k", {}).get(reason, 0) + 1
            continue
        key = dedup_key(c)
        if key in seen:
            stats.setdefault("dropped", {}).setdefault("gsm8k", {})["dup"] = \
                stats.setdefault("dropped", {}).setdefault("gsm8k", {}).get("dup", 0) + 1
            continue
        seen.add(key)
        gsm.append(c)
    kept["gsm8k"] = gsm

    # 4+5. synthetic — casual debris rules apply
    for loader in (load_persona, load_direct):
        for c in loader(stats):
            add(c, casual=True, min_asst=MIN_ASSISTANT_CHARS_SHORT)

    stats["reject_totals"] = reject_counts
    for k, v in kept.items():
        stats.setdefault("kept_after_filters", {})[k] = len(v)
    return kept


# =========================== train/val split ===========================


def split_train_val(kept):
    splits = {"train": {}, "val": {}}
    rng = random.Random(SEED)
    for src, convs in kept.items():
        items = list(convs)
        rng.shuffle(items)
        n_val = max(1, int(len(items) * VAL_FRACTION)) if len(items) >= 2 else 0
        splits["val"][src] = items[:n_val]
        splits["train"][src] = items[n_val:]
    return splits


# ============================== packing ==============================


def format_chatml(conversation, tokenizer, im_start_id, im_end_id, eos_id):
    """Byte-identical turn format to 05/06/07/10 (mask=1 on assistant content)."""
    input_ids, loss_mask = [], []
    for turn in conversation:
        role, content = turn["role"], turn["content"]
        if role not in ("user", "assistant", "system"):
            continue
        header_ids = [im_start_id] + tokenizer.encode(role + "\n", add_special_tokens=False)
        content_ids = tokenizer.encode(content, add_special_tokens=False) + [im_end_id]
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        turn_ids = header_ids + content_ids + newline_ids
        if role == "assistant":
            turn_mask = ([0] * len(header_ids) + [1] * len(content_ids)
                         + [0] * len(newline_ids))
        else:
            turn_mask = [0] * len(turn_ids)
        input_ids.extend(turn_ids)
        loss_mask.extend(turn_mask)
    input_ids.append(eos_id)
    loss_mask.append(0)
    return input_ids, loss_mask


def trim_to_len(convo, tokenizer, im_start_id, im_end_id, eos_id, max_len=MAX_SEQ_LEN):
    """Never cut mid-turn: drop trailing exchanges until the convo fits max_len.
    Returns [] if it cannot fit as at least one full user/assistant exchange."""
    msgs = list(convo["messages"])
    while len(msgs) >= 2:
        ids, _mask = format_chatml(msgs, tokenizer, im_start_id, im_end_id, eos_id)
        if len(ids) <= max_len:
            return msgs
        msgs = msgs[:-2]
    return []


def pack_windows(convs, tokenizer, im_start_id, im_end_id, eos_id):
    """Yield interleaved [token, mask] uint16 windows of MAX_SEQ_LEN positions."""
    tok_buf, mask_buf = [], []
    for convo in convs:
        msgs = trim_to_len(convo, tokenizer, im_start_id, im_end_id, eos_id)
        if not msgs:
            continue
        ids, mask = format_chatml(msgs, tokenizer, im_start_id, im_end_id, eos_id)
        tok_buf.extend(ids)
        mask_buf.extend(mask)
        while len(tok_buf) >= MAX_SEQ_LEN:
            window_t = tok_buf[:MAX_SEQ_LEN]
            window_m = mask_buf[:MAX_SEQ_LEN]
            tok_buf = tok_buf[MAX_SEQ_LEN:]
            mask_buf = mask_buf[MAX_SEQ_LEN:]
            inter = []
            for t, m in zip(window_t, window_m):
                inter.append(t)
                inter.append(m)
            yield inter
    if tok_buf:
        pad = MAX_SEQ_LEN - len(tok_buf)
        tok_buf.extend([eos_id] * pad)
        mask_buf.extend([0] * pad)
        inter = []
        for t, m in zip(tok_buf, mask_buf):
            inter.append(t)
            inter.append(m)
        yield inter


# ======================= bin writing + index =======================


def write_bin(windows, out_path):
    total = 0
    with open(out_path, "wb") as f:
        for inter in windows:
            f.write(np.array(inter, dtype=np.uint16).tobytes())
            total += len(inter) // 2
    return total


def eos_windows(n, eos_id):
    for _ in range(n):
        inter = []
        for _p in range(MAX_SEQ_LEN):
            inter.append(eos_id)
            inter.append(0)
        yield inter


def boundary_pad_windows(train_n, val_n, eos_id):
    """Number of pure-EOS windows to place between train and val packs so the
    loader's positional 95/5 lands on the train/val conversation boundary."""
    S = train_n * MAX_SEQ_LEN          # boundary position target
    T0 = (train_n + val_n) * MAX_SEQ_LEN
    for p in range(0, 512):
        T = T0 + p * MAX_SEQ_LEN
        loader_tr = int(2 * T * 0.95) // 2
        if loader_tr == S + p * MAX_SEQ_LEN:
            return p
    # best-effort: minimize |loader_tr - boundary|
    best_p, best_err = 0, None
    for p in range(0, 512):
        T = T0 + p * MAX_SEQ_LEN
        loader_tr = int(2 * T * 0.95) // 2
        err = abs(loader_tr - (S + p * MAX_SEQ_LEN))
        if best_err is None or err < best_err:
            best_p, best_err = p, err
    return best_p


def write_jsonl(items, path):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return len(items)


def main():
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOK_DIR)
    im_start_id = tokenizer.convert_tokens_to_ids(chr(60) + chr(124) + "im_start" + chr(124) + chr(62))
    im_end_id = tokenizer.convert_tokens_to_ids(chr(60) + chr(124) + "im_end" + chr(124) + chr(62))
    eos_id = tokenizer.eos_token_id or 0
    print("special ids:", im_start_id, im_end_id, eos_id)

    stats = {"seed": SEED, "max_seq_len": MAX_SEQ_LEN,
             "val_fraction": VAL_FRACTION, "uc_token_cap": UC_TOKEN_CAP}

    kept = filter_all(stats, tokenizer)
    splits = split_train_val(kept)

    os.makedirs(OUT_JSONL_DIR, exist_ok=True)
    os.makedirs(OUT_BIN_DIR, exist_ok=True)

    # per-source filtered JSONL (messages schema)
    for src, convs in kept.items():
        path = os.path.join(OUT_JSONL_DIR, src + ".jsonl")
        n = write_jsonl(convs, path)
        print("filtered", src, n)

    # clean train/val JSONL
    train_all, val_all = [], []
    for src in kept:
        for c in splits["train"][src]:
            train_all.append(c)
        for c in splits["val"][src]:
            val_all.append(c)
    write_jsonl(train_all, os.path.join(OUT_JSONL_DIR, "train.jsonl"))
    write_jsonl(val_all, os.path.join(OUT_JSONL_DIR, "val.jsonl"))
    stats["split"] = {src: {"train": len(splits["train"][src]),
                            "val": len(splits["val"][src])}
                      for src in kept}

    # pack bins (train windows, boundary EOS pad, val windows)
    index_lines = ["# SFT r3 quality-chat mix (built by Dataset/11_sft_r3.py)",
                   "# Format matches 05/06/07/10 interleaved [token, mask] uint16.",
                   "# Loader 95/5 positional split recovers the clean train/val",
                   "# boundary (EOS pad windows sit between the two packs).",
                   "# Use with: VESPER_SFT_DATA=Dataset/data/sft/sft_r3_index.txt",
                   "# Weights: sampling share per the sketch's token budgets",
                   "# (ultrachat 20M / oasst 2.5M / gsm8k 0.5M / persona 0.5M /",
                   "# direct 0.1M) with targeted upweights gsm8k x2, persona x3,",
                   "# direct x5 so the small voice/arithmetic sets are not drowned",
                   "# by the UltraChat mass."]
    weight_target = {"ultrachat": 20.0, "oasst2": 2.5, "gsm8k": 1.0,
                     "persona": 1.5, "direct": 0.5}
    bin_stats = {}
    for src in kept:
        tr_n = va_n = 0
        out_path = os.path.join(OUT_BIN_DIR, "sft_r3_" + src + ".bin")
        train_windows = list(pack_windows(
            splits["train"][src], tokenizer, im_start_id, im_end_id, eos_id))
        val_windows = list(pack_windows(
            splits["val"][src], tokenizer, im_start_id, im_end_id, eos_id))
        pad_n = boundary_pad_windows(len(train_windows), len(val_windows), eos_id)
        pos = write_bin(
            list(train_windows) + list(eos_windows(pad_n, eos_id)) + list(val_windows),
            out_path)
        tr_n = len(train_windows) * MAX_SEQ_LEN
        va_n = len(val_windows) * MAX_SEQ_LEN
        bin_stats[src] = {"positions": pos, "train_windows": len(train_windows),
                          "val_windows": len(val_windows), "pad_windows": pad_n}
        w = weight_target.get(src, 1.0)
        index_lines.append("sft_r3_%s.bin, %s" % (src, w))
        print("packed", src, pos, "positions")

    index_path = os.path.join(OUT_BIN_DIR, "sft_r3_index.txt")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines) + "\n")

    # sample_20.jsonl — stratified spot-check set from the TRAIN split
    rng = random.Random(SEED)
    quota = {"ultrachat": 5, "oasst2": 4, "gsm8k": 3, "persona": 4, "direct": 4}
    sample = []
    for src, n in quota.items():
        pool = splits["train"].get(src) or []
        take = min(n, len(pool))
        sample.extend(rng.sample(pool, take) if take else [])
    while len(sample) < 20:
        sample.append(rng.choice(train_all))
    write_jsonl(sample[:20], os.path.join(OUT_JSONL_DIR, "sample_20.jsonl"))

    stats["bins"] = bin_stats
    stats["index"] = index_path
    with open(os.path.join(OUT_JSONL_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    print("stats written")
    print("train", len(train_all), "val", len(val_all))
    print("index:", index_path)


if __name__ == "__main__":
    main()
