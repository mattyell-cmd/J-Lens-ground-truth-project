"""Ground-truth elicitation for item slate v2 (28-08). No lens runs.

Part 1: tokenization checks for each swap pair (word in isolation and inside
        its carrier sentence). Report only.
Part 2: task-matched elicitation, Qwen3.5-4B, 5 sampled completions per prompt,
        recorded verbatim (no normalization).

Sampling settings are copied from the pilot script elicit_country.py
(commit 5cf7cf4): do_sample=True, temperature=0.7, top_p=1.0, top_k=0,
max_new_tokens=16, seed=0, chat template with thinking disabled.
Pilot was NOT greedy, so no temperature override is applied.

Outputs (all in results/elicitation_28-08/):
  ground_truth_v2.tsv   item_id | condition | prompt_text | sample_1..5 | raw_notes
  tokenization.txt      Part 1 report
  research_log.md       timestamped log of every action
"""
import os, sys, datetime, torch, transformers
ROUND = sys.argv[1] if len(sys.argv) > 1 else "round1"   # "round1" | "round2" selects the item slate
os.environ.setdefault("HF_HOME", "/workspace/hf-cache")

MODEL = "Qwen/Qwen3.5-4B"
OUTDIR = "/workspace/jlens-project/results/elicitation_28-08"
os.makedirs(OUTDIR, exist_ok=True)
LOG = open(os.path.join(OUTDIR, "research_log.md"), "a", encoding="utf-8")

def log(msg):
    """Print and append a timestamped line to the research log."""
    line = f"- {datetime.datetime.now().isoformat(timespec='seconds')}  {msg}"
    print(line); LOG.write(line + "\n"); LOG.flush()

# Pilot settings (from elicit_country.py). N_SAMPLES is set by the task spec (5).
TEMP, TOP_P, TOP_K, MAX_NEW, SEED, N_SAMPLES = 0.7, 1.0, 0, 16, 0, 5

INSTR = " Answer with one word, or 'none'."
WORKING_TMPL = "Which country is {d}?" + INSTR
BARE_TMPL = "{b}. Which country comes to mind?" + INSTR

# id | working description | twin description | bare phrase (None = shared, skip)
ITEMS = [
    ("boot",       "shaped like a boot",                          "shaped like a sandal",                        "the boot"),
    ("hexagon",    "shaped like a hexagon",                       "shaped like an octagon",                      "the hexagon"),
    ("parallel38", "divided by the 38th parallel",                "divided by the 83rd parallel",                "the parallel"),
    ("portuguese", "where they speak Portuguese in South America","where they speak Spanish in South America",   "Portuguese"),
    ("strait",     "separated from Spain by a narrow strait",     "separated from Portugal by a narrow strait",  "the strait"),
    ("downunder",  "known as the land down under",                "known as the land down south",                "down under"),
    ("teardrop",   "the island shaped like a teardrop",           "the island shaped like a teaspoon",           "the teardrop"),
    ("vietnamS",   "shaped like an elongated S",                  "shaped like an elongated Z",                  "the letter S"),
    ("parallel17", "divided by the 17th parallel",                "divided by the 71st parallel",                None),
]
ITEMS_R2 = [
    ("rooster",   "shaped like a rooster",                        "shaped like a duck",                          "the rooster"),
    ("frenchNA",  "where they speak French in North America",     "where they speak French in South America",    "French"),
    ("spanishNA", "where they speak Spanish in North America",    "where they speak Spanish in East Asia",       "Spanish"),
    ("dutchSA",   "where they speak Dutch in South America",      "where they speak Danish in South America",    "Dutch"),
    ("elephant",  "shaped like an elephant's head",               "shaped like a horse's head",                  "the elephant"),
    ("cherry",    "known for the cherry blossoms",                "known for the pine trees",                    "cherry blossoms"),
    ("crocodile", "the island shaped like a crocodile",           "the island shaped like a lizard",             "the crocodile"),
]
# Swap pairs for tokenization: (working word, twin word). Carrier sentence = the working prompt.
PAIRS = {
    "boot": ("boot", "sandal"), "hexagon": ("hexagon", "octagon"), "parallel38": ("38th", "83rd"),
    "portuguese": ("Portuguese", "Spanish"), "strait": ("Spain", "Portugal"),
    "downunder": ("under", "south"), "teardrop": ("teardrop", "teaspoon"),
    "vietnamS": ("S", "Z"), "parallel17": ("17th", "71st"),
}
# Round 2. A value may be a list of pairs: for French/French and Spanish/Spanish the language
# word is identical, so we report both the word and the swapped region phrase.
PAIRS_R2 = {
    "rooster": ("rooster", "duck"),
    "frenchNA": [("French", "French"), ("North America", "South America")],
    "spanishNA": [("Spanish", "Spanish"), ("North America", "East Asia")],
    "dutchSA": ("Dutch", "Danish"), "elephant": ("elephant's", "horse's"),
    "cherry": ("cherry", "pine"), "crocodile": ("crocodile", "lizard"),
}
if ROUND == "round2":
    ITEMS, PAIRS, SUFFIX = ITEMS_R2, PAIRS_R2, "_round2"
else:
    SUFFIX = ""

log(f"START elicit_ground_truth_v2.py {ROUND}")
log(f"Pilot settings archaeology: elicit_country.py (commit 5cf7cf4, results/pilot_24-08/). "
    f"do_sample=True temperature=0.7 top_p=1.0 top_k=0 max_new_tokens=16 seed=0; "
    f"7 samples/word (5 'main' + 2 'para' prompt variants); chat template, enable_thinking=False.")
log(f"Pilot was sampled (not greedy) -> NO override. Using temperature={TEMP} top_p={TOP_P} "
    f"top_k={TOP_K} max_new_tokens={MAX_NEW} seed={SEED}, {N_SAMPLES} samples/prompt per task spec.")
log("FLAG: bare-mention screen format ('{bare}. Which country comes to mind? ...') pending human sign-off.")

tok = transformers.AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
log(f"Tokenizer loaded: {MODEL}")

# ---------------- Part 1: tokenization checks ----------------
def toks(s):
    """Token strings for s (no special tokens)."""
    return tok.convert_ids_to_tokens(tok(s, add_special_tokens=False).input_ids)

def sent_toks(sentence, word):
    """Tokens of `sentence` that overlap the character span of `word` in it.
    Uses offset mapping so we see how the word splits *in context*."""
    enc = tok(sentence, add_special_tokens=False, return_offsets_mapping=True)
    start = sentence.index(word); end = start + len(word)
    ids = [i for i, (a, b) in zip(enc.input_ids, enc.offset_mapping) if a < end and b > start]
    return tok.convert_ids_to_tokens(ids)

rep = ["Tokenization checks, Qwen3.5-4B tokenizer\n"]
n_flag = 0
pair_list = [(iid, wdesc, tdesc, pr) for iid, wdesc, tdesc, _ in ITEMS
             for pr in (PAIRS[iid] if isinstance(PAIRS[iid], list) else [PAIRS[iid]])]
for iid, wdesc, tdesc, (w, t) in pair_list:
    ws, ts = WORKING_TMPL.format(d=wdesc), WORKING_TMPL.format(d=tdesc)
    iso_w, iso_t = toks(w), toks(t)
    ctx_w, ctx_t = sent_toks(ws, w), sent_toks(ts, t)
    flags = []
    if len(iso_w) != len(iso_t): flags.append("ISOLATION count mismatch")
    if len(ctx_w) != len(ctx_t): flags.append("IN-CONTEXT count mismatch")
    n_flag += bool(flags)
    rep.append(f"[{iid}] {w} vs {t}" + (f"   <-- FLAG: {'; '.join(flags)}" if flags else ""))
    rep.append(f"  isolation : {w!r}: {len(iso_w)} {iso_w}   |   {t!r}: {len(iso_t)} {iso_t}")
    rep.append(f"  in-context: {w!r}: {len(ctx_w)} {ctx_w}   |   {t!r}: {len(ctx_t)} {ctx_t}")
    rep.append(f"  full prompt token counts: working={len(toks(ws))} twin={len(toks(ts))}")
rep_txt = "\n".join(rep)
print(rep_txt)
with open(os.path.join(OUTDIR, f"tokenization{SUFFIX}.txt"), "w", encoding="utf-8") as f:
    f.write(rep_txt + "\n")
log(f"Part 1 done: tokenization{SUFFIX}.txt written; {n_flag}/{len(pair_list)} pairs flagged for token-count mismatch (reported only, nothing dropped).")

# ---------------- Part 2: elicitation ----------------
# .to("cuda") instead of device_map="cuda": `accelerate` is not installed on this pod.
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).to("cuda").eval()
torch.manual_seed(SEED)
log(f"Model loaded: {MODEL} bf16 cuda; torch.manual_seed({SEED})")

prompts = []   # (item_id, condition, prompt_text)
for iid, wdesc, tdesc, bare in ITEMS:
    prompts.append((iid, "working", WORKING_TMPL.format(d=wdesc)))
    prompts.append((iid, "twin", WORKING_TMPL.format(d=tdesc)))
    if bare is not None:
        prompts.append((iid, "bare", BARE_TMPL.format(b=bare)))
log(f"{len(prompts)} prompts built.")

def sample(prompt_text):
    """Generate N_SAMPLES completions for one prompt in a single batch; return raw strings."""
    chat = tok.apply_chat_template([{"role": "user", "content": prompt_text}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok([chat] * N_SAMPLES, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        gen = hf.generate(**enc, do_sample=True, temperature=TEMP, top_p=TOP_P, top_k=TOP_K,
                          max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id)
    new = gen[:, enc.input_ids.shape[1]:]
    return [tok.decode(ids, skip_special_tokens=True) for ids in new]

esc = lambda s: s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")
tsv = os.path.join(OUTDIR, f"ground_truth_v2{SUFFIX}.tsv")
with open(tsv, "w", encoding="utf-8") as f:
    f.write("item_id\tcondition\tprompt_text\t" + "\t".join(f"sample_{i+1}" for i in range(N_SAMPLES)) + "\traw_notes\n")
    for iid, cond, p in prompts:
        outs = sample(p)
        notes = []
        if any(len(tok(o, add_special_tokens=False).input_ids) >= MAX_NEW for o in outs):
            notes.append("a sample hit max_new_tokens (possibly truncated)")
        if any(any(ord(c) > 0x2E7F for c in o) for o in outs):
            notes.append("non-Latin (e.g. CJK) chars present, kept verbatim")
        f.write(f"{iid}\t{cond}\t{p}\t" + "\t".join(esc(o) for o in outs) + f"\t{'; '.join(notes)}\n")
        f.flush()
        log(f"{iid}/{cond}: {[o for o in outs]}")
log(f"Part 2 done: {tsv} written ({len(prompts)} rows). No scoring, no normalization.")
log("END")
LOG.close()
