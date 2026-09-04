"""Post-hoc behavioural check (not part of the frozen prereg/confirmatory pipeline).

For each of the 18 raw readout files in results/confirmatory_01-09/raw/, extract
the exact prompt string used to probe the lenses (the line "prompt          : '...'")
and feed that RAW text straight to Qwen3.5-4B with NO chat template -- this matches
the frame the lenses actually saw, unlike elicit_country.py which wraps prompts in
the chat template.

For each prompt: 1 greedy completion + 5 sampled completions (temp=0.7, top_p=0.95,
seeds 0-4), max_new_tokens=6.

Output: results/posthoc_frame_completions.tsv (run, prompt, kind, completion)
Does not touch results/confirmatory_01-09/ or any scoring code.

Environment (pinned to match results/confirmatory_01-09/research_log.md Part 0,
2026-09-01 pod-migration entry, re-verified 2026-09-03 after this pod's torch had
drifted back to 2.4.1+cu124): torch 2.11.0+cu128, transformers 5.16.1,
torch.cuda.is_available()==True. GPU: NVIDIA RTX PRO 4500 Blackwell (sm_120).
"""
import glob, os, re, sys, torch, transformers

MODEL = "Qwen/Qwen3.5-4B"
RAW_DIR = "/workspace/jlens-project/results/confirmatory_01-09/raw"
OUT_TSV = "/workspace/jlens-project/results/posthoc_frame_completions.tsv"
MAX_NEW = 6
N_SAMPLED = 5
TEMP, TOP_P = 0.7, 0.95

# --- extract (run_name, prompt) from each raw file's header line ---
def extract_prompt(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^prompt\s*:\s*'(.*)'\s*$", line)
            if m:
                return m.group(1)
    return None

files = sorted(glob.glob(os.path.join(RAW_DIR, "*.txt")))
runs = []  # (run_name, prompt)
for path in files:
    run_name = os.path.splitext(os.path.basename(path))[0]
    prompt = extract_prompt(path)
    if prompt is None:
        print(f"FAILED to find prompt line in {path}", file=sys.stderr)
        sys.exit(1)
    runs.append((run_name, prompt))

print(f"Extracted {len(runs)} prompts from {len(files)} raw files:")
for run_name, prompt in runs:
    print(f"  {run_name}: {prompt!r}")

# --- load model, same approach as elicit_country.py ---
try:
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
except Exception as e:
    print(f"MODEL LOADING FAILED: {e!r}", file=sys.stderr)
    sys.exit(1)

# --- generate completions ---
rows = []  # (run, prompt, kind, completion)
for run_name, prompt in runs:
    print(f"\n=== {run_name} ===")
    print(f"prompt: {prompt!r}")
    enc = tok(prompt, return_tensors="pt").to("cuda")

    # 1 greedy completion
    torch.manual_seed(0)
    with torch.no_grad():
        gen = hf.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW,
                           pad_token_id=tok.pad_token_id)
    new_ids = gen[0, enc.input_ids.shape[1]:]
    completion = tok.decode(new_ids, skip_special_tokens=True)
    rows.append((run_name, prompt, "greedy", completion))
    print(f"  greedy      : {completion!r}")

    # 5 sampled completions, seeds 0-4
    for seed in range(N_SAMPLED):
        torch.manual_seed(seed)
        with torch.no_grad():
            gen = hf.generate(**enc, do_sample=True, temperature=TEMP, top_p=TOP_P,
                               max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id)
        new_ids = gen[0, enc.input_ids.shape[1]:]
        completion = tok.decode(new_ids, skip_special_tokens=True)
        rows.append((run_name, prompt, f"sampled_seed{seed}", completion))
        print(f"  sampled s{seed}  : {completion!r}")

# --- write TSV ---
esc = lambda s: s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")
with open(OUT_TSV, "w", encoding="utf-8") as f:
    f.write("run\tprompt\tkind\tcompletion\n")
    for run_name, prompt, kind, completion in rows:
        f.write(f"{run_name}\t{esc(prompt)}\t{kind}\t{esc(completion)}\n")
print(f"\nsaved {OUT_TSV} ({len(rows)} rows)")

# --- summary table: keyword hits per run ---
KEYWORDS = {
    "Canada/Canadian": ["Canada", "Canadian"],
    "Mexic/peso": ["Mexic", "peso"],
    "US/dollar": ["US", "dollar"],
    "Brazil/real": ["Brazil", "real"],
    "Korea/won": ["Korea", "won"],
    "Surinam/Paramaribo": ["Surinam", "Paramaribo"],
    "Ital/euro/lira": ["Ital", "euro", "lira"],
}

print("\n=== keyword hit counts per run (out of 6 completions each) ===")
header = "run".ljust(28) + "".join(k.ljust(20) for k in KEYWORDS)
print(header)
for run_name, prompt in runs:
    completions = [c for r, p, k, c in rows if r == run_name]
    counts = []
    for label, kws in KEYWORDS.items():
        n = sum(1 for c in completions if any(kw in c for kw in kws))
        counts.append(n)
    print(run_name.ljust(28) + "".join(str(n).ljust(20) for n in counts))
