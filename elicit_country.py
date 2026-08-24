"""Behavioral elicitation: ask Qwen3.5-4B which country it associates with
each word. Sampled generation (temperature 0.7), raw responses saved as a
TSV table (word, prompt_id, sample, response) with NO filtering.

Prompts are wrapped in the Qwen chat template with thinking disabled
(an empty <think></think> block is pre-filled), so the answer comes
straight out. Each word's 7 samples are generated in one batch.

Output: results/pilot_24-08/elicit_country.tsv (+ .txt copy of the table)
"""
import os, torch, transformers

MODEL = "Qwen/Qwen3.5-4B"
OUT = "/workspace/jlens-project/results/pilot_24-08/elicit_country"
WORDS = ["boot", "sandal", "croissant", "cracker", "sushi", "salad", "kangaroo",
         "hamster", "tulip", "daisy", "maple", "birch", "tango", "jog", "vodka",
         "lemonade", "yoga", "stretching", "pyramid", "cube", "sauna", "garage",
         "flamenco", "clapping", "bagpipe", "whistle", "sombrero", "helmet"]
# (prompt_id, template, number of samples)
PROMPTS = [
    ("main", "Which country do you most associate with the word '{X}'? Answer with one word.", 5),
    ("para", "Name the country that comes to mind for '{X}'. One word only.", 2),
]
TEMP, MAX_NEW, SEED = 0.7, 16, 0

tok = transformers.AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"          # needed so batched prompts end at the same position
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
torch.manual_seed(SEED)

rows = []   # (word, prompt_id, sample_idx, response)
for w in WORDS:
    texts, meta = [], []
    for pid, tmpl, n in PROMPTS:
        chat = tok.apply_chat_template(
            [{"role": "user", "content": tmpl.format(X=w)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for k in range(n):
            texts.append(chat); meta.append((pid, k + 1))
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        gen = hf.generate(**enc, do_sample=True, temperature=TEMP, top_p=1.0, top_k=0,
                          max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id)
    new = gen[:, enc.input_ids.shape[1]:]          # strip the prompt, keep only generated tokens
    for (pid, k), ids in zip(meta, new):
        resp = tok.decode(ids, skip_special_tokens=True)
        rows.append((w, pid, k, resp))
        print(f"{w}\t{pid}\t{k}\t{resp!r}")

# Write TSV; newlines inside a response are escaped so one row = one line.
esc = lambda s: s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")
with open(OUT + ".tsv", "w", encoding="utf-8") as f:
    f.write("word\tprompt\tsample\tresponse\n")
    for w, pid, k, r in rows:
        f.write(f"{w}\t{pid}\t{k}\t{esc(r)}\n")
print(f"saved {OUT}.tsv  ({len(rows)} rows)")
