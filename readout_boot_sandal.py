"""Top-10 readout at the final token of 'boot' / 'sandal', layers 10-28,
for three lenses: J-lens, R-lens, logit lens (no transport, J = I).

Score = raw logit from model.unembed (final norm -> lm_head). Output is the
raw table, no filtering. Also written to results/readout_boot_sandal.txt.
"""
import torch, transformers, jlens
from jlens.lens import JacobianLens

MODEL = "Qwen/Qwen3.5-4B"
LENS_DIR = "/workspace/lenses/qwen3.5-4b"
PROMPTS = {
    "boot":   "She bought a new boot at the market yesterday.",
    "sandal": "She bought a new sandal at the market yesterday.",
}
LAYERS = list(range(10, 29))
K = 10

hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cuda")
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
model = jlens.from_hf(hf, tok)
J = JacobianLens.load(f"{LENS_DIR}/j-lens/lens.pt")
R = JacobianLens.load(f"{LENS_DIR}/r-lens/lens.pt")

out_lines = []
def emit(s=""):
    print(s); out_lines.append(s)

def find_final_subtoken(ids, word):
    """Index of the last sub-token of `word`: walk the tokens, find the span
    whose concatenated text (stripped) equals the word, return its last index."""
    toks = [tok.decode([t]) for t in ids]
    for i in range(len(toks)):
        for j in range(i, len(toks)):
            if "".join(toks[i:j+1]).strip() == word:
                return j, toks
    raise ValueError(f"{word!r} not found in {toks}")

for word, prompt in PROMPTS.items():
    ids = model.encode(prompt)[0].tolist()
    pos, toks = find_final_subtoken(ids, word)
    emit("=" * 100)
    emit(f"prompt : {prompt!r}")
    emit(f"tokens : {toks}")
    emit(f"readout position {pos} = {toks[pos]!r}  (final sub-token of {word!r})")
    emit("=" * 100)

    # apply() runs the model once per call; all three share the same
    # activations since eval-mode forward is deterministic.
    jl, _, _ = J.apply(model, prompt, layers=LAYERS, positions=[pos])
    rl, _, _ = R.apply(model, prompt, layers=LAYERS, positions=[pos])
    ll, _, _ = J.apply(model, prompt, layers=LAYERS, positions=[pos],
                       use_jacobian=False)  # logit lens: unembed(h) directly

    for l in LAYERS:
        emit(f"\n--- layer {l} ---")
        for name, logits in [("J-lens", jl), ("R-lens", rl), ("logit ", ll)]:
            v, i = logits[l][0].topk(K)
            cells = [f"{tok.decode([t])!r}:{s:.2f}" for s, t in zip(v.tolist(), i.tolist())]
            emit(f"{name}  " + "  ".join(cells))
    emit()

path = "/workspace/jlens-project/results/readout_boot_sandal.txt"
with open(path, "w") as f:
    f.write("\n".join(out_lines))
print("saved", path)
