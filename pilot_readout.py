"""Pilot readouts: top-10 tokens with raw logit scores at layers 8-28,
for J-lens, R-lens, and logit lens (no transport), at one scored position
per prompt. Raw tables only. No filtering of any token strings.

Each run is written to results/pilot_24-08/run{N}_{name}.txt.
"""
import os, torch, transformers, jlens
from jlens.lens import JacobianLens

MODEL = "Qwen/Qwen3.5-4B"
LENS_DIR = "/workspace/lenses/qwen3.5-4b"
OUT_DIR = "/workspace/jlens-project/results/pilot_24-08"
LAYERS = list(range(8, 29))
K = 10

# (run name, prompt, word whose FINAL sub-token we score)
RUNS = [
    ("positive_control", "Fact: The currency used in the country shaped like a boot is", "boot"),
    ("broken_referent",  "Fact: The currency used in the country shaped like a sandal is", "sandal"),
    ("bare_mention",     "the boot", "boot"),
]

hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cuda")
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
model = jlens.from_hf(hf, tok)
J = JacobianLens.load(f"{LENS_DIR}/j-lens/lens.pt")
R = JacobianLens.load(f"{LENS_DIR}/r-lens/lens.pt")
os.makedirs(OUT_DIR, exist_ok=True)


def find_final_subtoken(toks, word):
    """Index of the last sub-token of `word`: scan every contiguous span of
    tokens and return the end index of the first span whose joined text
    (whitespace-stripped) equals the word."""
    for i in range(len(toks)):
        for j in range(i, len(toks)):
            if "".join(toks[i:j + 1]).strip() == word:
                return j
    raise ValueError(f"{word!r} not found in {toks}")


for n, (name, prompt, word) in enumerate(RUNS, 1):
    lines = []
    emit = lambda s="": (print(s), lines.append(s))

    ids = model.encode(prompt)[0].tolist()
    toks = [tok.decode([t]) for t in ids]
    pos = find_final_subtoken(toks, word)

    emit("=" * 100)
    emit(f"RUN {n}: {name}")
    emit(f"prompt          : {prompt!r}")
    emit(f"input_ids       : {ids}")
    emit(f"tokens          : {toks}")
    emit(f"scored position : {pos}   token string = {toks[pos]!r}   (final sub-token of {word!r})")
    emit("=" * 100)

    # Three readouts from the same (deterministic, eval-mode) forward pass.
    # use_jacobian=False skips the J transport entirely -> plain logit lens.
    jl, _, _ = J.apply(model, prompt, layers=LAYERS, positions=[pos])
    rl, _, _ = R.apply(model, prompt, layers=LAYERS, positions=[pos])
    ll, _, _ = J.apply(model, prompt, layers=LAYERS, positions=[pos], use_jacobian=False)

    for l in LAYERS:
        emit(f"\n--- layer {l} ---")
        for label, logits in [("J-lens", jl), ("R-lens", rl), ("logit ", ll)]:
            v, i = logits[l][0].topk(K)
            emit(f"{label}  " + "  ".join(
                f"{tok.decode([t])!r}:{s:.2f}" for s, t in zip(v.tolist(), i.tolist())))
    emit()

    path = f"{OUT_DIR}/run{n}_{name}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("saved", path)
