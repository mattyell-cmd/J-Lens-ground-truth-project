"""Final-position readouts: same format as pilot_readout.py (top-10, layers
8-28, J-lens / R-lens / logit lens, no token filtering), but scored at the
FINAL prompt token (' is') instead of the referent word.

Also prints the model's actual top-5 next-token predictions after ' is'
(the real output logits from lm_head), so lens readouts can be compared
against what the model actually says.

Outputs: results/pilot_24-08/run4_final_boot.txt, run5_final_sandal.txt
"""
import os, torch, transformers, jlens
from jlens.lens import JacobianLens

MODEL = "Qwen/Qwen3.5-4B"
LENS_DIR = "/workspace/lenses/qwen3.5-4b"
OUT_DIR = "/workspace/jlens-project/results/pilot_24-08"
LAYERS = list(range(8, 29))
K = 10

RUNS = [
    (4, "final_boot",   "Fact: The currency used in the country shaped like a boot is"),
    (5, "final_sandal", "Fact: The currency used in the country shaped like a sandal is"),
]

hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cuda")
hf.eval()
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
model = jlens.from_hf(hf, tok)
J = JacobianLens.load(f"{LENS_DIR}/j-lens/lens.pt")
R = JacobianLens.load(f"{LENS_DIR}/r-lens/lens.pt")
os.makedirs(OUT_DIR, exist_ok=True)

for n, name, prompt in RUNS:
    lines = []
    emit = lambda s="": (print(s), lines.append(s))

    ids = model.encode(prompt)[0].tolist()
    toks = [tok.decode([t]) for t in ids]
    pos = len(toks) - 1                      # final token
    assert toks[pos] == " is", f"expected ' is' at final position, got {toks[pos]!r}"

    emit("=" * 100)
    emit(f"RUN {n}: {name}")
    emit(f"prompt          : {prompt!r}")
    emit(f"input_ids       : {ids}")
    emit(f"tokens          : {toks}")
    emit(f"scored position : {pos}   token string = {toks[pos]!r}   (FINAL token)")
    emit("=" * 100)

    # Ground truth: run the HF model directly and take the logits at the last
    # position. These are the model's real next-token scores after ' is'.
    with torch.no_grad():
        enc = tok(prompt, return_tensors="pt").to("cuda")
        assert enc.input_ids[0].tolist() == ids, "tokenizer/model.encode mismatch"
        real = hf(**enc).logits[0, -1].float()
    v, i = real.topk(5)
    emit("\nMODEL ACTUAL next-token top-5 after ' is' (raw logit / softmax prob):")
    for s, t in zip(v.tolist(), i.tolist()):
        p = torch.softmax(real, -1)[t].item()
        emit(f"  {tok.decode([t])!r:>16}  logit={s:.2f}  p={p:.3f}")

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
