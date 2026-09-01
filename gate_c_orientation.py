"""Gate (c): orientation check -- is the transport J @ h, or J.T @ h?

A square matrix times a vector works either way round without erroring, so an
orientation bug is silent. This gate pins it down against the library itself.

Method:
  1. Ask jlens.apply() for the lens logits (the library's own code path).
  2. Recompute them by hand as  unembed(J_l @ h_l)  -- explicit matrix-vector.
     If (1) and (2) agree to numerical precision, J @ h is the right form.
  3. Control: also compute  unembed(J_l.T @ h_l).  If THAT also matched, the
     test would be vacuous -- so we check it genuinely diverges. A symmetric
     J would make the two identical; these J are not symmetric, and we
     measure that too.
"""
import torch, transformers, jlens
from jlens.lens import JacobianLens
from jlens.hooks import ActivationRecorder

MODEL = "Qwen/Qwen3.5-4B"
LENS = "/workspace/lenses/qwen3.5-4b/j-lens/lens.pt"
PROMPT = "Fact: The currency used in the country shaped like a boot is"
POS = -2  # README's example position

hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cuda")
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
model = jlens.from_hf(hf, tok)

lens = JacobianLens.load(LENS)  # local file, not the hub
layers = lens.source_layers
print("lens          :", repr(lens))
print("model         :", repr(model))
print("prompt        :", repr(PROMPT))

# --- 1. the library's answer -------------------------------------------
lens_logits, model_logits, input_ids = lens.apply(model, PROMPT, positions=[POS])
toks = [tok.decode([t]) for t in input_ids[0]]
print("tokens        :", toks)
print(f"position {POS}   : {toks[POS]!r}")

# --- 2. recompute by hand ----------------------------------------------
# Re-run the forward pass and grab the residual stream at each source layer.
# eval() + no dropout means this is deterministic, so the activations are the
# same ones apply() used internally.
ids = model.encode(PROMPT, max_length=512)
with ActivationRecorder(model.layers, at=layers) as rec:
    model.forward(ids)
    acts = {l: rec.activations[l].detach() for l in layers}

print(f"\n{'layer':>5}  {'max|manual-lib|':>15}  {'match':>5}  "
      f"{'cos(J@h, J.T@h)':>15}  {'asym ||J-J.T||_F':>16}  "
      f"{'top1 J@h':>16}  {'top1 J.T@h':>16}  {'same':>5}")

all_match = True
n_diverge = 0
for l in layers:
    h = acts[l][0][POS].float()               # [d_model] residual at that position
    J = lens.jacobians[l].to(h.device).float()

    manual = model.unembed(J @ h).float().cpu()      # the orientation we claim
    library = lens_logits[l][0]                      # what jlens produced
    flipped = model.unembed(J.T @ h).float().cpu()   # the control

    diff = (manual - library).abs().max().item()
    ok = torch.allclose(manual, library, atol=1e-2, rtol=1e-3)
    all_match &= ok

    # cosine between the two logit vectors: 1.0 would mean orientation is
    # indistinguishable here, so we want this well below 1.
    cos = torch.nn.functional.cosine_similarity(manual, flipped, dim=0).item()
    asym = ((J - J.T).norm() / J.norm()).item()  # 0 would mean J is symmetric

    t_manual = tok.decode([manual.argmax().item()])
    t_flip = tok.decode([flipped.argmax().item()])
    same = t_manual == t_flip
    n_diverge += (not same)

    print(f"{l:>5}  {diff:>15.5f}  {str(ok):>5}  {cos:>15.4f}  {asym:>16.4f}  "
          f"{t_manual!r:>16}  {t_flip!r:>16}  {str(same):>5}")

print(f"\nmanual J@h matches jlens.apply at all {len(layers)} layers: {all_match}")
print(f"transposed J.T@h changes the top-1 token at {n_diverge}/{len(layers)} layers")
print("  (if this were 0/31 the control would be vacuous -- the two")
print("   orientations would be empirically indistinguishable)")

print("\nmodel's own top-5 at this position:",
      [tok.decode([t]) for t in model_logits[0].topk(5).indices])
print("J-lens top-5 by layer:")
for l in layers:
    print(f"  layer {l:2d}", [tok.decode([t]) for t in lens_logits[l][0].topk(5).indices])

ok = all_match and n_diverge > 0
print(f"\nGATE (c): {'PASS' if ok else 'FAIL'}")
