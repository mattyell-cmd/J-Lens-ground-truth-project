"""Verify that `J @ h` is the correct transport orientation.

Method: run the README's currency-prompt example through jlens.apply (the
library's own code path), then recompute the same logits by hand with an
explicit J @ h matrix-vector product. If they match to numerical precision,
our orientation is right.

The control: also compute the TRANSPOSED orientation J.T @ h. If that also
matched, the test would be vacuous -- so we check it genuinely differs.
"""
import torch, transformers, jlens
from jlens.lens import JacobianLens
from jlens.hooks import ActivationRecorder

model_name = "Qwen/Qwen3.5-4B"
PROMPT = "Fact: The currency used in the country shaped like a boot is"
POS = -2

hf = transformers.AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.bfloat16, device_map="cuda")
tok = transformers.AutoTokenizer.from_pretrained(model_name)
model = jlens.from_hf(hf, tok)

# load from the local file rather than the hub
lens = JacobianLens.load("/workspace/lenses/qwen3.5-4b/j-lens/lens.pt")
layers = sorted(lens.jacobians.keys())

# ---- 1. the library's own answer -------------------------------------
lens_logits, model_logits, input_ids = lens.apply(model, PROMPT, positions=[POS])
toks = [tok.decode([t]) for t in input_ids[0]]
print(f"prompt tokens: {toks}")
print(f"position {POS} = {toks[POS]!r}\n")

# ---- 2. the same thing by hand ---------------------------------------
# re-run the model and grab the residual stream at every source layer
ids = model.encode(PROMPT, max_length=512)
with ActivationRecorder(model.layers, at=layers) as rec:
    model.forward(ids)
    acts = {l: rec.activations[l].detach() for l in layers}

print(f"{'layer':>5}  {'max|diff|':>10}  {'match':>5}   "
      f"{'top1 (J@h)':>14}  {'top1 (J.T@h)':>14}  {'same?':>5}")
all_match, orientation_matters = True, 0
for l in layers:
    h = acts[l][0][POS].float()                     # [d_model] residual at pos
    J = lens.jacobians[l].to(h.device).float()

    mine = model.unembed(J @ h).float().cpu()       # explicit matrix-vector
    theirs = lens_logits[l][0]                      # library's answer

    diff = (mine - theirs).abs().max().item()
    ok = torch.allclose(mine, theirs, atol=1e-2, rtol=1e-3)
    all_match &= ok

    flipped = model.unembed(J.T @ h).float().cpu()  # the control
    t_mine = tok.decode([mine.argmax().item()])
    t_flip = tok.decode([flipped.argmax().item()])
    same = t_mine == t_flip
    orientation_matters += (not same)

    print(f"{l:>5}  {diff:>10.5f}  {str(ok):>5}   "
          f"{t_mine!r:>14}  {t_flip!r:>14}  {str(same):>5}")

print(f"\nall layers match library: {all_match}")
print(f"orientation changes top-1 at {orientation_matters}/{len(layers)} layers "
      f"(if this were 0, the test would prove nothing)")

# what the model itself actually says, for reference
print("\nmodel's real top-5 at this position:",
      [tok.decode([t]) for t in model_logits[0].topk(5).indices])
print("J-lens top-5 by layer:")
for l in layers:
    print(f"  layer {l:2d}", [tok.decode([t]) for t in lens_logits[l][0].topk(5).indices])
