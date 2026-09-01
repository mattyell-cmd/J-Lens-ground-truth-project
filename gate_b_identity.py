"""Gate (b): J[30] should be (near) the identity matrix.

Why this must hold: the lens was fitted with target_layer=30 (see provenance).
J_l transports a residual from layer l into the layer-30 basis. Transporting
from layer 30 to layer 30 is a no-op, so J[30] has to be I.

If it isn't, every other J is suspect and no downstream number means anything.

The control matters as much as the test: we measure the same distance for
EVERY layer. If all 31 matrices were near-identity, "J[30] is identity" would
be trivially true and would prove nothing about the fit.
"""
import torch
import matplotlib
matplotlib.use("Agg")  # no display on the pod; render straight to file
import matplotlib.pyplot as plt

LENS = "/workspace/lenses/qwen3.5-4b/j-lens/lens.pt"

# weights_only=False so we can read the provenance dict too (it holds strings)
d = torch.load(LENS, map_location="cpu", weights_only=False)
J = d["J"]
layers = sorted(J.keys())
target_layer = d["provenance"]["target_layer"]

print(f"lens file      : {LENS}")
print(f"J              : dict of {len(layers)} matrices, layers {layers[0]}..{layers[-1]}")
print(f"each matrix    : {tuple(J[layers[0]].shape)} {J[layers[0]].dtype}")
print(f"target_layer   : {target_layer}   (this is the one that must be I)")
print(f"n_prompts      : {d['n_prompts']}")

D = d["d_model"]
I = torch.eye(D, dtype=torch.float32)

# --- the test: how far is J[30] from I? --------------------------------
Jt = J[target_layer].float()
diff = Jt - I

# Frobenius norm = sqrt of the sum of every squared entry. Dividing by
# ||I||_F (= sqrt(2560)) makes it a relative, scale-free number.
rel = (diff.norm() / I.norm()).item()
diag = Jt.diagonal()
off = diff.abs().clone()
off.fill_diagonal_(0)  # zero the diagonal so we can look at off-diagonal only

print(f"\n--- J[{target_layer}] vs identity ---")
print(f"  ||J-I||_F / ||I||_F   : {rel:.3e}")
print(f"  max |J-I| (any entry) : {diff.abs().max().item():.3e}")
print(f"  diagonal  min/mean/max: {diag.min().item():.6f} / "
      f"{diag.mean().item():.6f} / {diag.max().item():.6f}")
print(f"  max |off-diagonal|    : {off.max().item():.3e}")
print(f"  exactly equal to I    : {torch.equal(Jt, I)}")
print(f"  allclose(atol=1e-3)   : {torch.allclose(Jt, I, atol=1e-3)}")

# --- the control: same measurement for every layer ----------------------
print(f"\n--- control: distance from I at every layer ---")
print(f"{'layer':>5}  {'||J-I||_F/||I||_F':>18}  {'diag mean':>10}  {'max|off-diag|':>13}")
rels = []
for l in layers:
    Jl = J[l].float()
    dl = Jl - I
    r = (dl.norm() / I.norm()).item()
    o = dl.abs().clone()
    o.fill_diagonal_(0)
    rels.append(r)
    mark = "  <-- target" if l == target_layer else ""
    print(f"{l:>5}  {r:>18.4f}  {Jl.diagonal().mean().item():>10.4f}  "
          f"{o.max().item():>13.4f}{mark}")

others = [r for l, r in zip(layers, rels) if l != target_layer]
print(f"\nnon-target layers: min rel-dist = {min(others):.4f}, "
      f"median = {sorted(others)[len(others)//2]:.4f}, max = {max(others):.4f}")
print(f"target layer {target_layer}: rel-dist = {rel:.3e}")
# rel can be exactly 0.0 (J[30] stored as a literal identity), so guard the ratio
if rel > 0:
    print(f"target is {min(others)/rel:.0f}x closer to I than the nearest other layer")
else:
    print("target is EXACTLY I (bit-for-bit); nearest other layer is "
          f"{min(others):.4f} away")

ok = torch.allclose(Jt, I, atol=1e-3) and min(others) > max(100 * rel, 0.1)
print(f"\nGATE (b): {'PASS' if ok else 'FAIL'}")

# --- plot ---------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(layers, rels, marker="o")
plt.axhline(rel, ls="--", c="tab:red", alpha=0.6,
            label=f"layer {target_layer} = {rel:.1e}")
plt.yscale("log")  # target is ~1e-3 while others are ~1, so log is essential
plt.xlabel("source layer l")
plt.ylabel(r"$\|J_l - I\|_F \, / \, \|I\|_F$")
plt.title(f"Distance of each $J_l$ from the identity (target_layer={target_layer})")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("/workspace/jlens-project/results/gate_b_identity.png", dpi=150)
print("saved /workspace/jlens-project/results/gate_b_identity.png")
