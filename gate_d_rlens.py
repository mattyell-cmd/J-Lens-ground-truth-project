"""Gate (d): the R-lens file loads and has the same structure as the J-lens.

Checks:
  1. both files load; top-level keys match
  2. R["J"] is a dict with the same layer keys (0..30) and per-layer shape/dtype
  3. R[30] vs identity (max abs entry deviation)
  4. R != J: max abs difference at layer 15 (and every layer, as a control)
"""
import os, torch

ROOT = "/workspace/lenses/qwen3.5-4b"
print("--- directory listing ---")
for dp, dn, fn in os.walk(ROOT):
    for f in sorted(fn):
        p = os.path.join(dp, f)
        print(f"  {os.path.getsize(p):>12,d}  {p}")

# weights_only=False because the provenance dict holds plain strings
J = torch.load(f"{ROOT}/j-lens/lens.pt", map_location="cpu", weights_only=False)
R = torch.load(f"{ROOT}/r-lens/lens.pt", map_location="cpu", weights_only=False)

print("\n--- top-level keys ---")
print("  J:", list(J.keys()))
print("  R:", list(R.keys()))
print("  same:", list(J.keys()) == list(R.keys()))

print("\n--- provenance (R) ---")
for k, v in R["provenance"].items():
    print(f"  {k}: {v!r}")

Jm, Rm = J["J"], R["J"]
jk, rk = sorted(Jm), sorted(Rm)
print("\n--- layer keys ---")
print(f"  J: {len(jk)} layers {jk[0]}..{jk[-1]}, keytype {type(jk[0]).__name__}")
print(f"  R: {len(rk)} layers {rk[0]}..{rk[-1]}, keytype {type(rk[0]).__name__}")
print("  identical key sets:", jk == rk)

# check every layer's shape/dtype, not just the first
shape_ok = all(tuple(Rm[l].shape) == (2560, 2560) for l in rk)
dtype_set = {str(Rm[l].dtype) for l in rk}
print(f"  R all layers (2560,2560): {shape_ok}   dtypes present: {dtype_set}")
print(f"  J dtypes present        : {{{', '.join(sorted({str(Jm[l].dtype) for l in jk}))}}}")
print(f"  R n_prompts={R['n_prompts']} d_model={R['d_model']}  "
      f"source_layers={R['source_layers'][0]}..{R['source_layers'][-1]} "
      f"({len(R['source_layers'])})")

I = torch.eye(2560)
R30 = Rm[30].float()
print("\n--- R[30] vs identity ---")
print(f"  max |R[30] - I|      : {(R30 - I).abs().max().item():.3e}")
print(f"  ||R[30]-I||_F/||I||_F: {((R30 - I).norm() / I.norm()).item():.3e}")
print(f"  exactly equal        : {torch.equal(R30, I)}")

print("\n--- R vs J ---")
d15 = (Rm[15].float() - Jm[15].float()).abs()
print(f"  layer 15: max |R-J| = {d15.max().item():.4f}   mean |R-J| = {d15.mean().item():.4f}")
print(f"  layer 15: ||J||_F = {Jm[15].float().norm().item():.2f}   "
      f"||R||_F = {Rm[15].float().norm().item():.2f}")
print(f"\n  {'layer':>5}  {'max|R-J|':>9}  {'mean|R-J|':>10}  {'rel ||R-J||/||J||':>18}")
for l in rk:
    a, b = Rm[l].float(), Jm[l].float()
    d = (a - b).abs()
    print(f"  {l:>5}  {d.max().item():>9.4f}  {d.mean().item():>10.5f}  "
          f"{((a-b).norm()/b.norm()).item():>18.4f}")

ok = (list(J.keys()) == list(R.keys()) and jk == rk and shape_ok
      and d15.max().item() > 1e-2)
print(f"\nGATE (d): {'PASS' if ok else 'FAIL'}")
