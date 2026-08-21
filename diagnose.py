"""Is the rank-1-everywhere result real, or is gate.py self-confirming?

Two checks:
 1. how far each J is from the identity (if J ~ I, transport is a no-op)
 2. does rank stay 1 for MANY random tokens? if yes, the test proves nothing
    about the lens -- it just reflects that W_U[t] scores highest against
    itself, which is true by construction.
"""
import torch, transformers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

model_name = "Qwen/Qwen3.5-4B"
tok = transformers.AutoTokenizer.from_pretrained(model_name)
hf = transformers.AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.bfloat16, device_map="cuda")
W_U = hf.get_output_embeddings().weight.float()

d = torch.load("/workspace/lenses/qwen3.5-4b/j-lens/lens.pt", map_location="cuda")
J = d["J"]
layers = sorted(J.keys())

I = torch.eye(2560, device="cuda")
print("check 1: how far is each J from the identity?")
print("  (rel = ||J-I||_F / ||I||_F ; cos = similarity between v and J@v)")
tid = tok.encode(" Paris")[0]
v = W_U[tid]
rels, coss = [], []
for layer in layers:
    Jl = J[layer].float()
    rel = ((Jl - I).norm() / I.norm()).item()
    Jv = Jl @ v
    cos = torch.nn.functional.cosine_similarity(v, Jv, dim=0).item()
    rels.append(rel); coss.append(cos)
    print(f"  layer {layer:2d}  rel-dist-from-I {rel:6.3f}   cos(v, J@v) {cos:6.3f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(layers, coss, marker="o", color="tab:blue")
ax[0].set_xlabel("source layer"); ax[0].set_ylabel("cos(v, J@v)")
ax[0].set_title("direction preserved under transport\n(1.0 = unchanged)")
ax[0].grid(alpha=0.3)
ax[1].plot(layers, rels, marker="o", color="tab:red")
ax[1].set_xlabel("source layer"); ax[1].set_ylabel("||J - I||_F / ||I||_F")
ax[1].set_title("how far J is from the identity\n(0 = J does nothing)")
ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/workspace/jlens-project/results/transport_fidelity.png", dpi=150)
print("saved /workspace/jlens-project/results/transport_fidelity.png")

# check 2: the control. 200 random tokens, does the target always win?
print("\ncheck 2: rank of 200 random tokens (not just ' Paris')")
g = torch.Generator(device="cpu").manual_seed(0)
tids = torch.randint(0, W_U.shape[0], (200,), generator=g).tolist()
for layer in [0, 10, 20, 30]:
    Jl = J[layer].float()
    ranks = []
    for t in tids:
        scores = W_U @ (Jl @ W_U[t])
        ranks.append((scores > scores[t]).sum().item() + 1)
    ranks = torch.tensor(ranks, dtype=torch.float)
    frac1 = (ranks == 1).float().mean().item()
    print(f"  layer {layer:2d}  frac at rank 1: {frac1:.3f}   median rank: "
          f"{ranks.median().item():.0f}   max rank: {ranks.max().item():.0f}")
