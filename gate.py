import torch, transformers
import matplotlib
matplotlib.use("Agg")            # no display on the pod, so render straight to file
import matplotlib.pyplot as plt

model_name = "Qwen/Qwen3.5-4B"

# tokenizer turns text <-> token ids; model holds the weights (incl. W_U)
tok = transformers.AutoTokenizer.from_pretrained(model_name)
hf = transformers.AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.bfloat16, device_map="cuda")

# W_U: the unembedding. one row per vocab token, each d_model long
W_U = hf.get_output_embeddings().weight
print("W_U shape:", W_U.shape, W_U.dtype)

# one fp32 copy of W_U, made ONCE. it's ~1.5GB; the old code rebuilt it
# every loop iteration, which was 31 wasted allocations.
W_U_f32 = W_U.float()

d = torch.load("/workspace/lenses/qwen3.5-4b/j-lens/lens.pt", map_location="cuda")
J = d["J"]

# J is a dict {layer_index: [2560,2560] tensor}, NOT one stacked tensor.
# so we can't use J.shape[0] / J[i] positionally -- we iterate the keys.
layers = sorted(J.keys())
print(f"J: dict of {len(layers)} matrices, layers {layers[0]}..{layers[-1]}, "
      f"each {tuple(J[layers[0]].shape)} {J[layers[0]].dtype}")
print("target_layer from provenance:", d["provenance"]["target_layer"])

# pick a token, grab its row = the direction meaning "say this token"
target = " Paris"
ids = tok.encode(target)
print(f"encode({target!r}) -> {ids} -> {[tok.decode([i]) for i in ids]}")
assert len(ids) == 1, f"{target!r} is not a single token; rank would be ill-defined"
tid = ids[0]
v = W_U_f32[tid]

# push it through J at each layer, decode, print top 5 + the target's rank
ranks = []
for layer in layers:
    transported = J[layer].float() @ v           # translate to final-layer coords
    scores = W_U_f32 @ transported               # score every vocab token
    top = scores.topk(5).indices
    # rank 1 = target is the model's top choice. count how many tokens
    # strictly outscore it, then +1 to make it 1-indexed.
    rank = (scores > scores[tid]).sum().item() + 1
    ranks.append(rank)
    print(f"layer {layer:2d}  rank {rank:6d}  top5 {[tok.decode([t]) for t in top]}")

# sanity check: J[30] is exactly the identity, so transporting from layer 30
# to layer 30 is a no-op and the target must come back as rank 1. if this
# fails, something upstream is wrong and none of the other numbers mean anything.
tgt_layer = d["provenance"]["target_layer"]
print(f"\nsanity: rank at target_layer {tgt_layer} =", ranks[layers.index(tgt_layer)],
      "(must be 1)")

plt.figure(figsize=(8, 5))
plt.plot(layers, ranks, marker="o")
plt.yscale("log")                # ranks span 1 to ~150k, so log makes it readable
plt.gca().invert_yaxis()         # better (lower) rank at the top
plt.xlabel("source layer")
plt.ylabel("rank of target token (1 = best)")
plt.title(f"J-lens decode: rank of {target!r} by layer")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/workspace/jlens-project/results/rank_by_layer.png", dpi=150)
print("saved /workspace/jlens-project/results/rank_by_layer.png")
