"""Gate (a): model config shapes.

Claim under test: vocab 248320, hidden 2560, 32 layers.

We check the claim three ways, because a config file can disagree with the
weights that actually get loaded:
  1. what the config file says
  2. what the loaded unembedding matrix (W_U) actually measures
  3. what jlens's own wrapper reports (this is the number the lens code uses)
"""
import torch, transformers, jlens

MODEL = "Qwen/Qwen3.5-4B"
EXPECT = {"vocab_size": 248320, "hidden_size": 2560, "num_hidden_layers": 32}

# --- 1. the config file ------------------------------------------------
# This checkpoint is a *multimodal* wrapper (vision tower + text decoder),
# so the text numbers are nested under text_config, not at the top level.
# get_text_config() is the accessor that digs them out.
cfg = transformers.AutoConfig.from_pretrained(MODEL)
tcfg = cfg.get_text_config()
print("architectures      :", cfg.architectures)
print("model_type (top)   :", cfg.model_type)
print("model_type (text)  :", tcfg.model_type)
print("has vision_config  :", hasattr(cfg, "vision_config"))
print()
print(f"{'field':<20} {'config':>10} {'expected':>10}  match")
for field, want in EXPECT.items():
    got = getattr(tcfg, field)
    print(f"{field:<20} {got:>10} {want:>10}  {got == want}")
print()
print("tie_word_embeddings:", tcfg.tie_word_embeddings)

# --- 2. the actual loaded weights --------------------------------------
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cuda")

# W_U ("unembedding"): one row per vocab token, each row d_model long.
# Multiplying a residual by it scores every token in the vocabulary.
W_U = hf.get_output_embeddings().weight
print("\nW_U shape          :", tuple(W_U.shape), W_U.dtype)
print("  -> vocab from W_U:", W_U.shape[0], "| hidden from W_U:", W_U.shape[1])
print("tokenizer len      :", len(tok))

# --- 3. what jlens itself sees -----------------------------------------
model = jlens.from_hf(hf, tok)
print("\njlens wrapper      :", repr(model))
print("layout             :", model.layout)
print("n_layers           :", model.n_layers)
print("d_model            :", model.d_model)
print("len(model.layers)  :", len(model.layers))

# --- verdict -----------------------------------------------------------
ok = (tcfg.vocab_size == 248320 == W_U.shape[0]
      and tcfg.hidden_size == 2560 == W_U.shape[1] == model.d_model
      and tcfg.num_hidden_layers == 32 == len(model.layers) == model.n_layers)
print("\nGATE (a):", "PASS" if ok else "FAIL")
