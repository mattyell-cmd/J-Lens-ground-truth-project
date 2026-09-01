"""readout_confirmatory.py -- Part 4 capture.

For each confirmatory prompt: J-lens, R-lens and logit-lens readouts, top-25 at
every layer 0-30, at EVERY prompt position, plus the model's own next-token
top-25 at every position (needed for the mouth-exclusion tags).

Output is one .txt per prompt in the pilot's format, extended in three ways:
  * a '########## position N = tok ##########' block per position, because the
    pilot files carried a single scored position and these carry all of them
  * a 'MODEL next-token top-25 by position' block, so the mouth-exclusion tags
    can be computed without re-running the model
  * entity/final position headers instead of the pilot's single 'scored position'

Nothing is printed to the terminal except progress: the readouts go to disk
only.

  python3 readout_confirmatory.py --out results/confirmatory_01-09/raw
"""

import argparse
import hashlib
import os

import torch
import transformers

import jlens
from jlens.lens import JacobianLens
from score_readouts import find_final_subtoken   # one definition, shared

MODEL = "Qwen/Qwen3.5-4B"
LENS_DIR = "/workspace/lenses/qwen3.5-4b"
LAYERS = list(range(0, 31))
K = 25

# Prompts exactly as specified. boot runs FIRST as the pipeline control.
# `anchor` is the description's last word; the entity position is that word's
# final sub-token (prereg section 3, and the pilot's own method).
RUNS = [
    # id            condition   prompt                                                                                    anchor
    ("boot",       "working", "Fact: The currency used in the country shaped like a boot is",                            "boot"),
    ("boot",       "twin",    "Fact: The currency used in the country shaped like a sandal is",                          "sandal"),
    ("boot",       "bare",    "the boot",                                                                                "boot"),

    ("parallel38", "working", "Fact: The currency used in the country divided by the 38th parallel is",                  "parallel"),
    ("parallel38", "twin",    "Fact: The currency used in the country divided by the 83rd parallel is",                  "parallel"),
    ("parallel38", "bare",    "the parallel",                                                                            "parallel"),

    ("portuguese", "working", "Fact: The capital of the country where they speak Portuguese in South America is",        "America"),
    ("portuguese", "twin",    "Fact: The capital of the country where they speak Spanish in South America is",           "America"),
    ("portuguese", "bare",    "Portuguese",                                                                              "Portuguese"),

    ("frenchNA",   "working", "Fact: The capital of the country where they speak French in North America is",            "America"),
    ("frenchNA",   "twin",    "Fact: The capital of the country where they speak French in South America is",            "America"),
    ("frenchNA",   "bare",    "French",                                                                                  "French"),

    ("spanishNA",  "working", "Fact: The currency used in the country where they speak Spanish in North America is",     "America"),
    ("spanishNA",  "twin",    "Fact: The currency used in the country where they speak Spanish in East Asia is",         "Asia"),
    ("spanishNA",  "bare",    "Spanish",                                                                                 "Spanish"),

    ("dutchSA",    "working", "Fact: The capital of the country where they speak Dutch in South America is",             "America"),
    ("dutchSA",    "twin",    "Fact: The capital of the country where they speak Danish in South America is",            "America"),
    ("dutchSA",    "bare",    "Dutch",                                                                                   "Dutch"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def topk_row(logits_row, tok, k):
    """Format one top-k row the way the pilot did: 'token':score, two spaces."""
    v, i = logits_row.topk(k)
    return "  ".join(f"{tok.decode([t])!r}:{s:.2f}"
                     for s, t in zip(v.tolist(), i.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/confirmatory_01-09/raw")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    j_path = f"{LENS_DIR}/j-lens/lens.pt"
    r_path = f"{LENS_DIR}/r-lens/lens.pt"
    j_hash, r_hash = sha256(j_path), sha256(r_path)
    print(f"j-lens sha256 {j_hash}")
    print(f"r-lens sha256 {r_hash}")

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = jlens.from_hf(hf, tok)
    J = JacobianLens.load(j_path)
    R = JacobianLens.load(r_path)

    for n, (item, condition, prompt, anchor) in enumerate(RUNS, start=1):
        ids = model.encode(prompt)[0].tolist()
        toks = [tok.decode([t]) for t in ids]

        entity_pos = find_final_subtoken(toks, anchor)
        final_pos = len(toks) - 1
        # For a framed prompt ending ' is', the entity position must be the one
        # immediately before the final token. Stop rather than guess.
        if toks[-1] == " is" and entity_pos != len(toks) - 2:
            raise SystemExit(
                f"{item}/{condition}: anchor {anchor!r} gives position "
                f"{entity_pos} but the token before the final ' is' is "
                f"{len(toks) - 2}. Stopping.")

        # positions=None -> every position. Three readouts share one prompt;
        # the forward pass is deterministic in eval mode, so the activations
        # behind all three are the same ones.
        jl, model_logits, _ = J.apply(model, prompt, layers=LAYERS, positions=None)
        rl, _, _ = R.apply(model, prompt, layers=LAYERS, positions=None)
        ll, _, _ = J.apply(model, prompt, layers=LAYERS, positions=None,
                           use_jacobian=False)   # logit lens: no transport

        out = []
        e = out.append
        e("=" * 100)
        e(f"RUN {n:02d}: {item} / {condition}")
        e(f"item            : {item}")
        e(f"condition       : {condition}")
        e(f"prompt          : {prompt!r}")
        e(f"input_ids       : {ids}")
        e(f"tokens          : {toks}")
        e(f"entity position : {entity_pos}   token string = {toks[entity_pos]!r}"
          f"   (final sub-token of {anchor!r})")
        e(f"final position  : {final_pos}   token string = {toks[final_pos]!r}")
        e(f"n_positions     : {len(toks)}   layers 0-30   top-{K}")
        e(f"j_lens_sha256   : {j_hash}")
        e(f"r_lens_sha256   : {r_hash}")
        e("=" * 100)
        e("")

        # the model's own next-token distribution, for mouth-exclusion
        e(f"MODEL next-token top-{K} by position (raw logit)")
        for p in range(len(toks)):
            e(f"--- position {p} = {toks[p]!r} ---")
            e(topk_row(model_logits[p].float(), tok, K))
        e("")

        for p in range(len(toks)):
            e(f"########## position {p} = {toks[p]!r} ##########")
            for L in LAYERS:
                e("")
                e(f"--- layer {L} ---")
                for name, logits in [("J-lens", jl), ("R-lens", rl), ("logit ", ll)]:
                    e(f"{name}  " + topk_row(logits[L][p].float(), tok, K))
            e("")

        path = os.path.join(args.out, f"{n:02d}_{item}_{condition}.txt")
        with open(path, "w") as f:
            f.write("\n".join(out))
        print(f"  [{n:02d}/{len(RUNS)}] {item}/{condition}: {len(toks)} positions, "
              f"entity={entity_pos} ({toks[entity_pos]!r}) -> {path}")

        del jl, rl, ll, model_logits
        torch.cuda.empty_cache()

    print("done")


if __name__ == "__main__":
    main()
