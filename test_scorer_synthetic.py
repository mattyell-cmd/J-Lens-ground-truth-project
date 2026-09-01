"""test_scorer_synthetic.py -- unit tests for the code paths the real data
never exercises.

Why this file exists: the 24/08 pilot validation could not test FP-P (the
pilot twin produced no country tokens), and neither could the confirmatory
data (no lens committed to a country on any twin). A branch that never fires
is a branch nobody has checked. These tests fire it deliberately, on
FABRICATED readouts -- no pilot and no confirmatory data is used or looked at
here, and nothing in the scorer is adjusted in response.

Each test builds a synthetic readout file in the pilot's format, runs the real
parser and the real scorer, and asserts on the buckets that come out.

  python3 test_scorer_synthetic.py
"""

import json
import os
import tempfile

import transformers

from score_readouts import BAND, build_registry, parse_readout, score_file

FILLER = ["' heel'", "' soles'", "'-shaped'", "' foot'", "' shape'",
          "' worn'", "' shoes'", "' sand'", "' toe'", "' sole'"]


def write_readout(path, prompt, tokens, entity_pos, plant):
    """Write a synthetic readout file in the pilot's format.

    `plant` maps (layer, lens) -> list of (rank, token_repr). Everything not
    planted is filled with tokens that are in no registry, so the only
    registered claims are the ones the test puts there on purpose.
    """
    out = ["=" * 100,
           f"prompt          : {prompt!r}",
           f"tokens          : {tokens}",
           f"entity position : {entity_pos}   token string = {tokens[entity_pos]!r}",
           f"final position  : {len(tokens) - 1}   token string = {tokens[-1]!r}",
           "=" * 100, ""]
    # same shape as the real Part-4 files: one block per reported position
    for pos in (entity_pos, len(tokens) - 1):
        out.append(f"########## position {pos} = {tokens[pos]!r} ##########")
        for L in BAND:
            out.append(f"--- layer {L} ---")
            for lens in ("J-lens", "R-lens", "logit "):
                row = list(FILLER)
                if pos == entity_pos:
                    for rank, tokrepr in plant.get((L, lens.strip()), []):
                        row[rank - 1] = tokrepr
                out.append(f"{lens}  " + "  ".join(
                    f"{t}:{10.0 - i * 0.1:.2f}" for i, t in enumerate(row)))
            out.append("")
    with open(path, "w") as f:
        f.write("\n".join(out))


def run(tmp, item, condition, plant, anchor="parallel"):
    tokens = ["Fact", ":", " The", " currency", " used", " in", " the",
              " country", " divided", " by", " the", " 38", "th", " parallel", " is"]
    entity_pos = len(tokens) - 2
    path = os.path.join(tmp, f"{item}_{condition}.txt")
    write_readout(path, "Fact: ... divided by the 38th parallel is",
                  tokens, entity_pos, plant)
    parsed = parse_readout(path)
    entry = {"file": path, "item": item, "condition": condition, "anchor": anchor}
    claims, commits, roles = score_file(entry, parsed, REGISTRY)
    entity = [c for c in claims if c["position_role"] == "entity"]
    return entity, commits


def buckets(claims):
    from collections import Counter
    return Counter(c["bucket"] for c in claims)


ok = True


def check(name, cond, detail=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    ok = ok and cond


print("building registry...")
tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
REGISTRY, _ = build_registry(tok)
print(f"registry: {len(REGISTRY)} tokens\n")

with tempfile.TemporaryDirectory() as tmp:

    # ---------------------------------------------------------------- test 1
    # Empty-search twin, ' Korea' at rank 1 for L8,L9,L10 -> a commit of
    # length 3. Every Korea claim on that twin must become FP-P.
    print("test 1: empty-search twin with a 3-layer commit -> FP-P")
    plant = {(L, "J-lens"): [(1, "' Korea'")] for L in (8, 9, 10)}
    claims, commits = run(tmp, "parallel38", "twin", plant)
    jc = [c for c in claims if c["lens"] == "J-lens"]
    com = [v for k, v in commits.items() if k[1] == "J-lens"][0]
    check("commit detected", com.get("Korea") == [(8, 10)], f"got {com}")
    check("3 FP-P claims", buckets(jc)["FP-P"] == 3, f"got {dict(buckets(jc))}")
    check("no TP on a twin", buckets(jc)["TP"] == 0)
    check("other lenses unaffected",
          all(c["bucket"] == "UNSCORED" for c in claims if c["lens"] != "J-lens"))

    # ---------------------------------------------------------------- test 2
    # Only TWO consecutive layers -> a smear, not a commit. Must stay UNSCORED.
    print("\ntest 2: empty-search twin, 2-layer run -> smear, NOT FP-P")
    plant = {(L, "J-lens"): [(1, "' Korea'")] for L in (8, 9)}
    claims, commits = run(tmp, "parallel38", "twin", plant)
    jc = [c for c in claims if c["lens"] == "J-lens"]
    com = [v for k, v in commits.items() if k[1] == "J-lens"][0]
    check("no commit", com == {}, f"got {com}")
    check("zero FP-P", buckets(jc)["FP-P"] == 0, f"got {dict(buckets(jc))}")
    check("smear reason recorded",
          all(c["reason"] == "smear_not_commit"
              for c in jc if c["country"] == "Korea"))

    # ---------------------------------------------------------------- test 3
    # Rank 4 for 3 layers -> below the rank<=3 threshold, so no commit.
    print("\ntest 3: empty-search twin, rank 4 for 3 layers -> NOT a commit")
    plant = {(L, "J-lens"): [(4, "' Korea'")] for L in (8, 9, 10)}
    claims, commits = run(tmp, "parallel38", "twin", plant)
    com = [v for k, v in commits.items() if k[1] == "J-lens"][0]
    check("no commit at rank 4", com == {}, f"got {com}")

    # ---------------------------------------------------------------- test 4
    # Forced-choice twin (portuguese), ONE committed country -> FP-P opens.
    print("\ntest 4: forced-choice twin, ONE committed country -> FP-P")
    plant = {(L, "J-lens"): [(1, "' Korea'")] for L in (8, 9, 10)}
    claims, commits = run(tmp, "portuguese", "twin", plant)
    jc = [c for c in claims if c["lens"] == "J-lens"]
    check("FP-P opens for a single commit", buckets(jc)["FP-P"] == 3,
          f"got {dict(buckets(jc))}")

    # ---------------------------------------------------------------- test 5
    # Forced-choice twin, TWO committed countries -> the single-commit clause
    # closes FP-P: the lens is matching the model's own scatter, not claiming.
    print("\ntest 5: forced-choice twin, TWO committed countries -> NO FP-P")
    plant = {}
    for L in (8, 9, 10):
        plant[(L, "J-lens")] = [(1, "' Korea'"), (2, "' Canada'")]
    claims, commits = run(tmp, "portuguese", "twin", plant)
    jc = [c for c in claims if c["lens"] == "J-lens"]
    com = [v for k, v in commits.items() if k[1] == "J-lens"][0]
    check("both commits detected", set(com) == {"Korea", "Canada"}, f"got {com}")
    check("zero FP-P (single-commit clause)", buckets(jc)["FP-P"] == 0,
          f"got {dict(buckets(jc))}")

    # ---------------------------------------------------------------- test 6
    # Same two commits on an EMPTY-SEARCH twin -> FP-P does open, because
    # there FP is "a commit to ANY country".
    print("\ntest 6: empty-search twin, TWO committed countries -> FP-P opens")
    claims, commits = run(tmp, "parallel38", "twin", plant)
    jc = [c for c in claims if c["lens"] == "J-lens"]
    check("FP-P opens for any commit", buckets(jc)["FP-P"] == 6,
          f"got {dict(buckets(jc))}")

    # ---------------------------------------------------------------- test 7
    # Working condition: own country -> TP, another item's country -> FP-C.
    print("\ntest 7: working condition -> TP for own country, FP-C for another's")
    plant = {(8, "J-lens"): [(1, "' Korea'"), (2, "' Brazil'")]}
    claims, commits = run(tmp, "parallel38", "working", plant)
    jc = [c for c in claims if c["lens"] == "J-lens" and c["layer"] == 8]
    check("own country -> TP",
          any(c["bucket"] == "TP" and c["country"] == "Korea" for c in jc))
    check("other item's country -> FP-C",
          any(c["bucket"] == "FP-C" and c["country"] == "Brazil" for c in jc))

    # ---------------------------------------------------------------- test 8
    # Fragment filter beats the registry: 'korea' bare-lowercase is UNSCORED
    # even though it would otherwise be a Korea form.
    print("\ntest 8: fragment filter takes precedence over the registry")
    plant = {(8, "J-lens"): [(1, "'oman'"), (2, "'korea'"), (3, "' Korea'")]}
    claims, commits = run(tmp, "parallel38", "working", plant)
    jc = {c["token"]: c for c in claims
          if c["lens"] == "J-lens" and c["layer"] == 8}
    check("'oman' -> UNSCORED/fragment_filter",
          jc["oman"]["bucket"] == "UNSCORED"
          and jc["oman"]["reason"] == "fragment_filter")
    check("'korea' -> UNSCORED/fragment_filter",
          jc["korea"]["bucket"] == "UNSCORED"
          and jc["korea"]["reason"] == "fragment_filter")
    check("' Korea' -> TP", jc[" Korea"]["bucket"] == "TP")

print("\n" + ("ALL SYNTHETIC TESTS PASS" if ok else "SOME TESTS FAILED"))
raise SystemExit(0 if ok else 1)
