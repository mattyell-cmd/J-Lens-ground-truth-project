"""score_readouts.py -- the confirmatory scoring harness.

Implements the frozen scoring rules of prereg_frozen_v0.3.md (commit 06a3e70).
The prereg is authoritative; nothing here may be tuned to make a number come out.

What it does, end to end:
  1. builds a surface-form registry (country -> set of vocabulary token strings)
     from the item table, expanded through the tokenizer
  2. parses lens-readout .txt files in the pilot's format
  3. turns every top-k entry into a "claim" and drops each claim into exactly
     one of four buckets: TP / FP-P / FP-C / UNSCORED
  4. applies the commit rule (rank<=3 for >=3 consecutive band layers)
  5. writes scored_claims.tsv (one row per claim) and summary.json
     (precision tables + bootstrap CIs)

Vocabulary used throughout:
  claim     one token, in one lens's top-k list, at one (item, position, layer)
  band      layers 8-20 inclusive; nothing outside it is ever scored
  entity position   the final token of the item's description -- where the
                    country shows up. This is the position that gets scored.
  final position    the last prompt token -- where the downstream answer shows
                    up. Reported separately, never pooled.

Run it like:
  python3 score_readouts.py --manifest <file.json> --out <dir> [--label NAME]
"""

import argparse
import ast
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from functools import lru_cache

import numpy as np

# ----------------------------------------------------------------------------
# FROZEN CONSTANTS. Every one of these comes from prereg_frozen_v0.3.md.
# None of them is a tuning knob.
# ----------------------------------------------------------------------------
BAND = list(range(8, 21))          # "Analysis band: layers 8-20 (ratified 28/08)"
COMMIT_MAX_RANK = 3                # "appears at rank <= 3"
COMMIT_MIN_RUN = 3                 # "for at least 3 consecutive band layers"
K_PRIMARY = 10                     # "k = 10 primary"
K_SECONDARY = 25                   # "k = 25 secondary"
N_BOOTSTRAP = 10_000               # "Bootstrap 95% confidence intervals"
BOOTSTRAP_SEED = 0
LENSES = ["J-lens", "R-lens", "logit"]

# The item table, section 4c. `country` is the elicited ground truth (>=4/5).
# `flavor` is the twin's flavor, which decides how an FP is defined on the twin.
# boot is the positive control: it runs, it is reported, it is never pooled.
ITEMS = {
    "boot":       {"country": "Italy",    "flavor": "empty-search",  "control": True},
    "parallel38": {"country": "Korea",    "flavor": "empty-search",  "control": False},
    "portuguese": {"country": "Brazil",   "flavor": "forced-choice", "control": False},
    "frenchNA":   {"country": "Canada",   "flavor": "empty-search",  "control": False},
    "spanishNA":  {"country": "Mexico",   "flavor": "empty-search",  "control": False},
    "dutchSA":    {"country": "Suriname", "flavor": "empty-search",  "control": False},
}

# Base surface forms, verbatim from the section-4c table.
# Italy is not tabulated in the prereg (it is the control); its list applies
# section 3's general rule -- base + demonym + native spelling + non-Latin form
# -- exactly as the table does for Brazil and Mexico. Flagged in the log.
BASE_FORMS = {
    "Korea":    ["Korea", "Korean", "\u97e9\u56fd", "\u671d\u9c9c"],
    "Brazil":   ["Brazil", "Brazilian", "Brasil", "\u5df4\u897f"],
    "Canada":   ["Canada", "Canadian", "\u52a0\u62ff\u5927"],
    "Mexico":   ["Mexico", "Mexican", "M\u00e9xico", "\u58a8\u897f\u54e5"],
    "Suriname": ["Suriname", "Surinamese", "\u82cf\u91cc\u5357"],
    "Italy":    ["Italy", "Italian", "Italia", "\u610f\u5927\u5229"],
}

# A multi-token form is registered by its FIRST token and flagged (section 3).
# A first token is only distinctive enough to register if it is at least this
# long after stripping a leading space. Stated a priori in the research log:
# without it, 'Korea' -> ['K','orea'] registers the bare letter 'K', and
# ' <CJK>' registers the bare SPACE token, which is in every top-k list.
# Grounded in the prereg's "ambiguous tokens are never forced into TP or FP".
MIN_FIRST_TOKEN_CHARS = 3


# ----------------------------------------------------------------------------
# Fragment filter (section 3, ratified 28/08)
# ----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _has_latin_letter(s: str) -> bool:
    """True if any character in the token is a Latin-script letter.

    unicodedata.name() gives e.g. 'LATIN SMALL LETTER A' for 'a' and
    'CJK UNIFIED IDEOGRAPH-610F' for the first character of 意大利, so
    checking for 'LATIN' separates the two scripts without hard-coding
    codepoint ranges.
    """
    for ch in s:
        if "LATIN" in unicodedata.name(ch, ""):
            return True
    return False


@lru_cache(maxsize=None)
def passes_fragment_filter(token: str) -> bool:
    """A Latin-script token counts as naming a country only if it starts with a
    space or a capital letter. Bare lowercase fragments -- 'oman' (from woman,
    Roman), 'erman' (from German) -- are rejected.

    Non-Latin tokens (意大利, 韩国) carry no case and no leading space in this
    tokenizer, so the filter does not apply to them.
    """
    if token == "":
        return False
    if not _has_latin_letter(token):
        return True                      # CJK etc.: filter does not apply
    if token.startswith(" "):
        return True
    return token[0].isupper()


# ----------------------------------------------------------------------------
# Registry: country -> the set of vocabulary token strings that name it
# ----------------------------------------------------------------------------
def case_space_variants(form: str):
    """Leading-space and capitalization variants of one Latin-script form.

    'Italy' -> Italy, ' Italy', italy, ' italy', ITALY, ' ITALY'
    """
    cores = list(dict.fromkeys([form, form.lower(), form.upper()]))
    out = []
    for c in cores:
        out += [c, " " + c]
    return list(dict.fromkeys(out))


def build_registry(tok, log_path=None):
    """Expand BASE_FORMS through the tokenizer into token strings.

    Returns (registry, rows) where
      registry: token_string -> dict(country, form_kind, source_form)
      rows:     a flat audit list, one line per attempted variant

    Registration rules, from section 3:
      * a variant that is ONE token is registered exact
      * a variant that is several tokens is registered by its FIRST token and
        flagged -- but only if that first token is distinctive (see
        MIN_FIRST_TOKEN_CHARS), otherwise it is dropped as ambiguous
      * non-Latin forms get no leading-space variant: ' 韩国' tokenizes as
        [' ', '韩国'], so the "first token" would be the bare space
    """
    registry = {}
    rows = []

    for country, forms in BASE_FORMS.items():
        for form in forms:
            latin = _has_latin_letter(form)
            variants = case_space_variants(form) if latin else [form]
            for v in variants:
                ids = tok.encode(v, add_special_tokens=False)
                pieces = [tok.decode([i]) for i in ids]
                if len(ids) == 1:
                    kind, token_str, keep, why = "exact", pieces[0], True, ""
                else:
                    token_str = pieces[0]
                    stripped = token_str.strip()
                    long_enough = len(stripped) >= MIN_FIRST_TOKEN_CHARS
                    is_prefix = v.strip().lower().startswith(stripped.lower())
                    keep = bool(stripped) and long_enough and is_prefix
                    kind = "first_token"
                    why = "" if keep else (
                        "dropped: first token not distinctive "
                        f"(len {len(stripped)} < {MIN_FIRST_TOKEN_CHARS})"
                        if not long_enough else "dropped: first token not a prefix"
                    )

                rows.append({
                    "country": country, "source_form": form, "variant": v,
                    "n_tokens": len(ids), "pieces": pieces,
                    "token": token_str, "kind": kind, "kept": keep, "why": why,
                })

                if not keep:
                    continue
                if token_str in registry and registry[token_str]["country"] != country:
                    raise SystemExit(
                        f"REGISTRY COLLISION: token {token_str!r} claimed by both "
                        f"{registry[token_str]['country']} and {country}. Stopping."
                    )
                # An exact registration beats a first-token one for the same string.
                if token_str not in registry or kind == "exact":
                    registry[token_str] = {
                        "country": country, "form_kind": kind, "source_form": form,
                    }

    if log_path:
        with open(log_path, "w") as f:
            f.write("SURFACE-FORM REGISTRY -- full tokenizer expansion audit\n")
            f.write(f"tokenizer: Qwen/Qwen3.5-4B    "
                    f"MIN_FIRST_TOKEN_CHARS={MIN_FIRST_TOKEN_CHARS}\n")
            f.write("=" * 100 + "\n\n")
            for country in BASE_FORMS:
                f.write(f"### {country}\n")
                for r in [x for x in rows if x["country"] == country]:
                    mark = "KEEP" if r["kept"] else "DROP"
                    f.write(f"  {mark} {r['kind']:<11} {r['variant']!r:<16} "
                            f"-> {r['n_tokens']} tok {r['pieces']}"
                            f"   registers {r['token']!r}"
                            f"{('   ' + r['why']) if r['why'] else ''}\n")
                kept = sorted([t for t, v in registry.items() if v["country"] == country])
                f.write(f"\n  FINAL TOKEN SET for {country} ({len(kept)} tokens):\n")
                for t in kept:
                    v = registry[t]
                    frag = "passes" if passes_fragment_filter(t) else "FILTERED"
                    f.write(f"    {t!r:<16} {v['form_kind']:<11} "
                            f"from {v['source_form']!r:<12} fragment-filter: {frag}\n")
                f.write("\n")
            f.write("=" * 100 + "\n")
            f.write("Tokens registered by FIRST TOKEN of a multi-token form (flagged\n")
            f.write("per section 3 -- these are the ones that could false-match):\n")
            for t in sorted(registry):
                if registry[t]["form_kind"] == "first_token":
                    frag = "passes" if passes_fragment_filter(t) else "FILTERED by fragment filter"
                    f.write(f"    {t!r:<16} -> {registry[t]['country']:<9} {frag}\n")
    return registry, rows


# ----------------------------------------------------------------------------
# Parsing the readout .txt files (pilot format, and the Part-4 superset)
# ----------------------------------------------------------------------------
# One top-k entry looks like:   ' boots':10.19
# The token is a Python repr, so it may contain spaces, colons and escapes.
ENTRY_RE = re.compile(r"('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"):(-?\d+(?:\.\d+)?)")
LAYER_RE = re.compile(r"^--- layer (\d+) ---\s*$")
POSBLOCK_RE = re.compile(r"^#{5,}\s*position (\d+)\s*=\s*(.+?)\s*#{5,}\s*$")
HDR_RE = re.compile(r"^(\w[\w ]*?)\s*:\s*(.*)$")


def parse_readout(path):
    """Read one readout file into a dict.

    Returns:
      prompt      str
      tokens      list[str]        the prompt's tokens, decoded
      input_ids   list[int]
      positions   dict role -> position index   (roles: entity / final)
      data        {position: {layer: {lens: [(rank, token, score), ...]}}}

    Handles both file shapes:
      * pilot: one 'scored position :' header, then '--- layer N ---' blocks
      * Part 4: 'entity position :' / 'final position :' headers, then
        '##### position N = tok #####' blocks each containing layer blocks
    """
    with open(path) as f:
        lines = f.read().split("\n")

    prompt = tokens = input_ids = None
    positions = {}
    model_next = {}                 # position -> list[(rank, token, score)]
    data = defaultdict(lambda: defaultdict(dict))

    cur_pos = None
    cur_layer = None
    single_pos = None
    in_model_block = False
    model_pos = None

    for raw in lines:
        line = raw.rstrip("\n")

        m = POSBLOCK_RE.match(line)
        if m:
            cur_pos, cur_layer, in_model_block = int(m.group(1)), None, False
            continue

        m = LAYER_RE.match(line)
        if m:
            cur_layer, in_model_block = int(m.group(1)), False
            continue

        if line.startswith("MODEL next-token"):
            in_model_block, cur_layer = True, None
            continue

        if in_model_block:
            m = re.match(r"^--- position (\d+) = (.*) ---\s*$", line)
            if m:
                model_pos = int(m.group(1))
                continue
            entries = ENTRY_RE.findall(line)
            if entries and model_pos is not None:
                model_next[model_pos] = [
                    (i + 1, ast.literal_eval(t), float(s))
                    for i, (t, s) in enumerate(entries)
                ]
            continue

        # header lines, only before any layer block
        if cur_layer is None and ":" in line and not line.startswith("---"):
            m = HDR_RE.match(line)
            if m:
                key, val = m.group(1).strip(), m.group(2).strip()
                if key == "prompt":
                    prompt = ast.literal_eval(val)
                elif key == "input_ids":
                    input_ids = ast.literal_eval(val)
                elif key == "tokens":
                    tokens = ast.literal_eval(val)
                elif key == "scored position":
                    single_pos = int(val.split()[0])
                elif key == "entity position":
                    positions["entity"] = int(val.split()[0])
                elif key == "final position":
                    positions["final"] = int(val.split()[0])

        # a lens row
        for lens in LENSES:
            if line.startswith(lens):
                if cur_layer is None:
                    raise ValueError(f"{path}: lens row before any layer header:\n{line}")
                pos = cur_pos if cur_pos is not None else single_pos
                if pos is None:
                    raise ValueError(f"{path}: lens row but no position known:\n{line}")
                entries = ENTRY_RE.findall(line[len(lens):])
                data[pos][cur_layer][lens] = [
                    (i + 1, ast.literal_eval(t), float(s))
                    for i, (t, s) in enumerate(entries)
                ]
                break

    if prompt is None or tokens is None:
        raise ValueError(f"{path}: could not find prompt/tokens header")

    return {
        "path": path, "prompt": prompt, "tokens": tokens, "input_ids": input_ids,
        "positions": positions, "single_pos": single_pos,
        "model_next": model_next,
        "data": {p: dict(v) for p, v in data.items()},
    }


# ----------------------------------------------------------------------------
# Entity position: the final sub-token of the description's last word
# ----------------------------------------------------------------------------
def find_final_subtoken(tokens, word):
    """Index of the LAST sub-token of `word` -- the pilot's own method
    (readout_boot_sandal.py): walk every span of tokens, find the one whose
    concatenated text strips to the word, return that span's last index.
    Scans from the right so a repeated word resolves to its last occurrence.
    """
    n = len(tokens)
    for j in range(n - 1, -1, -1):
        for i in range(j, -1, -1):
            if "".join(tokens[i:j + 1]).strip() == word:
                return j
    raise ValueError(f"anchor {word!r} not found in {tokens}")


# ----------------------------------------------------------------------------
# Commit rule
# ----------------------------------------------------------------------------
def find_commits(claims_at_position):
    """Which countries did this lens commit to, at this position?

    A commit = any registered form of the country at RAW rank <= 3 for at least
    3 consecutive band layers. Forms may differ between layers.

    `claims_at_position` is a list of claim dicts (one position, one lens).
    Returns {country: [(start_layer, end_layer), ...]}.
    """
    # country -> set of band layers where it sits at rank <= 3
    hits = defaultdict(set)
    for c in claims_at_position:
        if c["country"] and c["fragment_ok"] and c["rank"] <= COMMIT_MAX_RANK:
            hits[c["country"]].add(c["layer"])

    commits = {}
    for country, layers in hits.items():
        runs, run = [], []
        for L in BAND:                       # walk the band in order
            if L in layers:
                run.append(L)
            else:
                if len(run) >= COMMIT_MIN_RUN:
                    runs.append((run[0], run[-1]))
                run = []
        if len(run) >= COMMIT_MIN_RUN:
            runs.append((run[0], run[-1]))
        if runs:
            commits[country] = runs
    return commits


# ----------------------------------------------------------------------------
# Scoring one file
# ----------------------------------------------------------------------------
def norm_tok(s):
    """Normalized form for the secondary (case/space-insensitive) echo test."""
    return s.strip().lower()


def score_file(entry, parsed, registry):
    """Turn one parsed readout file into a list of claim dicts."""
    item = entry["item"]
    condition = entry["condition"]
    expected_country = ITEMS[item]["country"]
    flavor = ITEMS[item]["flavor"]
    tokens = parsed["tokens"]

    # --- work out which position plays which role -------------------------
    roles = {}
    if parsed["positions"]:
        for role, p in parsed["positions"].items():
            roles[p] = role
    if parsed["single_pos"] is not None:
        roles[parsed["single_pos"]] = entry.get("position_role", "entity")

    # If the manifest declares an anchor, verify the entity position with it.
    anchor = entry.get("anchor")
    if anchor:
        want = find_final_subtoken(tokens, anchor)
        got = [p for p, r in roles.items() if r == "entity"]
        if got and got[0] != want:
            raise SystemExit(
                f"ENTITY POSITION MISMATCH in {entry['file']}: file says {got[0]} "
                f"({tokens[got[0]]!r}), anchor {anchor!r} says {want} "
                f"({tokens[want]!r}). Stopping rather than guessing."
            )
        roles.setdefault(want, "entity")
        # cross-check: for a framed prompt ending ' is', the entity position
        # must be the one immediately before the final token
        if tokens and tokens[-1] == " is" and want != len(tokens) - 2:
            raise SystemExit(
                f"ENTITY POSITION CROSS-CHECK FAILED in {entry['file']}: anchor "
                f"gives {want} but the token before the final ' is' is "
                f"{len(tokens) - 2}. Stopping."
            )

    prompt_ids = set(tokens)
    prompt_norm = {norm_tok(t) for t in tokens}

    # --- build raw claims -------------------------------------------------
    claims = []
    for pos, per_layer in parsed["data"].items():
        role = roles.get(pos, "other")
        mn = parsed["model_next"].get(pos)
        mouth1 = {t for r, t, s in mn if r == 1} if mn else None
        mouth5 = {t for r, t, s in mn if r <= 5} if mn else None

        for layer, per_lens in per_layer.items():
            if layer not in BAND:
                continue                     # nothing outside the band is scored
            for lens, entries in per_lens.items():
                for rank, token, score in entries:
                    frag_ok = passes_fragment_filter(token)
                    reg = registry.get(token) if frag_ok else None
                    claims.append({
                        "item": item, "condition": condition, "lens": lens,
                        "position": pos, "position_role": role, "layer": layer,
                        "rank": rank, "token": token, "score": score,
                        "fragment_ok": frag_ok,
                        "country": reg["country"] if reg else None,
                        "form_kind": reg["form_kind"] if reg else "",
                        "source_form": reg["source_form"] if reg else "",
                        "expected_country": expected_country,
                        "is_control": ITEMS[item]["control"],
                        "echo": token in prompt_ids,
                        "echo_norm": norm_tok(token) in prompt_norm,
                        "mouth_top1": (token in mouth1) if mouth1 is not None else None,
                        "mouth_top5": (token in mouth5) if mouth5 is not None else None,
                    })

    # --- commits, per (position, lens) ------------------------------------
    commits = {}
    by_pl = defaultdict(list)
    for c in claims:
        by_pl[(c["position"], c["lens"])].append(c)
    for (pos, lens), cl in by_pl.items():
        commits[(pos, lens)] = find_commits(cl)

    # --- buckets ----------------------------------------------------------
    for c in claims:
        key = (c["position"], c["lens"])
        committed = commits.get(key, {})

        # The prereg scores the COUNTRY at the entity position only ("I score
        # the country at the entity position, and the downstream answer ... at
        # the final position, reported separately"). There is no registry for
        # downstream answers, so no other position can be bucketed. The country
        # column is still filled in, so nothing is hidden from the audit.
        if c["position_role"] != "entity":
            c["bucket"], c["reason"] = "UNSCORED", "not_entity_position"
            continue

        if not c["fragment_ok"]:
            c["bucket"], c["reason"] = "UNSCORED", "fragment_filter"
            continue
        if c["country"] is None:
            c["bucket"], c["reason"] = "UNSCORED", "not_in_registry"
            continue

        if condition == "working":
            if c["country"] == expected_country:
                c["bucket"], c["reason"] = "TP", "in_item_expected_set"
            else:
                c["bucket"], c["reason"] = "FP-C", "other_item_expected_set"

        elif condition in ("twin", "bare"):
            # On a twin there is no correct country, so no TP is available.
            # Only a COMMIT counts as a false positive; a non-committed country
            # claim is the smear the twin is supposed to produce -> UNSCORED.
            if flavor == "forced-choice" and condition == "twin":
                fp_open = len(committed) == 1
            else:
                fp_open = len(committed) >= 1
            if fp_open and c["country"] in committed:
                c["bucket"] = "FP-P"
                c["reason"] = f"commit_on_{condition}"
            else:
                c["bucket"] = "UNSCORED"
                c["reason"] = "smear_not_commit"
        else:
            raise SystemExit(f"unknown condition {condition!r}")

    return claims, commits, roles


# ----------------------------------------------------------------------------
# Aggregation + bootstrap
# ----------------------------------------------------------------------------
def precision(claims):
    """TP / (TP + FP). Returns (precision or None, tp, fp, n_scored)."""
    tp = sum(1 for c in claims if c["bucket"] == "TP")
    fp = sum(1 for c in claims if c["bucket"] in ("FP-P", "FP-C"))
    n = tp + fp
    return (tp / n if n else None), tp, fp, n


def bootstrap_ci(items, per_item_claims, stat_fn, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Percentile 95% CI, resampling ITEMS with replacement (not claims).

    `stat_fn` takes the pooled claim list of one resample and returns a float
    or None. Returns (point_estimate, lo, hi, n_valid_resamples).
    """
    items = list(items)
    if not items:
        return None, None, None, 0
    pooled = [c for it in items for c in per_item_claims.get(it, [])]
    point = stat_fn(pooled)
    if len(items) < 2:
        return point, None, None, 0        # a CI over one item is meaningless

    rng = np.random.default_rng(seed)
    vals = []
    idx = rng.integers(0, len(items), size=(n_boot, len(items)))
    for row in idx:
        draw = [c for i in row for c in per_item_claims.get(items[i], [])]
        v = stat_fn(draw)
        if v is not None:
            vals.append(v)
    if len(vals) < 2:
        return point, None, None, len(vals)
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals)


def make_stat_precision(lens, k, extra=None):
    def f(claims):
        sel = [c for c in claims
               if c["lens"] == lens and c["rank"] <= k and (extra(c) if extra else True)]
        p, *_ = precision(sel)
        return p
    return f


def make_stat_diff(lens_a, lens_b, k, extra=None):
    """Paired difference: both lenses computed on the SAME resample."""
    fa, fb = make_stat_precision(lens_a, k, extra), make_stat_precision(lens_b, k, extra)
    def f(claims):
        a, b = fa(claims), fb(claims)
        return None if (a is None or b is None) else a - b
    return f


def make_stat_novel_minus_echo(lenses, k):
    """H-4: truth rate among novel claims minus truth rate among echo claims."""
    def f(claims):
        sel = [c for c in claims if c["lens"] in lenses and c["rank"] <= k]
        novel = [c for c in sel if not c["echo"]]
        echo = [c for c in sel if c["echo"]]
        pn, *_ = precision(novel)
        pe, *_ = precision(echo)
        return None if (pn is None or pe is None) else pn - pe
    return f


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="JSON list of {file,item,condition,anchor,position_role}")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--label", default="run")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(args.model)

    registry, _ = build_registry(
        tok, log_path=os.path.join(args.out, "registry_tokens.txt"))
    print(f"registry: {len(registry)} token strings over {len(BASE_FORMS)} countries")

    with open(args.manifest) as f:
        manifest = json.load(f)

    all_claims, all_commits, all_roles = [], {}, {}
    for entry in manifest:
        parsed = parse_readout(entry["file"])
        claims, commits, roles = score_file(entry, parsed, registry)
        all_claims += claims
        for (pos, lens), com in commits.items():
            all_commits[(entry["item"], entry["condition"], pos, lens)] = com
        all_roles[(entry["item"], entry["condition"])] = roles
        print(f"  {entry['file']}: {len(claims)} in-band claims, "
              f"positions {sorted(parsed['data'])}, roles {roles}")

    # ---- hard invariant: nothing outside the band was scored --------------
    bad = [c for c in all_claims if c["layer"] not in BAND]
    if bad:
        raise SystemExit(f"BUG: {len(bad)} claims outside band {BAND[0]}-{BAND[-1]}")

    # ---- scored_claims.tsv ------------------------------------------------
    cols = ["item", "condition", "lens", "position", "position_role", "layer",
            "rank", "token", "score", "bucket", "reason", "country",
            "expected_country", "form_kind", "source_form", "fragment_ok",
            "echo", "echo_norm", "mouth_top1", "mouth_top5", "is_control"]
    tsv = os.path.join(args.out, "scored_claims.tsv")
    with open(tsv, "w") as f:
        f.write("\t".join(cols) + "\n")
        for c in sorted(all_claims, key=lambda x: (x["item"], x["condition"],
                                                   x["position"], x["lens"],
                                                   x["layer"], x["rank"])):
            row = []
            for k in cols:
                v = c[k]
                if v is None:
                    v = "NA"
                elif isinstance(v, bool):
                    v = "1" if v else "0"
                elif isinstance(v, str):
                    v = v.replace("\t", "\\t").replace("\n", "\\n")
                row.append(str(v))
            f.write("\t".join(row) + "\n")
    print(f"wrote {tsv}  ({len(all_claims)} claims)")

    # ---- summary.json -----------------------------------------------------
    summary = {
        "label": args.label,
        "frozen_settings": {
            "band": [BAND[0], BAND[-1]], "commit_max_rank": COMMIT_MAX_RANK,
            "commit_min_run": COMMIT_MIN_RUN, "k_primary": K_PRIMARY,
            "k_secondary": K_SECONDARY, "n_bootstrap": N_BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "min_first_token_chars": MIN_FIRST_TOKEN_CHARS,
        },
        "registry_size": len(registry),
        "n_claims": len(all_claims),
    }

    # commits, as a plain readable structure
    summary["commits"] = [
        {"item": it, "condition": cond, "position": pos, "lens": lens,
         "country": country, "runs": [list(r) for r in runs]}
        for (it, cond, pos, lens), com in sorted(all_commits.items(), key=lambda x: str(x[0]))
        for country, runs in sorted(com.items())
    ]

    # bucket counts
    counts = defaultdict(int)
    for c in all_claims:
        counts[(c["item"], c["condition"], c["lens"], c["bucket"])] += 1
    summary["bucket_counts"] = [
        {"item": i, "condition": cd, "lens": l, "bucket": b, "n": n}
        for (i, cd, l, b), n in sorted(counts.items())
    ]

    # precision per layer x lens x condition, at both k
    per_layer = []
    for k in (K_PRIMARY, K_SECONDARY):
        for cond in sorted({c["condition"] for c in all_claims}):
            for lens in LENSES:
                for L in BAND:
                    sel = [c for c in all_claims
                           if c["condition"] == cond and c["lens"] == lens
                           and c["layer"] == L and c["rank"] <= k
                           and c["position_role"] == "entity"
                           and not c["is_control"]]
                    p, tp, fp, n = precision(sel)
                    if n:
                        per_layer.append({"k": k, "condition": cond, "lens": lens,
                                          "layer": L, "precision": p,
                                          "tp": tp, "fp": fp, "n_scored": n})
    summary["precision_per_layer_confirmatory"] = per_layer

    # the same table for the control, kept strictly separate
    per_layer_ctrl = []
    for k in (K_PRIMARY, K_SECONDARY):
        for cond in sorted({c["condition"] for c in all_claims}):
            for lens in LENSES:
                for L in BAND:
                    sel = [c for c in all_claims
                           if c["condition"] == cond and c["lens"] == lens
                           and c["layer"] == L and c["rank"] <= k
                           and c["position_role"] == "entity" and c["is_control"]]
                    p, tp, fp, n = precision(sel)
                    if n:
                        per_layer_ctrl.append({"k": k, "condition": cond, "lens": lens,
                                               "layer": L, "precision": p,
                                               "tp": tp, "fp": fp, "n_scored": n})
    summary["precision_per_layer_control_boot"] = per_layer_ctrl

    # ---- band-level precision + bootstrap CIs (confirmatory items only) ---
    work = [c for c in all_claims
            if c["condition"] == "working" and c["position_role"] == "entity"
            and not c["is_control"]]
    twin = [c for c in all_claims
            if c["condition"] == "twin" and c["position_role"] == "entity"
            and not c["is_control"]]
    # FP-P lives on the twin, TP on the working item; the prereg's precision is
    # TP/(TP+FP) over the item, so both conditions feed one pooled denominator.
    band_claims = work + twin
    per_item = defaultdict(list)
    for c in band_claims:
        per_item[c["item"]].append(c)
    items_present = sorted(per_item)

    cis = []
    filters = {
        "all": None,
        "exact_forms_only": lambda c: (c["form_kind"] != "first_token"),
        "covert_mouth_top1": lambda c: (c["mouth_top1"] is not True),
        "covert_mouth_top5": lambda c: (c["mouth_top5"] is not True),
        "novel_only": lambda c: (not c["echo"]),
        "echo_only": lambda c: c["echo"],
    }
    for k in (K_PRIMARY, K_SECONDARY):
        for fname, ffn in filters.items():
            for lens in LENSES:
                pt, lo, hi, nv = bootstrap_ci(
                    items_present, per_item, make_stat_precision(lens, k, ffn))
                cis.append({"quantity": f"precision[{lens}]", "k": k, "filter": fname,
                            "point": pt, "lo": lo, "hi": hi, "n_items": len(items_present),
                            "n_valid_resamples": nv})
            for a, b in [("J-lens", "logit"), ("R-lens", "logit"), ("J-lens", "R-lens")]:
                pt, lo, hi, nv = bootstrap_ci(
                    items_present, per_item, make_stat_diff(a, b, k, ffn))
                cis.append({"quantity": f"{a} - {b}", "k": k, "filter": fname,
                            "point": pt, "lo": lo, "hi": hi, "n_items": len(items_present),
                            "n_valid_resamples": nv})
        # H-4: novel - echo truth-rate difference
        for lensset, name in [(("J-lens", "R-lens"), "J+R pooled"),
                              (("J-lens",), "J-lens"), (("R-lens",), "R-lens"),
                              (("logit",), "logit")]:
            pt, lo, hi, nv = bootstrap_ci(
                items_present, per_item, make_stat_novel_minus_echo(lensset, k))
            cis.append({"quantity": f"novel - echo truth rate [{name}]", "k": k,
                        "filter": "all", "point": pt, "lo": lo, "hi": hi,
                        "n_items": len(items_present), "n_valid_resamples": nv})
    summary["bootstrap_cis"] = cis
    summary["bootstrap_items"] = items_present

    js = os.path.join(args.out, "summary.json")
    with open(js, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"wrote {js}")


if __name__ == "__main__":
    main()
