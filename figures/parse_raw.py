"""Parse the confirmatory raw readout files into a tidy table:
one row per (run, position, layer, lens, rank) with the token string.
Also loads the surface-form registry."""
import re, glob, os, json
import pandas as pd

RAW = '../results/confirmatory_01-09/raw'
REG = '../results/confirmatory_01-09/registry_tokens.txt'

def load_registry(exact_only=False):
    reg = {}
    cur = None
    for line in open(REG, encoding='utf-8'):
        m = re.match(r"\s*FINAL TOKEN SET for (\w+)", line)
        if m:
            cur = m.group(1); reg[cur] = set(); continue
        if cur and line.startswith("    '"):
            m = re.match(r"    '(.*?)'\s+(exact|first_token)", line)
            if m:
                tok, kind = m.group(1), m.group(2)
                if exact_only and kind != 'exact':
                    continue
                reg[cur].add(tok)
        elif cur and line.strip() == '':
            cur = None
    return reg

TOKEN_RE = re.compile(r"""('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"):(-?[\d.]+)""")

def parse_run(path):
    txt = open(path, encoding='utf-8').read()
    head = txt.split('=' * 100)[1]
    meta = {}
    for line in head.splitlines():
        if ':' in line:
            k, v = line.split(':', 1); meta[k.strip()] = v.strip()
    item, cond = meta['item'], meta['condition']
    prompt = meta['prompt'].strip("'")
    tokens = eval(meta['tokens'])
    entity = int(re.match(r'(\d+)', meta['entity position']).group(1))
    final = int(re.match(r'(\d+)', meta['final position']).group(1))
    rows = []
    # model next-token block (raw logit) — positions
    model_next = {}
    mblock = txt.split('MODEL next-token top-25 by position (raw logit)')[1].split('##########')[0]
    for pm in re.finditer(r"--- position (\d+) = .*? ---\n(.*?)(?=\n--- position|\Z)", mblock, re.S):
        p = int(pm.group(1)); toks = [eval(t) for t, s in TOKEN_RE.findall(pm.group(2))]
        model_next[p] = toks
    for pm in re.finditer(r"########## position (\d+) = (.*?) ##########\n(.*?)(?=\n########## position|\Z)", txt, re.S):
        pos = int(pm.group(1)); body = pm.group(3)
        for lm in re.finditer(r"--- layer (\d+) ---\n(.*?)(?=\n--- layer|\Z)", body, re.S):
            L = int(lm.group(1))
            for line in lm.group(2).splitlines():
                m = re.match(r'(J-lens|R-lens|logit)\s+(.*)', line)
                if not m: continue
                lens = m.group(1)
                toks = [eval(t) for t, s in TOKEN_RE.findall(m.group(2))]
                for r, t in enumerate(toks, 1):
                    rows.append((item, cond, pos, L, lens, r, t))
    df = pd.DataFrame(rows, columns=['item', 'condition', 'position', 'layer', 'lens', 'rank', 'token'])
    info = dict(item=item, condition=cond, prompt=prompt, tokens=tokens, entity=entity, final=final,
                model_next=model_next)
    return df, info

def load_all():
    dfs, infos = [], {}
    for p in sorted(glob.glob(os.path.join(RAW, '*.txt'))):
        df, info = parse_run(p); dfs.append(df); infos[(info['item'], info['condition'])] = info
    return pd.concat(dfs, ignore_index=True), infos

def country_rank(df, reg):
    """For each (item, condition, position, layer, lens, country): best rank of any registered form (NaN if absent from top-25)."""
    tok2c = {}
    for c, toks in reg.items():
        for t in toks: tok2c[t] = c
    d = df[df.token.isin(tok2c)].copy()
    d['country'] = d.token.map(tok2c)
    return d.groupby(['item', 'condition', 'position', 'layer', 'lens', 'country'])['rank'].min().reset_index()

if __name__ == '__main__':
    df, infos = load_all()
    print(df.shape)
    print(df.groupby(['item', 'condition']).size())
    df.to_pickle('raw_readouts.pkl')
    json.dump({f'{k[0]}|{k[1]}': v for k, v in infos.items()}, open('run_info.json', 'w'), ensure_ascii=False, indent=1)
    reg = load_registry(); print({k: len(v) for k, v in reg.items()})
