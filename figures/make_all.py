"""Rebuild Figures 1-4 from the raw confirmatory readouts. Run from the figures/ directory:
    python3 make_all.py
Writes raw_readouts.pkl, run_info.json, entity_country_ranks.csv, then out/*.png."""
import json, subprocess, sys, pandas as pd
import parse_raw
df, infos = parse_raw.load_all()
df.to_pickle('raw_readouts.pkl')
json.dump({f'{k[0]}|{k[1]}': v for k, v in infos.items()}, open('run_info.json', 'w'), ensure_ascii=False, indent=1)
cr = parse_raw.country_rank(df, parse_raw.load_registry())
rows = []
for (item, cond), g in cr.groupby(['item', 'condition']):
    rows.append(g[g.position == infos[(item, cond)]['entity']])
pd.concat(rows).to_csv('entity_country_ranks.csv', index=False)
for s in ['fig2.py', 'fig3.py', 'fig4.py', 'fig5_strip.py']:
    print(s); subprocess.run([sys.executable, s], check=True)
