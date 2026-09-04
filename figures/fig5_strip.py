"""Fig 5 (strip version) — the headline item at full depth. For each lens, two rows of cells
(Canada = wrong, Mexico = certified answer) x layers 0-30; the number in each cell is the
rank in that layer's readout at the entity position ('America'). Same colour bins as Fig 4."""
import json, numpy as np, pandas as pd
from style import *

ea = pd.read_csv('entity_country_ranks.csv'); infos = json.load(open('run_info.json'))
LAYERS = np.arange(0, 31); prompt = infos['spanishNA|working']['prompt']

def ranks_for(country, lens):
    g = ea[(ea.item == 'spanishNA') & (ea.condition == 'working') & (ea.country == country) & (ea.lens == lens)]
    return g.set_index('layer')['rank'].reindex(LAYERS).values

def bin_of(r):
    if np.isnan(r): return None
    return 2 if r <= 3 else (1 if r <= 10 else 0)

ROWS = [('Canada', RED_BINS, 'Canada \u2014 wrong'), ('Mexico', GREEN_BINS, 'Mexico \u2014 the certified answer')]
LENSES = ['J-lens', 'R-lens']
CELL = 0.245
LEFT, RIGHT, TOP, BOTTOM, GAP = 2.05, 0.15, 1.05, 0.98, 0.55
BAND_FILL, BAND_EDGE = '#C9DCF3', '#2F6DB5'
n_c = len(LAYERS)
fig_w = LEFT + n_c * CELL + RIGHT
fig_h = TOP + BOTTOM + len(LENSES) * (2 * CELL) + (len(LENSES) - 1) * GAP
fig = plt.figure(figsize=(fig_w, fig_h))

for k, lens in enumerate(LENSES):
    y_in = BOTTOM + (len(LENSES) - 1 - k) * (2 * CELL + GAP)
    ax = fig.add_axes([LEFT / fig_w, y_in / fig_h, n_c * CELL / fig_w, 2 * CELL / fig_h])
    ax.set_xlim(0, n_c); ax.set_ylim(2, 0); ax.set_aspect('equal')
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_yticks([])
    for i, (country, bins, label) in enumerate(ROWS):
        r = ranks_for(country, lens)
        for j, L in enumerate(LAYERS):
            b = bin_of(r[j])
            if b is None:
                fc, tc, txt, fs, fw = ABSENT_FILL, '#A8A8A8', '\u00b7', 8, 'normal'
            else:
                fc = bins[b]; tc = 'white' if b == 2 else INK; txt = f'{int(r[j])}'; fs = 8.6; fw = 'bold' if b == 2 else 'normal'
            ax.add_patch(matplotlib.patches.Rectangle((j, i), 1, 1, fc=fc, ec=BAND_FILL if 8 <= L <= 20 else 'white', lw=1.6, zorder=2))
            ax.text(j + 0.5, i + 0.5, txt, ha='center', va='center', fontsize=fs, color=tc, fontweight=fw, zorder=3)
        ax.text(-0.25, i + 0.5, label, ha='right', va='center', fontsize=9.2, color=bins[2])
    # frozen band: tinted wash behind the cells + coloured border
    ax.add_patch(matplotlib.patches.Rectangle((8 - 0.22, -0.34), 13 + 0.44, 2.68, fc=BAND_FILL, ec=BAND_EDGE, lw=1.6, zorder=0, clip_on=False))
    ax.text(-0.25, -0.36, LENS_LABEL[lens], ha='right', va='bottom', fontsize=11, fontweight='bold', color=LENS[lens])
    if k == 0:
        ax.text(14.5, -0.42, 'frozen scoring band (layers 8\u201320)', ha='center', va='bottom', fontsize=9, color=BAND_EDGE, fontweight='bold')
    if k == len(LENSES) - 1:
        ax.set_xticks(np.arange(0, 31, 2) + 0.5); ax.set_xticklabels([str(x) for x in range(0, 31, 2)], fontsize=9)
        ax.tick_params(axis='x', length=3, colors='#444444'); ax.set_xlabel('layer', fontsize=10.5)
    else:
        ax.set_xticks([])

fig.text(0.02, 1 - 0.22 / fig_h, '\u201c' + prompt + '\u201d', ha='left', va='top', fontsize=11, fontweight='bold')
fig.text(0.02, 1 - 0.50 / fig_h, 'read at \u201cAmerica\u201d  \u00b7  model answers Mexico  \u00b7  number = rank in that layer\u2019s readout (1 = top)',
         ha='left', va='top', fontsize=9.4, color=MUTED)

# legend
def swatch(x_in, y_in, fc, ec='none', lw=0):
    fig.patches.append(matplotlib.patches.Rectangle((x_in / fig_w, y_in / fig_h), 0.2 / fig_w, 0.2 / fig_h, fc=fc, ec=ec, lw=lw,
                                                    transform=fig.transFigure, figure=fig))
y = 0.16
x = LEFT - 0.9
for lab, b in [('rank 1\u20133', 2), ('rank 4\u201310', 1), ('rank 11\u201325', 0)]:
    swatch(x, y, RED_BINS[b]); swatch(x + 0.24, y, GREEN_BINS[b])
    fig.text((x + 0.54) / fig_w, (y + 0.10) / fig_h, lab, ha='left', va='center', fontsize=8.8); x += 1.45
swatch(x, y, ABSENT_FILL); fig.text((x + 0.30) / fig_w, (y + 0.10) / fig_h, '\u00b7 = not in the top-25', ha='left', va='center', fontsize=8.8); x += 1.65
fig.patches.append(matplotlib.patches.Rectangle((x / fig_w, y / fig_h), 0.2 / fig_w, 0.2 / fig_h, fc=BAND_FILL, ec=BAND_EDGE, lw=1.6, transform=fig.transFigure, figure=fig))
fig.text((x + 0.30) / fig_w, (y + 0.10) / fig_h, 'frozen scoring band', ha='left', va='center', fontsize=8.8)
fig.savefig('out/fig5_spanishNA_full_depth.png'); print('saved', fig_w, fig_h)
