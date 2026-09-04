"""Fig 4 — at the entity position, in the frozen band (layers 8-20): the best rank reached by
each registered country, taking the better of J-lens and R-lens. Square cells, a value in
every cell. Green = the row's certified answer; red = any other registered country."""
import json, numpy as np, pandas as pd
from style import *

ea = pd.read_csv('entity_country_ranks.csv'); infos = json.load(open('run_info.json'))
s = json.load(open('../results/confirmatory_01-09/summary.json'))
band = ea[ea.layer.between(8, 20) & ea.lens.isin(['J-lens', 'R-lens'])]
best = band.groupby(['item', 'condition', 'country'])['rank'].min()
commits = {(c['item'], c['condition'], c['country']) for c in s['commits']}

COUNTRIES = ['Canada', 'Mexico', 'Brazil', 'Suriname', 'Korea', 'Italy']
ANSWER = {'frenchNA': 'Canada', 'spanishNA': 'Mexico', 'portuguese': 'Brazil', 'dutchSA': 'Suriname', 'parallel38': 'Korea', 'boot': 'Italy'}
ROWS = [  # (group, item, condition)
    ('North America', 'frenchNA', 'working'), ('North America', 'spanishNA', 'working'),
    ('South America', 'portuguese', 'working'), ('South America', 'dutchSA', 'working'),
    ('South America', 'frenchNA', 'twin'), ('South America', 'dutchSA', 'twin'), ('South America', 'portuguese', 'twin'),
    ('East Asia', 'spanishNA', 'twin'),
    ('no continent word', 'parallel38', 'working'), ('no continent word', 'parallel38', 'twin'),
    ('no continent word', 'boot', 'working'), ('no continent word', 'boot', 'twin'),
]

def bin_of(r):
    if np.isnan(r): return None
    return 2 if r <= 3 else (1 if r <= 10 else 0)

n_r, n_c = len(ROWS), len(COUNTRIES)
CELL = 0.60
LEFT, RIGHT, TOP, BOTTOM = 3.75, 0.95, 1.25, 2.45
fig_w, fig_h = LEFT + n_c * CELL + RIGHT + 0.9, TOP + n_r * CELL + BOTTOM
fig = plt.figure(figsize=(fig_w, fig_h))
ax = fig.add_axes([LEFT / fig_w, BOTTOM / fig_h, n_c * CELL / fig_w, n_r * CELL / fig_h])
ax.set_xlim(0, n_c); ax.set_ylim(n_r, 0); ax.set_aspect('equal')
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_xticks([]); ax.set_yticks([])

for i, (group, item, cond) in enumerate(ROWS):
    info = infos[f'{item}|{cond}']; ans = ANSWER[item] if cond == 'working' else None
    for j, country in enumerate(COUNTRIES):
        r = best.get((item, cond, country), np.nan)
        b = bin_of(r)
        if b is None:
            fc, tc, txt = ABSENT_FILL, '#9A9A9A', '>25'
        else:
            fc = (GREEN_BINS if country == ans else RED_BINS)[b]
            tc = 'white' if b == 2 else INK
            txt = f'{int(r)}'
        ax.add_patch(matplotlib.patches.Rectangle((j, i), 1, 1, fc=fc, ec='white', lw=2.2))
        ax.text(j + 0.5, i + 0.5, txt, ha='center', va='center', fontsize=10.5 if b is not None else 8.5,
                color=tc, fontweight='bold' if b == 2 else 'normal')
        if (item, cond, country) in commits:
            ax.add_patch(matplotlib.patches.Rectangle((j + 0.07, i + 0.07), 0.86, 0.86, fc='none', ec=INK, lw=1.6, zorder=5))
    # row label: prompt (wrapped) + what the model answers
    prompt = info['prompt']
    tail = f'model answers {ans}' if cond == 'working' else 'broken twin \u00b7 model answers none'
    if item == 'boot' and cond == 'working': tail += ' (control)'
    ax.text(-0.12, i + 0.5, '\u201c' + wrap(prompt, 46) + '\u201d\n' + tail, ha='right', va='center', fontsize=8.2, linespacing=1.28,
            color=INK, transform=ax.transData)

# group separators + labels
group_bounds = {}
for i, (g, _, _) in enumerate(ROWS): group_bounds.setdefault(g, [i, i])[1] = i
for k, (g, (a, b)) in enumerate(group_bounds.items()):
    if k > 0: ax.plot([-6.0, n_c], [a, a], color='#BBBBBB', lw=0.8, clip_on=False, zorder=6)
    ax.text(n_c + 0.18, 0.5 * (a + b + 1), g, rotation=270, ha='center', va='center', fontsize=8.6, color=MUTED, clip_on=False)

for j, c in enumerate(COUNTRIES):
    ax.text(j + 0.5, -0.12, c, ha='center', va='bottom', fontsize=10.5, fontweight='bold')
fig.text(0.02, 1 - 0.28 / fig_h, 'Best rank reached by each registered country at the entity position, layers 8\u201320, better of J-lens and R-lens',
         ha='left', va='top', fontsize=11, fontweight='bold')
fig.text(0.02, 1 - 0.58 / fig_h, 'A value in every cell: the rank, or \u201c>25\u201d when the country is not in the top-25 at any of the 13 band layers.',
         ha='left', va='top', fontsize=9, color=MUTED)

# legend (figure coordinates, bottom-left)
def swatch(x_in, y_in, fc, ec='none', lw=0):
    fig.patches.append(matplotlib.patches.Rectangle((x_in / fig_w, y_in / fig_h), 0.26 / fig_w, 0.26 / fig_h,
                                                    fc=fc, ec=ec, lw=lw, transform=fig.transFigure, figure=fig))
def ltext(x_in, y_in, txt, **kw):
    fig.text(x_in / fig_w, (y_in + 0.13) / fig_h, txt, ha='left', va='center', fontsize=8.8, **kw)
y0 = BOTTOM - 0.62
for k, (label, bins) in enumerate([('the certified answer', GREEN_BINS), ('a different country', RED_BINS)]):
    y = y0 - k * 0.42
    fig.text((LEFT - 0.15) / fig_w, (y + 0.13) / fig_h, label, ha='right', va='center', fontsize=8.8)
    for m, (lab, b) in enumerate([('rank 1\u20133', 2), ('rank 4\u201310', 1), ('rank 11\u201325', 0)]):
        x = LEFT + m * 1.25
        swatch(x, y, bins[b]); ltext(x + 0.34, y, lab)
y = y0 - 2 * 0.42
swatch(LEFT, y, ABSENT_FILL); ltext(LEFT + 0.34, y, '>25 = not in the top-25 at any band layer')
y = y0 - 3 * 0.42
swatch(LEFT, y, 'none', ec=INK, lw=1.6); ltext(LEFT + 0.34, y, 'boxed = a commit under the frozen rule (rank \u2264 3 for 3 consecutive layers)')

# footnote on countries the registry did not watch
foot = ('Not in the registry, so not counted: on the East-Asia twin, \u201cChinese\u201d / \u201cChina\u201d reach rank 1\u20132 in both lenses across the band '
        '(Japan rank 3); on the Spanish-in-South-America twin, Chile reaches rank 3 and Argentina rank 6 (J-lens, layers 8\u20139).')
fig.text(0.02, 0.10 / fig_h, wrap(foot, 150), ha='left', va='bottom', fontsize=8.2, color=MUTED, linespacing=1.3)
fig.savefig('out/fig4_prototype_grid.png'); print('saved')
