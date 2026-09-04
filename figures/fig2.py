"""Fig 2 — for each working item, the rank of the country the lenses track, at the entity
position, across the frozen band (layers 8-20), for J-lens, R-lens and the logit lens."""
import json, numpy as np, pandas as pd
from style import *

ea = pd.read_csv('entity_country_ranks.csv'); infos = json.load(open('run_info.json'))
BAND = np.arange(8, 21)

# (item, tracked country, model's certified answer, verdict, tag text)
PANELS = [
    ('boot',       'Italy',  'Italy',    'correct', 'positive control'),
    ('spanishNA',  'Canada', 'Mexico',   'wrong',   'WRONG'),
    ('frenchNA',   'Canada', 'Canada',   'correct', 'correct'),
    ('dutchSA',    'Brazil', 'Suriname', 'wrong',   'WRONG'),
    ('portuguese', 'Brazil', 'Brazil',   'correct', 'correct, weak'),
    ('parallel38', 'Korea',  'Korea',    'none',    'nothing'),
]

def ranks_for(item, country, lens):
    g = ea[(ea.item == item) & (ea.condition == 'working') & (ea.country == country) & (ea.lens == lens)]
    r = g.set_index('layer')['rank'].reindex(BAND)
    return r.values

fig, axes = plt.subplots(3, 2, figsize=(10.4, 10.6), sharex=True, sharey=True)
for ax, (item, tracked, answer, verdict, tag) in zip(axes.ravel(), PANELS):
    prompt = infos[f'{item}|working']['prompt']
    rank_axis(ax, absent=False, label=False)
    commit_zone(ax, verdict, 8, 20)
    for lens in ['J-lens', 'R-lens']:
        plot_rank_line(ax, BAND, ranks_for(item, tracked, lens), lens, absent=False)
    # panel header: prompt (2 lines) + what is being tracked
    head = '\u201c' + wrap(prompt, 46) + '\u201d'
    sub = f'model answers {answer}  \u00b7  lens tracks {tracked}'
    ax.set_title(head + '\n' + sub, loc='left', fontsize=9.6, pad=6, linespacing=1.35)
    # verdict pill, top right inside the axes
    ax.text(1.0, 1.035, tag, transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            color='white' if verdict != 'none' else INK, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', fc=LINE[verdict], ec='none', alpha=0.95), zorder=6)
    ax.set_xlim(7.5, 20.5); ax.set_xticks(range(8, 21, 2))
    ax.tick_params(length=3)

for ax in axes[-1]: ax.set_xlabel('layer')
for ax in axes[:, 0]: ax.set_ylabel('rank of the tracked country\n(1 = top of list, log scale)')
fig.legend(handles=lens_handles(lenses=['J-lens', 'R-lens'], absent=False, zone=True, gaps=True), loc='lower center', ncol=4,
           frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.6, columnspacing=1.6)

# column headers
fig.text(0.29, 0.985, 'lens reads the certified answer', ha='center', va='top', fontsize=12, fontweight='bold', color=GREEN)
fig.text(0.76, 0.985, 'lens reads a wrong country, or nothing', ha='center', va='top', fontsize=12, fontweight='bold', color=RED)
fig.tight_layout(rect=(0, 0.05, 1, 0.975), h_pad=2.6, w_pad=1.6)
fig.savefig('out/fig2_commits_by_item.png')
print('saved')
