"""Fig 3 — top-10 precision on the five working items (boot excluded), and the bootstrap
distribution of that precision when the five items are resampled 10,000 times."""
import numpy as np, pandas as pd, json
from style import *

t = pd.read_csv('../results/confirmatory_01-09/scored_claims.tsv', sep='\t')
s = json.load(open('../results/confirmatory_01-09/summary.json'))
K = 10
w = t[(t.condition == 'working') & (t.is_control == 0) & (t['rank'] <= K) & (t.bucket.isin(['TP', 'FP-C']))]
items = sorted(s['bootstrap_items'])
per_item = w.groupby(['lens', 'item', 'bucket']).size().unstack(fill_value=0).reindex(
    pd.MultiIndex.from_product([LENS_ORDER, items]), fill_value=0)
for b in ['TP', 'FP-C']:
    if b not in per_item: per_item[b] = 0
totals = per_item.groupby(level=0).sum()

# --- bootstrap: resample the 5 items with replacement, 10,000 times, seed 0 ---
rng = np.random.default_rng(s['frozen_settings']['bootstrap_seed'])
n_boot = s['frozen_settings']['n_bootstrap']
draws = rng.integers(0, len(items), size=(n_boot, len(items)))
boot = {}
for lens in ['J-lens', 'R-lens']:
    tp = per_item.loc[lens, 'TP'].values; fp = per_item.loc[lens, 'FP-C'].values
    TP = tp[draws].sum(1); FP = fp[draws].sum(1)
    ok = (TP + FP) > 0
    boot[lens] = TP[ok] / (TP[ok] + FP[ok])
    lo, hi = np.percentile(boot[lens], [2.5, 97.5])
    print(lens, 'point', totals.loc[lens, 'TP'] / (totals.loc[lens, 'TP'] + totals.loc[lens, 'FP-C']),
          'CI', round(lo, 3), round(hi, 3), 'valid', ok.sum(), 'share at 1.0', round((boot[lens] == 1).mean(), 3))
print(per_item)

SPIKE_NOTE = {
    'J-lens': 'the spike at 1.0 = resamples that never draw the Spanish-NA item,\nwhich holds every false claim',
    'R-lens': 'the spike at 0 = resamples that never draw the French-NA item,\nwhich holds every true claim',
}
fig = plt.figure(figsize=(10.6, 6.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.45], hspace=0.32, wspace=0.28, left=0.07, right=0.985, top=0.92, bottom=0.10)
axb = fig.add_subplot(gs[:, 0]); axh = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]

# --- (a) stacked bars ---
x = np.arange(3); wd = 0.6
for i, lens in enumerate(LENS_ORDER):
    tp, fp = totals.loc[lens, 'TP'], totals.loc[lens, 'FP-C']
    axb.bar(i, tp, wd, color=GREEN, zorder=3)
    axb.bar(i, fp, wd, bottom=tp, color=RED, zorder=3)
    prec = tp / (tp + fp)
    axb.text(i, tp + fp + 1.2, f'{tp} true / {tp + fp} claims\nprecision {prec:.2f}', ha='center', va='bottom', fontsize=10, linespacing=1.3)
    axb.text(i, -3.0, LENS_LABEL[lens], ha='center', va='top', fontsize=11, color=LENS[lens], fontweight='bold')
axb.set_xticks([]); axb.set_xlim(-0.6, 2.6); axb.set_ylim(0, 66)
axb.set_ylabel('claims made in the top-10\n(five working items, entity position, layers 8\u201320)')
axb.yaxis.grid(True, color=GRID, zorder=0); axb.set_axisbelow(True)
axb.spines['bottom'].set_visible(False)
axb.legend(handles=[matplotlib.patches.Patch(fc=GREEN, label='true \u2014 the certified answer'),
                    matplotlib.patches.Patch(fc=RED, label='false \u2014 a different country')],
           loc='upper right', frameon=False)
axb.set_title('a.  How many claims, how many wrong', loc='left', fontweight='bold', fontsize=11)

# --- (b, c) bootstrap histograms ---
bins = np.arange(-0.0125, 1.0126, 0.025)
for ax, lens in zip(axh, ['J-lens', 'R-lens']):
    v = boot[lens]; col = LENS[lens]
    ax.hist(v, bins=bins, color=col, zorder=3)
    lo, hi = np.percentile(v, [2.5, 97.5]); pt = totals.loc[lens, 'TP'] / (totals.loc[lens, 'TP'] + totals.loc[lens, 'FP-C'])
    share1 = (v == 1).mean()
    ax.axvline(pt, ymax=0.57, color=INK, lw=1.0, ls=(0, (3, 2)), zorder=4)
    ymax = ax.get_ylim()[1] * 1.95; ax.set_ylim(0, ymax)
    # annotation stack at the top of the panel: lens + observed value / note / 95% bracket
    ax.text(0.0, ymax * 0.99, f'{LENS_LABEL[lens]}  \u2014  observed precision {pt:.2f} (dashed line)', ha='left', va='top',
            fontsize=10.5, fontweight='bold', color=col)
    ax.text(0.0, ymax * 0.86, SPIKE_NOTE[lens], ha='left', va='top', fontsize=8.8, color=MUTED, linespacing=1.3)
    yb = ymax * 0.62
    ax.annotate('', xy=(lo, yb), xytext=(hi, yb), arrowprops=dict(arrowstyle='|-|', color=MUTED, lw=1.0, mutation_scale=4))
    ax.text(lo + 0.01, yb + ymax * 0.015, f'95% interval [{lo:.2f}, {hi:.2f}]', ha='left', va='bottom', fontsize=9, color=MUTED)
    # percentage on the two end spikes
    for val in (0.0, 1.0):
        c = (v == val).sum()
        if c / len(v) >= 0.05:
            top = np.abs(v - val) <= 0.03
            ytop = np.histogram(v[top], bins=bins)[0].max()
            ax.text(val, ytop + ymax * 0.012, f'{c / len(v):.0%}', ha='center', va='bottom', fontsize=9, color=col, fontweight='bold')
    ax.set_xlim(-0.03, 1.03); ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=3)
    ax.set_ylabel('resamples'); ax.set_yticks([0, 1000, 2000, 3000])
axh[0].set_title('b.  Precision when the five items are resampled 10,000 times', loc='left', fontweight='bold', fontsize=11)
axh[1].set_xlabel('top-10 precision in the resample')
fig.savefig('out/fig3_precision_bootstrap.png'); print('saved')
