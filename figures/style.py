"""One visual language for Figs 2-5.

Colour rules (used identically in every figure):
  lens identity   J-lens purple, R-lens orange, logit lens grey
  correctness     certified answer = green, wrong country = red, nothing = grey
Rank axis rule:   log scale, 1 at the top, ticks at 1 / 3 / 10 / 25, an 'absent'
                  row below 25 for layers where the country is not in the top-25.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import textwrap

for f in ['Lato-Regular', 'Lato-Bold', 'Lato-Italic', 'Lato-BoldItalic']:
    font_manager.fontManager.addfont(f'fonts/{f}.ttf')

plt.rcParams.update({
    'font.family': 'Lato',
    'font.size': 11,
    'axes.titlesize': 11,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#444444',
    'axes.linewidth': 0.8,
    'xtick.color': '#444444',
    'ytick.color': '#444444',
    'text.color': '#222222',
    'axes.labelcolor': '#222222',
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 220,
})

# ---- palette -------------------------------------------------------------
LENS = {'J-lens': '#5B4BB8', 'R-lens': '#E07B24', 'logit': '#8C8C8C'}
LENS_LABEL = {'J-lens': 'J-lens', 'R-lens': 'R-lens', 'logit': 'logit lens'}
LENS_MARKER = {'J-lens': 'o', 'R-lens': 's', 'logit': '^'}
LENS_ORDER = ['J-lens', 'R-lens', 'logit']

GREEN = '#2E8B57'      # certified answer / true claim
RED = '#C0392B'        # wrong country / false claim
GREY = '#8C8C8C'       # nothing / unscored
FILL = {'correct': '#DDEFE4', 'wrong': '#F8DFDC', 'none': '#EDEDED'}
LINE = {'correct': GREEN, 'wrong': RED, 'none': GREY}
GRID = '#E6E6E6'
INK = '#222222'
MUTED = '#666666'

# three shades for rank bins, light -> dark, per hue (used by Fig 4)
GREEN_BINS = ['#BFE3CC', '#6DB98A', '#2E8B57']
RED_BINS = ['#F4C2BC', '#E07A6E', '#C0392B']
ABSENT_FILL = '#F1F1F1'

ABSENT_Y = 50          # where 'absent' markers sit on the log rank axis

def wrap(s, width):
    return '\n'.join(textwrap.wrap(s, width))

def rank_axis(ax, absent=True, label=True):
    """Log rank axis: 1 at the top; 1 / 3 / 10 / 25 ticks; absent row."""
    ax.set_yscale('log')
    ax.set_ylim(ABSENT_Y * 1.6 if absent else 32, 0.78)
    ticks = [1, 3, 10, 25] + ([ABSENT_Y] if absent else [])
    ax.set_yticks(ticks)
    ax.set_yticklabels(['1', '3', '10', '25'] + (['absent'] if absent else []))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for t in [1, 3, 10, 25]:
        ax.axhline(t, color=GRID, lw=0.8, zorder=0)
    if absent:
        ax.axhline(ABSENT_Y, color=GRID, lw=0.8, ls=(0, (2, 2)), zorder=0)
    if label:
        ax.set_ylabel('rank of the country in the readout\n(1 = top of list, log scale)')

def commit_zone(ax, verdict, x0, x1):
    """Shade ranks 1-3 (where a commit can happen) in the verdict colour."""
    ax.axhspan(0.78, 3.4, xmin=0, xmax=1, color=FILL[verdict], zorder=0, lw=0)

def plot_rank_line(ax, layers, ranks, lens, lw=2.0, ms=6, zorder=3, label=None, absent=True):
    """ranks: array with NaN where the country is absent from the top-25.
    Present layers are drawn as a line; absent layers as small hollow markers on the absent row."""
    layers = np.asarray(layers); ranks = np.asarray(ranks, dtype=float)
    col = LENS[lens]; mk = LENS_MARKER[lens]
    present = ~np.isnan(ranks)
    # line only through consecutive present layers
    ax.plot(np.where(present, layers, np.nan), ranks, '-', color=col, lw=lw, zorder=zorder)
    ax.plot(layers[present], ranks[present], mk, color=col, ms=ms, zorder=zorder + 1,
            label=label if label else LENS_LABEL[lens])
    if absent and (~present).any():
        off = {'J-lens': 0.80, 'R-lens': 1.0, 'logit': 1.25}[lens]   # three thin absent rows
        ax.plot(layers[~present], np.full((~present).sum(), ABSENT_Y * off), mk, mfc='white',
                mec=col, mew=0.9, ms=ms * 0.62, alpha=0.9, zorder=zorder)

def lens_handles(absent=True, zone=True, lenses=None, gaps=False):
    h = [matplotlib.lines.Line2D([], [], color=LENS[l], marker=LENS_MARKER[l], lw=2, ms=6,
                                 label=LENS_LABEL[l]) for l in (lenses or LENS_ORDER)]
    if gaps:
        h.append(matplotlib.lines.Line2D([], [], color=MUTED, lw=0, label='no marker = not in the top-25 at that layer'))
    if absent:
        h.append(matplotlib.lines.Line2D([], [], color=MUTED, marker='o', mfc='white', lw=0, ms=4.5,
                                         label='hollow = not in the top-25 at that layer'))
    if zone:
        h.append(matplotlib.patches.Patch(fc='#E4E4E4', ec='none', label='shaded = commit zone (rank \u2264 3)'))
    return h

def lens_legend(ax, loc='lower left', **kw):
    return ax.legend(handles=lens_handles(absent=False, zone=False), loc=loc, frameon=False, handlelength=1.6, **kw)
