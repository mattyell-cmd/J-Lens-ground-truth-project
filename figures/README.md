# Figures

`python3 make_all.py` (from this directory) rebuilds Figures 1-4 in `out/` from
`../results/confirmatory_01-09/raw/*.txt` and the frozen surface-form registry.

| file in `out/` | figure in the write-up |
|---|---|
| fig2_commits_by_item.png | Figure 1 - rank of the tracked country, layers 8-20, per item |
| fig3_precision_bootstrap.png | Figure 2 - precision bars and bootstrap distributions |
| fig5_spanishNA_full_depth.png | Figure 3 - the headline item at full depth (rank strip) |
| fig4_prototype_grid.png | Figure 4 - best rank per registered country, all working items and twins |

Ranks are recomputed from the raw files with the same 52-form registry the scorer used;
all 105 scored band claims match `scored_claims.tsv`. The bootstrap in Figure 2 uses
`numpy.random.default_rng(0)` and reproduces `summary.json` (9,907 valid resamples).
Requires matplotlib and pandas. Fonts: Lato (OFL) in `fonts/`.
