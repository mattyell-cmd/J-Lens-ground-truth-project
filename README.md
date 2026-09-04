# Do J-lens readouts mean what readers take them to mean? A per-claim precision audit against certified ground truth

Application project for MATS 12.0 (Neel Nanda stream), September 2026. Twenty hours, pre-registered.

**Write-up:** https://docs.google.com/document/d/1ZqaXSBElBQoaoQOD_EY_Sfln1XaqESAXaWP53qI7osQ/edit

## What this is

The J-lens and R-lens (Anthropic's global-workspace paper; open fits by camilablank/workspace-lenses for
Qwen3.5-4B) read a model's middle layers as a ranked list of words. Published evaluations measure recall:
plant a concept, check the lens finds it. This project measures precision: when the lens puts a country at
the top of its list, is that the model's own answer? Ground truth is the model's behaviour, certified in
advance ("Which country is {description}? One word or 'none'", 5 samples, >= 4/5), with broken-twin controls
that should produce no answer.

Headline: on "the country where they speak Spanish in North America", the model answers Mexico on every
trial; both lenses commit to Canada across layers 8-17 (R-lens at rank 1 for 8-16). Mexico appears at
rank 1 only at layers 26-28. See the write-up for the argument and the limitations.

## Pre-registration

`prereg_frozen_v0.3.md`, frozen 31 August 2026 at commit `06a3e70`.
SHA-256: `71e47abec9c61b03962daa6cfedf1907b9e239ece9cf8ae1a40409066ea06a63`
One dated amendment (commit `d040f19`) corrected a robustness claim about the commit rule; no rule changed.

## Layout

| path | what |
|---|---|
| `prereg_frozen_v0.3.md` | frozen design, hypotheses, decision rules |
| `CLAUDE.md` | brief under which the coding agent worked (stop points, no rule tuning) |
| `elicit_country.py`, `elicit_ground_truth_v2.py` | ground-truth elicitation (v2 = the "one word or 'none'" format used) -> `results/elicitation_28-08/` |
| `gate_a_config.py`, `gate_b_identity.py`, `gate_c_orientation.py`, `gate_d_rlens.py`, `gate.py`, `shapes.py`, `verify_orientation.py` | pipeline gates run before any readout: tensor shapes, layer-30 lens = identity, multiplication direction, lens file hashes |
| `readout_confirmatory.py` | captures top-25 readouts for J-lens, R-lens and logit lens at every layer 0-30 and position -> `results/confirmatory_01-09/raw/` |
| `score_readouts.py` | the frozen scorer: claims, TP / FP-C / FP-P / UNSCORED, commits, mouth and echo tags, bootstrap -> `scored_claims.tsv`, `summary.json` |
| `test_scorer_synthetic.py`, `results/harness_validation/` | 16 synthetic tests and the validation gate (scorer had to reproduce five pilot verdicts, no tuning) |
| `results/confirmatory_01-09/` | the confirmatory run: raw readouts, scored claims, summary, registry, gates, research log |
| `results/pilot_24-08/`, `pilot_readout*.py`, `readout_boot_sandal.py` | the boot / sandal pilot that fixed the scoring band and the commit rule |
| `fig5.py`, `posthoc_completions.py`, `diagnose.py` | post-hoc inspection scripts (completion sampling was attempted on 3 Sept and did not complete) |
| `figures/` | rebuilds Figures 1-4 of the write-up from the raw files: `cd figures && python3 make_all.py` |
| `session_transcript_2026-09-01.txt` | transcript of the agent session that built and ran the confirmatory pipeline |

## Reproducing

Model: `Qwen/Qwen3.5-4B`. Lenses: `camilablank/workspace-lenses` (Qwen3.5-4B, 25-prompt fit); the
lens file SHA-256s are recorded in each raw file header and in `results/confirmatory_01-09/gates/`.
Run the gates, then `readout_confirmatory.py`, then `score_readouts.py`. Weights and lens files are
not in this repo (`.gitignore`).

## LLM use

Claude (chat) as thinking partner and drafter; Claude Code for the pipeline under `CLAUDE.md`;
Gemini for adversarial feedback. Every experimental ruling was mine; unratified suggestions were flagged
`[ME:]` in the research log and ratified or overruled. The agent caught the self-confirming probe method
(23 Aug). Details in the write-up's appendix.

## Licence

Code: MIT. Results, figures and text: CC BY 4.0.
