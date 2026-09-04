# Verified results notes — confirmatory run 01/09
Compiled end of day 01/09. Every number below was cross-checked twice: computed independently from scored_claims.tsv AND read from the official summary.json on the pod. Matt personally viewed the commits list, bucket_counts, and all decision-bearing bootstrap_cis rows. Raw-readout verdicts (spanishNA, parallel38) still pending Matt's personal read — flagged below where relevant.

---

## 1. Sanity checks — all passed
- Boot positive control reproduced the pilot EXACTLY despite new GPU/torch/scorer: J-lens Italy commit L9–11, R-lens L8–11, logit no commit. (Control only; never pooled.)
- Twins and bare: ZERO commits by any lens anywhere → FP-P = 0. Broken referents produced smears, not commitments = correct behaviour = the empty-search precision test passed for both lenses.
- Data-reading gotcha: pandas default settings silently drop ~3,000 rows of scored_claims.tsv (quote chars in tokens). Must use `quoting=csv.QUOTE_NONE` → exactly 29,250 rows, matching the log.

## 2. The commit list (official record, 6 commits total)
| item | lens | country | layers | verdict |
|---|---|---|---|---|
| boot (control) | J-lens | Italy | 9–11 | expected (positive control) |
| boot (control) | R-lens | Italy | 8–11 | expected (positive control) |
| frenchNA | J-lens | Canada | 8–20 (whole band) | TP — textbook success |
| frenchNA | R-lens | Canada | 8–20 (whole band) | TP — textbook success |
| **spanishNA** | **J-lens** | **Canada** | **8–13** | **FALSE POSITIVE — expected Mexico** |
| **spanishNA** | **R-lens** | **Canada** | **8–17, reaches rank 1** | **FALSE POSITIVE — expected Mexico** |

The central exhibit: model behaviorally certified Mexico 5/5 (says peso); both lenses sustain a rank-≤3 commitment to Canada. Mexico appears exactly twice in the whole item (J-lens, ranks 11 and 25). Mouth tags = 0 → the model was NOT about to say Canada; this is not output-parroting.

## 3. Headline precision (working condition, confirmatory items, boot excluded)
k=10 (primary):
- J-lens: 30 TP / 9 FP → 0.769
- R-lens: 32 TP / 21 FP → 0.604
- logit: 2 TP / 0 FP → 1.000 (n=2, under the 10-claim minimum → descriptive)

k=25 (secondary):
- J-lens: 45/23 → 0.662
- R-lens: 51/37 → 0.580
- logit: 4/0 → 1.000 (n=4 → descriptive)

All 60 FPs are FP-C (cross-item, the prereg's stricter primary test): spanishNA→Canada (J 15, R 26 at k=25) and dutchSA→Brazil smear (J 8, R 11, deep ranks, no commit; mostly drops out at k=10).

## 4. Item-by-item story
- **frenchNA (→Canada):** strongest item. Full-band commit both lenses; even logit surfaces Canada (ranks 6–7, its only TPs).
- **spanishNA (→Mexico):** the headline FP. Committed Canada, both lenses.
- **dutchSA (→Suriname):** Suriname NEVER appears at any rank in any lens. Brazil smear instead (ranks mostly 11–23, no commit).
- **parallel38 (→Korea):** total silence — zero country tokens at the entity position in-band. Lenses show relational vocabulary instead ("north", "between", "dividing", "crossing"). Recall zero → contributes no claims; report descriptively. NOTE: silence is a claim about the entity position + band L8–20; anything seen elsewhere in the raw file = labeled-exploratory. [Pending Matt's raw read.]
- **portuguese (→Brazil):** modest. 5 TPs each J/R, all ranks 4+, no commit.

## 5. The frame-prototype pattern (the interpretive spine)
Across ALL conditions, lens readouts at the entity position track the geographic frame's prototype country, not the resolved referent:
- "…in North America" → Canada (on frenchNA AND spanishNA)
- "…in South America" → Brazil (dutchSA, Danish twin, French-in-SA twin, faintly Portuguese bare)
- "…in East Asia" → Korea smear (Spanish-in-EA twin — uncommitted)
- no continent word (parallel38) → no country at all

Implication: frenchNA's success may run on the same mechanism as spanishNA's failure — the prototype happens to be right there. So even the lens's correct answers can't validate it: right-for-the-wrong-reason looks identical from outside.

Irony worth one sentence: Korea DOES appear in the dataset — as an uncommitted smear on the "Spanish in East Asia" twin — so the lenses can produce Korea tokens; they produce them from the geography cue, not from the description that actually resolves to Korea.

## 6. Hypothesis slate under the frozen decision rules (fully verified)
- **H-1 (precision < 1; J ≠ R): UNMET on both clauses.** Precision CIs [0, 1] for both lenses (J point 0.769, R point 0.604). J−R difference point +0.165, CI [0, 1] includes 0. Cause: n=5 items; FPs concentrated in spanishNA, so item-resampling swings between ~perfect and ~poor (~100/10,000 resamples had zero claims and were dropped; n_valid ≈ 9,900).
  - Descriptive note that survives: J−R lower bound is exactly 0, never negative → J at least as precise as R in >97.5% of resamples. Consistent direction, unconfirmable magnitude.
- **H-2 (lenses beat logit): DESCRIPTIVE** (logit under the pre-registered 10-claim minimum: 2 at k=10, 4 at k=25). Direction actually REVERSED: J−logit CI [−0.47, 0.0], R−logit [−0.65, 0.0] — upper bounds exactly 0, logit at least as precise in essentially every valid resample. Caveat: differences computable only in resamples containing frenchNA (n_valid 6,693), i.e., conditioned on logit's one talkative item. Fair framing = trade-off: the baseline almost never speaks and was never wrong when it did; the lenses speak 10–20× more and pay for it in precision.
- **H-4 (novel > echo truth rate): DESCRIPTIVE** — n_echo_scored = 0 everywhere; all scored claims were novel; the echo comparison cannot be formed. Novel truth rates: J 0.769, R 0.604 (k=10) — identical to precision because everything was novel.
- **Sensitivity analyses:** exact-forms-only rows IDENTICAL to primary (every digit) → "no conclusion rests on a flagged first-token form." Mouth-filter rows identical too → no scored claim was the model's own next token.

## 7. The central claim, exactly calibrated (Claim C framing)
- Claim A (meaningful readable structure in intermediate layers): SUPPORTED — twins scatter, parallel38 goes silent, boot/frenchNA read correctly; lens output tracks prompt semantics lawfully.
- Claim B (readout reports the workspace state): mostly survives, with an asterisk — a ranked list cannot distinguish "committed" from "leaning," so faithful reporting of a weak state still looks like a strong claim.
- **Claim C (readout = what the model has concluded): FAILS, demonstrated.** Certified counterexample: lens says Canada, model says Mexico, repeatedly, confidently, both sides. This was never a property of the lens — it's the reader's inference habit, and it fails measurably when frame-prototype and resolution diverge.
- Existence proof, not a rate: the [0,1] CIs limit how far the RATE generalises; they don't touch the EXISTENCE of the failure mode. One certified counterexample suffices for Claim C.
- Rival reading to flag openly: the lens may be faithfully reporting a transient association the model later corrects. FP under frozen rules either way; the pre-registered causal spot-check is what would separate the fork (spanishNA-Canada commit = the FP target; frenchNA-Canada = the paired TP comparison; final-token-substitution control could distinguish reading-the-frame from reading-the-resolution).

## 8. Coined sentences (use or adapt)
- "These lenses answer 'what is this state near?' — but they get read as answering 'what has the model concluded?' On items where the frame's dominant association and the correct resolution diverge, that gap is measurable, and we measured it."
- "The lenses read the workspace, but the workspace at early layers contains associations, not conclusions — and the readout format erases the difference."
- "A top-k list is a forced-choice format: it cannot say 'diffuse and uncertain.' A mild real leaning gets rendered as rank 1." (= the same demand-characteristic flaw as the invalidated 24/08 forced-choice elicitation — the readout format has the demand characteristic.)
- "J-lens was at least as precise as R-lens in >97.5% of resamples, though the difference CI includes 0."
- "The baseline almost never speaks, and was never wrong when it did; the lenses speak 10–20× more often and pay for it in precision."
- "All scored claims were novel; the echo comparison could not be formed."
- "No conclusion rests on a flagged surface form: the exact-forms-only sensitivity analysis is identical to the primary, to every digit."
- "Logged hours reflect active work; incidental thinking time not counted." (time-log rule)

## 9. Status & remaining
- Backup: jlens_backup_2026-09-01.tar.gz (6.0M, 370 files, scored_claims.tsv + raw/ confirmed inside) verified and downloaded to laptop. Lens .pt files on /root vanish on pod stop — expected; hashes in log; re-download.
- Git: clean through commit 3423464 (session transcript + generator).
- Time: 01/09 logged 2:30. Counting rule stated in log.
- Remaining: (1) Matt's raw read of spanishNA + parallel38, verdicts logged; (2) spot-check go/no-go with fresh eyes — default DROP and describe as next experiment; skip-ahead stays parked; (3) skeleton, claims-first; (4) prose Thu, exec summary last, standalone; (5) "how I used LLMs" note; (6) submit Thu evening, never Fri. Outside 20h: check referee notice status.
