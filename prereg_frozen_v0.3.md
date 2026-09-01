Experiment pre-reg doc: for each test, what am i testing and why.
Draft v3, not frozen
Freeze procedure: read doc, make edits, commit to repo and record commit hash and date. Changes beyond this are added to bottom as dated amendments.
Freeze is to ensure experiment asks a specific question with a specific answer and to ensure claims and theoretical explanations are based on data and are not cherry picked or worked out post-hoc. 


Lens readouts seen before freeze not included in confirmatory analysis. Behavioural elicitation (prompting and reading answers) does not count, lens output does


Version history: v0.1 (24/08) drafted before the pilot. v0.2 (28/08) replaced the item design after the pilot invalidated the original word-association approach, and ratified all open decisions. v0.3 (31/08) is a readability redraft of v0.2 with no substantive changes.


________________


1. Questions and hypotheses 
Overall direction- Does J-lens or R-lens actually measure ground truth (based on what the model knows)?


Primary question- When a lens reads out a token at a given layer, is that readout true? Existing evaluations measure recall whether the lens finds things known to be there. I want to measure ground-truth and precision of the J-lens and R-lens against the knowledge the model has. Specifically, I will measure per-claim precision for the J-lens and R-lens, against the logit lens as a baseline.
  

Secondary question- Does a lens read the layer it points at, or does it anticipate later layers? (StellaAthena’s q in R-lens lesswrong)
 I answer this by comparing when the lens first shows an answer against when a simple probe can first find that answer in the same activations. This analysis runs strictly after the precision analysis is complete — if time runs out, it degrades to discussion only.


Hypotheses
* H-1 (primary): precision will be measurably below 1 and will differ between the J and R lenses. Confirmed if the 95% bootstrap CI for band-level precision excludes 1; the J/R difference confirmed if the 95% CI for the (J − R) difference excludes 0.
* H-2: Both J and R precision will be higher than logit-lens precision. Confirmed per lens if the 95% bootstrap CI for (lens − logit) precision difference (excludes 0 (0 = no advantage over the baseline; the interval excluding it rules that out). If the logit lens produces fewer than 10 scorable claims in the band, the comparison is reported descriptively
* H-3 (tao-hpu): tao-hpu report that true covert claims tend to be context registers (what kind of text this is) rather than content plans (what the answer will be). A true, covert content hit on my unseen items, the boot→Italy pattern, is a counterexample to that; scarcity of such hits corroborates it. I will report either outcome.
* H-4 (Ratnaditya): Every claim is either an input echo (matches token from prompt) or novel (doesn’t match). Prediction: the true-positive rate among novel claims will exceed the positive rate among echo claims, confirmed if the 95% bootstrap CI for the difference in novel - echo truth rates excludes 0. If confirmed, free echo check is useful proxy for expensive ground truth check. (fewer than 10 echo or novel claims → descriptive)


* Directional expectations: R-lens precision above J-lens in the analysis band (per the R-lens post), precision rising through the band.


Scope Every item resolves to a country. 
Reasoning 
1. Countries are a closed set (~200), so scattered (samples name different referrents) vs committed (samples converging on one) can be easily recognised; an open answer would make scattered answers that are correct (i.e. synonyms) impossible to disentangle. 
2. Every item is the same kind of inference, so difficulty is matched across items
3. A fixed label space makes the skip-ahead probe trainable. Whether results generalize beyond country-referents is out of scope  (future work). 
2. Apparatus (fixed)
* Model: Qwen3.5-4B. Config details verified before each scored run.
* Lenses: the matched J + R pair from https://huggingface.co/camilablank/workspace-lenses/tree/main
* File hashes recorded at the pipeline gate. Logit lens (J = identity) as baseline.
* Layer convention: "layer L" means the residual stream at L, transported to layer 30 (the penultimate layer; J[30] = identity by construction), then decoded through the unembedding. Every figure caption states this.
* Pipeline gates, all before any scored run: tensor details match config; J[30] ≈ identity; orientation check; and the boot positive control (section 4, below) reproduces its expected pattern. If the control fails, something in the pipeline is broken and nothing else runs.
3. Scoring rules
The unit of analysis is a claim: one token appearing in a lens's top-k list at one (item, position, layer). Precision is computed over claims, not items — five items yield hundreds of claims across layers, positions, and lenses.


* k = 10 primary (comparable to the pass@10 convention in prior evals); k = 25 secondary (the paper's capacity convention).
* Scored positions. The pilot showed a division of labor: the intermediate answer (the country) appears at the entity position — the final token of the description — while the final answer appears at the last prompt position. I score the country at the entity position, and the downstream answer (where an item has one) at the final position, reported separately.
* Buckets. Each claim lands in exactly one: True-positive (TP) (in the item's expected set), False-positive paired (FP-P) (a country commit on that item's broken twin — see 4b), False-positive cross (FP-C) (in a different item's expected set, e.g. Brazil tokens on the Korea item), or UNSCORED (none of the above). Ambiguous tokens are never forced into TP or FP. Cross-item exclusion is the stricter test and is primary.
* Precision = TP / (TP + FP), per layer × lens × condition. Per-layer scoring is primary; best-rank-in-band is secondary.
* Analysis band: layers 8–20 (ratified 28/08). The pilot showed the Italy signal starting at L8–11 (below the originally drafted floor of 12), a context-echo cluster at L22–25, and self-echo at L26–28. The run-1 audit confirmed both echo clusters sit outside this band.
* Surface forms. A country's expected set includes all its registered spellings: leading-space and capitalization variants, demonyms, and non-Latin forms (Italy registers 意大利; elicitation returned "Nederland" for Dutch, confirming the need). The full lists are fixed in the item table at freeze. Multi-token forms are registered by their first token and flagged.
* Fragment filter (ratified 28/08). A Latin-script token counts as naming a country only if it starts with a space or a capital letter. Bare lowercase fragments — 'oman' (from woman, Roman), 'erman' (from German) — go to UNSCORED. Without this rule, sub-word debris would hand false FPs disproportionately to the logit-lens baseline, an error that flatters the lenses under test. Validated on run 1: logit L8 'oman' correctly excluded.
* Commit versus smear (ratified 28/08). This operationalizes what counts as a false positive on twins. A lens commits to a country at a position if any registered form of that country appears at rank ≤ 3 for at least 3 consecutive band layers. Forms may differ across layers — a streak carried by 意大利 at one layer and ' Ital' at the next still counts. Anything weaker is a smear. Validated against run 1, where the right answers were known from hand-reading: J-lens commits to Italy at L9–11 ✓, R-lens at L8–11 ✓, logit lens never (no Italy form in any top-10) ✓ — the rule mechanically reproduces the pilot verdict, and is robust to tightening the rank threshold to 2.
* Rare-token exclusion: resolved, none applied. v0.1 conditioned an exclusion rule on a pilot token-frequency diagnostic. That diagnostic was not run (confirmed from the pod record, 28/08). Per the pre-agreed branch: no exclusion, recorded as a limitation. Rare-token junk, if present, is absorbed by the fragment filter and the UNSCORED bucket.
4. Items
4a. Design and selection criteria
Each item is a triple: a working description, a broken twin, and a bare-mention control.


* The working description must require runtime resolution: the underlying fact must be one the model demonstrably has (verified by elicitation, below), while the phrasing must be a framing the model has plausibly never seen as a completable string — inverse or compositional. The model has to work the answer out in the forward pass, not autocomplete it. Excluded on principle: stored-string mappings — nicknames, epithets, symbol lookups, canonical taught sentences.
* The twin must be search-sustaining: a question the model can genuinely attempt — hold the property, search for referents — and fail. Not one it can reject without looking ("surrounded by South Korea" fails this test; "shaped like a sandal" passes). Twins are minimal lexical swaps: matched part of speech, comparable frequency, tokenization checked on the pod and logged.
* Two twin flavors, both permitted, with separate FP definitions:
   * Empty-search (sandal, octagon, 83rd parallel): a valid question whose search returns nothing. FP = a commit (per the section-3 rule) to any country.
   * Forced-choice (Spanish in South America): a valid question with many possible referents. FP = a commit to a single country while the model's own behavior scatters. Bonus analysis: whether the lens's smear overlaps the behaviorally valid referent set — matching uncertainty, not just matching answers.
4b. Ground truth comes from the model, not from me
The pilot proved my knowledge is the wrong reference: under free association the model says boot->Germany; under the referential task, boot->Italy. Elicitation later showed the model lacks "hexagon->France" entirely. Ground truth is therefore task-conditional and model-relative, and I measure it directly:


* Template: "Which country is {description}? Answer with one word, or 'none'." Grammar adjusted per item; the one-word-or-'none' instruction identical everywhere. The 'none' option is essential — the pilot's forced-choice format manufactured associations (cube->Cuba 6/7 by phonology) because non-commitment was unexpressible.
* 5 samples per prompt; sampling settings taken from the pilot script (logged); greedy decoding prohibited, because variation between samples is what distinguishes commitment from scatter.
* Working items need one country in >=4 of 5 samples; that country's surface forms become the expected set. Below threshold = the model lacks the fact = item excluded (familiarity screen).
* Twins must scatter or answer 'none'. If a "broken" twin instead resolves at >=4/5, the model's answer becomes ground truth and the item is recategorized (logged per item at vetting).
* Bare-mention contamination screen: "{bare phrase}. Which country comes to mind? Answer with one word, or 'none'." If the bare phrase alone elicits the target at >=4/5, the working/bare contrast cannot exist and the item is excluded.
* All raw completions kept verbatim, including non-English forms. I vetted every row personally before this table was filled.
4c. The item table (filled 31/08 from vetted elicitation, rounds 1 and 2)
Confirmatory set — n = 5 (pre-registered floor of 4: met; authoring closed):


id
	working description
	elicited answer
	twin
	flavor
	twin result
	bare screen
	parallel38
	divided by the 38th parallel
	Korea 5/5
	83rd parallel
	empty-search
	none 5/5
	pass (scatter)
	portuguese
	speak Portuguese in South America
	Brazil 5/5
	Spanish in S. America
	forced-choice
	scatter
	pass (bare->Portugal, not target)
	frenchNA
	speak French in North America
	Canada 5/5
	French in S. America
	empty-search
	scatter
	pass (bare->France 5/5, not target)
	spanishNA
	speak Spanish in North America
	Mexico 5/5
	Spanish in East Asia
	empty-search
	none 5/5
	pass (bare->Spain 3, Mexico 2 — under threshold)
	dutchSA
	speak Dutch in South America
	Suriname 5/5
	Danish in S. America
	empty-search
	none 5/5
	pass (bare->Nederland, not target)
	

Expected-set surface forms (from elicitation plus tokenizer check; space/capital variants implied): Korea -> {Korea, Korean, 韩国, 朝鲜 — both Chinese forms ratified 28/08, since the English answer "Korea" doesn't distinguish North from South and the lens may surface either}; Brazil -> {Brazil, Brazilian, Brasil, 巴西}; Canada -> {Canada, Canadian, 加拿大}; Mexico -> {Mexico, Mexican, México, 墨西哥}; Suriname -> {Suriname, Surinamese, 苏里南}.


Positive control: boot. Working: Italy 5/5. Twin (sandal): scatter. Bare: none 4/5 — which is also the legitimate-format replication of the pilot's no-Italy-at-bare-mention result. I viewed boot's lens readouts during the 24/08 pilot, so the firewall rule strikes it from confirmatory aggregates. It runs first in the confirmatory session as a pipeline gate, with a pre-registered expectation: an Italy commit at the entity position within the band, per the pilot and the ratified commit rule. Reported separately, never pooled.


Excluded items, with reasons — the full record:


Excluded items, with reasons. Nine candidate items were dropped, each by a pre-stated rule. Full elicitation data for every item is in the committed TSVs (results/elicitation_28-08/).
* Seven items failed the familiarity screen — asked the working question five times, the model never gave any country in at least 4 of 5 samples, meaning it simply does not have the fact the item relies on. Notably, several of these facts are common knowledge to a Western reader (France's hexagonal shape; Sri Lanka as the "teardrop" island; Vietnam's S-shape): the model, trained on a Chinese-heavy corpus, doesn't share them. This is itself a finding — what counts as a "well-known fact" is relative to the model's training data, not the evaluator's culture — and it is why ground truth here is measured from the model rather than assumed. (One entertaining case: asked which country is shaped like a rooster — a standard description of China's map in Chinese schooling — the model twice answered France, whose national symbol is a rooster. It answered with its strongest association, not the question's category.)
* Two items were caught by the bare-contamination screen — the bare phrase alone, with no question attached, elicited the target country in >=4 of 5 samples ("down under" -> Australia; "cherry blossoms" -> Japan), so a lens readout on these could reflect mere word-association rather than task-driven inference. Both items were included expecting this: they are nickname/symbol items of exactly the type the selection criteria exclude, entered deliberately to test whether the screen catches what the theory says it should. It caught both, with the predictions logged before the runs — which is the evidence that the screen can be trusted on the items it passed.
* One item (strait) was dropped by judgment call, logged 28/08. Designed to elicit Morocco ("separated from Spain by a narrow strait"), it instead elicited Portugal 5/5 — acceptable in itself, since ground truth is the model's answer. But the reversed twin elicited Spain 5/5 and the bare phrase alone also elicited Spain 5/5: the model appears to be answering "name X's neighbor" rather than resolving a strait at all. Since the item can't isolate the computation it was built to test, any lens readout on it would be uninterpretable, and it was dropped before any lens run.


Composition, stated plainly as a limitation: four of the five confirmatory items are the same genre — a language crossed with a region. The shape genre died wholesale on the familiarity screen, except the boot control. The set is more homogeneous than designed. The corollary is itself a reportable finding: item survival was decided by the model's training distribution, not my cultural common knowledge — hexagon->France and teardrop->Sri Lanka simply aren't in a Chinese-heavy-trained model (hypothetical explanation). Ground truth is model-relative as well as task-conditional.
5. Baselines (all mandatory)
1. Logit lens — identical scoring with J = identity. The floor any real lens must beat.
2. Mouth-exclusion (following tao-hpu): drop claims that match the model's own top next-token prediction at the readout position, and report "covert" precision — what the lens finds beyond what the model was about to say. Ratified 28/08: top-1 exclusion primary, top-5 secondary. Top-5 risks deleting genuine intermediates that sit near the answer at the final position (' Italian' near ' Euro'); top-1 barely bites at entity positions. Reporting both shows the covert result isn't an artifact of filter strictness.
3. Input-echo: claims matching tokens in the prompt are reported separately; a TP must beat this to count as computation rather than copying.
4. Output-echo curve (Fig-28a style) for context.
6. Primary and secondary analyses
Skip-ahead onset-lag (secondary; runs only after precision is complete).


* Onset definition (ratified 28/08): persistence required. A lens's onset for a country is the earliest band layer where a registered form appears in the top-k at that layer and the next (>=2 consecutive layers; mixed forms count; fragment filter applies). Single-layer flickers ("blips") are excluded from the statistic and reported descriptively — e.g. the R-lens 欧元区 (eurozone, a downstream concept) at L8 rank 10 in run 1, a candidate anticipation exhibit if it recurs on unseen items.
* Probe (ratified 28/08): per-layer logistic regression on the residual activations — deliberately the simplest instrument in the same access class as the lens (a learned direction plus a squash). A more powerful probe could find information the lens can't linearly access, and would wrongly convict the lens of anticipation. L2-regularized (strength by a small held-out sweep, logged); trained on working-item activations labeled with elicited countries; scored only on held-out items, leave-one-item-out — with 2,560 dimensions and 5 items, in-sample separation is meaningless. Detection threshold: held-out accuracy beats chance with a bootstrap 95% CI clear of the chance line, with the same 2-layer persistence rule as the lens, so both clocks carry the same bias and it cancels in the difference.
* Statistic: the distribution of (lens onset − probe onset) per lens. Negative lag = anticipatory readout. Logit-lens onset provides the anticipation floor.
* Bias, stated: persistence and wide small-n CIs both push onsets late, which pushes measured lag positive — against the anticipation conclusion. So a null result here is weak evidence, but a surviving negative lag is strong. The cost is insensitivity to weak single-layer anticipation, noted as a limitation.
* Sequencing guard: no probe training until the precision analysis is fully done. If time fails, skip-ahead reverts to discussion-only.


Secondary and exploratory:


* Forced-choice association map (exploratory, defined before the confirmatory run): if lens FPs occur on twins, compare their content to the 24/08 forced-choice elicitation data — invalidated as ground truth, retained as a map of the model's association gradients — to test whether lens hallucinations track association priors: "leaning toward X" read out as "committed to X."
* Register-versus-content taxonomy: every TP and FP is labeled as content (a claim about the answer itself, e.g. Italy at the boot position) or register (a claim about the character of the text, e.g. its language or format). tao-hpu found that covert lens readouts are almost exclusively register — "reading the room, not the mind." My items are built to carry covert content, so this taxonomy tests their pattern directly: content-heavy true positives are a counterexample; content claims failing while register survives corroborates them, now with a truth dimension their analysis lacked.
* H-4 cross-tab: baseline-failure versus bucket.
* Causal spot-check: ~5 TPs (patch or ablate the activation; expect the answer to degrade) and ~3 FPs (expect nothing), with tao-hpu's final-token-substitution control.
* counts (~5, ~3) are time-budget targets, not statistical thresholds; no rate is claimed from n~8.
* Causal-onset sweep (pre-registered stretch): extend the spot-check to a layer sweep — the earliest layer where patching the working item's entity activation into the twin run moves behavior toward the expected answer is the causal onset. Three onsets (lens, probe, causal) per item form a convergent-validity matrix. Attempted only after precision and onset-lag are complete and drafted; if not reached, the spot-check stands alone. The transplant-success threshold is defined before running, only if reached.
7. Analysis principles
Bootstrap 95% confidence intervals on every headline rate. No hypothesis is called confirmed on point estimates alone; overlapping intervals are reported as overlapping. Every raw readout behind every aggregate is saved and read by me personally. The time log is maintained. The write-up is my prose, with LLM critique rounds.


Decision rules are stated as confidence-interval criteria rather than significance tests; a 95% CI excluding a null value is equivalent to a two-tailed test at α = .05 against it, but the intervals additionally report the magnitudes, which are the quantities of interest.
8. What sits outside this registration
The 24/08 pilot — boot/sandal/bare readouts, band curves, word-association elicitation — is exploratory and predates this registration; its items are struck from confirmatory aggregates (boot survives only as the positive control). The boot->Germany result (7/7 under forced-choice word association) is presented in the write-up as an exhibit under the invalidated format, not as certified ground truth; its legitimate-format replication is the bare-mention screen, which the boot item passed (none 4/5, no Italy). The 28/08 elicitation rounds generate ground truth and sit inside this registration's pipeline, but involved no lens readouts.

________________

AMENDMENTS

Amendment 1 (01/09/2026) — correction to the commit-rule robustness claim in section 3.

Section 3, under "Commit versus smear", states that the commit rule, validated against run 1, "is robust to tightening the rank threshold to 2". That sentence is too strong and is corrected here.

Re-checking run 1 mechanically with the scoring harness (score_readouts.py, results/harness_validation/), the frozen rule — any registered form at rank <= 3 for >= 3 consecutive band layers — reproduces all three pilot verdicts exactly: J-lens commits to Italy at L9-11, R-lens at L8-11, logit lens never. That part of section 3 stands unchanged.

The robustness claim does not. Tightening the rank threshold from 3 to 2:
  * J-lens: unaffected. Its qualifying layers at rank <= 2 are still 9, 10 and 11 ('意大利' at rank 1, 2 and 2), so the commit remains L9-11.
  * R-lens: the commit DISAPPEARS. Its streak is carried at rank exactly 3 at L10 (' italian', behind ' heel' and '不长') and at L11 (' italian', behind ' heel' and '的剑'). At rank <= 2 only L8 and L9 qualify — a run of 2, below the 3-layer minimum.

Corrected sentence: the rule is robust to tightening the rank threshold to 2 for the J-lens verdict, but not for the R-lens verdict, whose streak depends on rank-3 positions at L10 and L11.

Nothing operative changes. The frozen threshold is and remains rank <= 3, which is what the harness runs and what the confirmatory analysis will use. This amendment corrects a factual statement about the pilot data; it does not alter a rule, a threshold, a band, or a hypothesis. It was written before any confirmatory readout was produced.

Consequence to carry into the write-up: the R-lens commit verdict is threshold-sensitive in a way the J-lens verdict is not, and any claim that the commit rule is insensitive to the rank threshold must be qualified accordingly.
