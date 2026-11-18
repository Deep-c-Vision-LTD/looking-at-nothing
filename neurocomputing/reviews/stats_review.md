# Independent review of the re-analysis rule: statistics and honesty

- Reviewer: independent statistics and honesty review, 18 November 2026 (about 10:20 to 10:50 UTC). I did not write any of the reviewed files.
- Files reviewed (SHA-256 at review time):
  - `neuro/reanalysis/REANALYSIS_RULE.md` b5a164a2927cb929549b6bde9c43f1e876bf08597942f7071f98e9760f95ac2a
  - `neuro/reanalysis/analyze_re.py` 472ae9e35faa3936476f12f697a8cdcbcf768e70a44a13def168c7cabc6d01c8 (equals the hash printed in `tests_output.txt`)
  - `neuro/reanalysis/test_analyze_re.py` 539d5c8814d9076a9d1e38b7745ac8c7fb9ef73fd09312a136b863bef43ffbb6
  - `neuro/reanalysis/tests_output.txt` 7714621bac5e65eb506fdec91ce66a168df757e331ae7253b4e111cc7acf903f
  - `neuro/reanalysis/RF_RULE_CORRECTIONS.md` 9f7cc356d322cfda806ca1cc4c0b04f9521b18001b7901f7dfbf16634d22ff1c
  - `neuro/REWRITE_PLAN.md` (Critic section K.1 to K.8), plus `RUN_PLAN.md`, `fix_rule.py`, `extract_known.py` for context.
- What I ran (CPU, `nice -n 10`, 4 threads, numpy and scipy only, no torch, no GPU, no scoring of any model or null, no statistic of the rule on any real score):
  - `review_rule/verify_known.py` and `review_rule/compare_tables.py`: read the result JSONs and compare every number of the rule's Section 1.1 tables and of the Section 5.4 table (arithmetic on published values only).
  - `review_rule/independent_check.py` (output `independent_check_output.txt`, 91 s): independent wild cluster bootstrap and BCa on synthetic data, compared with `analyze_re.py`; coverage under three DGPs that differ from the test file's; the G = 6 p-value bound; Holm; title-rule edge cases on synthetic cells.
  - `review_rule/g4_check.py` (output `g4_check_output.txt`): coverage with 4 clusters (exploratory item A10).
  - Read-only git queries on `looking-at-nothing` (log, reflog, ls-files, rev-parse).
- Housekeeping: my first run imported `analyze_re` under `python -I`, which ignores `PYTHONDONTWRITEBYTECODE` and wrote `reanalysis/__pycache__/analyze_re.cpython-312.pyc`. I moved that folder to `neuro/_trash/pycache_review_stats_20261118/` so that it cannot be pushed with the rule. The folder `review_rule/checks/` is not mine (another reviewer); I did not touch it.

## Verdict

All numbers the rule cites as known match the result files (37 table rows, the 18-row P4 table, every registers, sweep, token, receptive-field and test-output number I checked; findings 34 and 35). The interval and test methods are correctly derived and correctly implemented: my independent wild cluster bootstrap agrees with `analyze_re.wild_cluster` to 1.8e-15 on identical weights, my BCa agrees exactly, and Holm agrees exactly. Coverage under symmetric errors is nominal; under strong skew no method with 6 to 12 clusters is nominal, which the rule should say more plainly (finding 17).

Four BLOCKERS should be fixed before the push, because once the rule is public they can only be changed by a logged deviation:
1. The title rule awards the stronger title when a title-bearing feature set has no valid non-R4 readout at all (demonstrated with the code).
2. The fixed P2 wording and the abstract use a square drop ratio whose denominator fails the rule's own stability screen (IIE, percentile-approximate |t| 3.43), and describe the nulls' gains as "cost all of it".
3. The honesty preamble omits that P2's square restriction leaves out known unfavourable results for MSDB (pooled +0.0083 [-0.0004, +0.0171]; triangles -0.0042 [-0.0123, +0.0031]); the fallback highlight would then be false for MSDB as written.
4. The fixed seed sentence ("seeds share the frozen forward pass and the batch order") is factually wrong: each seed has its own batch order and dropout stream.

Count: 4 BLOCKER, 17 SHOULD-FIX, 14 NOTE.

## Answers to the seven questions

| # | question | answer | findings |
|---|---|---|---|
| 1 | every number cited as known matches the result files | yes, all checked numbers match; two counting slips (16 non-R4 cells, not 15; "11 distinct margins" depends on an unstated grid) | 18, 34, 35 |
| 2 | honesty preamble omits nothing known or public | no: it omits the remote push of all P1 to P4 inputs on 17 October (commit 9159a90), the unfavourable pooled and triangle alternation results that P2 leaves out, the post hoc origin of the null family, the alternating displays and R1n, the IIE square-ratio instability, and the known values that bear on P6 | 2, 3, 13, 14, 15 |
| 3 | interval methods correct and correctly implemented | yes (Webb weights, 6^6 enumeration, CR1, WCR, WCU, two-stage indices, BCa, BCa inversion, Fieller, Holm); text and code differ on the random-draw p formula; decisions (WCR) and intervals (WCU) can disagree and precedence is not stated | 16, 17, 22, 23, 24 |
| 4 | Holm family stated and sensible; P4 eligibility fixed; ratios guarded | families stated and sensible; P4 list fixed before results and correctly computed; ratio guard is breached by the fixed P2 wording and will likely be breached for MSDB's rigid share | 2, 11, 24, 25 |
| 5 | P5 and P6 categories exhaustive, wording not stronger than data; title rule unambiguous; accuracy criterion for new cells defined | P5 categories are exhaustive but categories 1 and 2 are worded more strongly than the tests allow; the title rule has a vacuous-truth loophole, ignores P3, leaves "R4 undefined" open, and lets a missing hs[11] run escape condition (d); the accuracy criterion is defined but rounding differs between the two codes that apply it | 1, 5, 6, 7, 8, 9, 11, 12, 19 |
| 6 | nothing lets an analysis be chosen after its result is seen | mostly; open: retraining, extra seeds or longer schedules for G3/A14 after outputs are seen, skipping hs[11], scoring scripts without an in-code guard, two exploratory items with open options | 7, 10, 20, 33 |
| 7 | Critic fixes 2 to 12, 16, 18, 22, 24, 25 applied | all applied; fix 4 was applied with a false statement inherited from the Critic, fix 7 with a grid-dependent count, fix 12 without the ASPP reason, fix 16 without "on squares" and without the P2 condition | 4, 12, 18, 28, table at the end |

## Findings

### 1. BLOCKER: the title rule gives T1 when a title-bearing feature set has no valid readout to compare

Evidence. Section 5.6 (c) requires that "every valid non-R4 cell keeps R4's sign". If none of R1n, R1, R2, R3 on a title-bearing set meets the accuracy criterion, (c) is vacuously true. `title_decision` (analyze_re.py lines 475 to 506) only loops over valid cells, so it returns T1. Demonstrated in `independent_check_output.txt` part 7(a): all four new readouts on `dinov2reg_L12` below the criterion, "stronger = True; title = Looking at Nothing: The Checkpoint and Layer ...". This is the case Critic K.3 point 3 exists to prevent (readout arm and checkpoint arm never crossed). It is not hypothetical for block 6: on the public checkpoint the pointwise readouts missed the criterion by 0.042 and 0.045 at block 6, and the registers block-6 threshold is 0.7077 (registers R4 mean CC 0.7377), above the plain R1n and R1 values (0.6999, 0.6976).

Fix. Add to Section 5.6 before (c): "(c0) on each title-bearing feature set at least two non-R4 cells are valid (the smallest number of valid non-R4 readouts on any feature set of the published swap); if fewer than two are valid on any title-bearing set, the scoped title is used, and the paper states how many readouts met the accuracy criterion on each new feature set." Mirror it in `title_decision`: count cells with status "valid" per title-bearing set and append a reason when the count is below 2. Add a unit test (the synthetic case of part 7(a) must give the scoped title). If the author prefers one valid cell as the minimum, write that number instead; the point is that the number is fixed now.

### 2. BLOCKER: the fixed P2 wording uses an unstable ratio and calls the nulls' gain a "cost"

Evidence. The square drop ratio D(20) = (r(illusory) - r(alt20)) / (r(illusory) - r(mis20)) has the square uniform loss as its denominator. From `v3/scores/results.json` (square, drop_uni 20): IIE +0.0541 [+0.0233, +0.0852], percentile-approximate |t| 3.43; MSDB +0.0471 [+0.0275, +0.0653], |t| 4.88. For the nulls the denominator is negative (rotation moves mass into the figure): corner sigma 40 -0.0083 [-0.0143, -0.0023], |t| 2.71; corner sigma 60 -0.0043 [-0.0079, -0.0008], |t| 2.37. Section 4.3 says an unstable ratio (|t| < 4) is "never used in the title, abstract, highlights or contributions". Yet Section 7.1 fixes the P2 sentence as "cost [0.55] and [0.56] of the uniform-rotation loss; for every local null it cost all of it (1.00 to 1.02)", the range includes corner sigma 40 (1.018) and 60 (1.020), and the abstract draft in REWRITE_PLAN Section 2.3 carries "(0.55 and 0.56; local maps 1.00 to 1.02)". The rule computed the percentile-approximate |t| for P4 (and disclosed 2.99 for SigLIP-L R3) but not for these ratios; `known_values.json` stores `L20_square` without its interval. "Cost all of it" is also wrong for the nulls, whose uniform-rotation "loss" is a gain.

Fix. (a) Add to Section 1.1 item 1: "The square drop ratios have the square uniform loss as denominator: IIE +0.0541 [+0.0233, +0.0852] (percentile-approximate |t| 3.43), MSDB +0.0471 [+0.0275, +0.0653] (4.88). For the nulls that denominator is negative (rotation moves mass into the figure); corner sigma 40 and 60 have |t| 2.71 and 2.37. IIE's ratio is therefore expected to carry the flag 'unstable' under Section 4.3." (b) Replace the P2 sentence of Section 7.1 by a ratio-free one: "P2 met: 'On squares, alternating the rotation direction, which keeps the axis mirror symmetries, kept more mass inside the figure than uniform rotation (difference [x] [CI] for IIE and [y] [CI] for MSDB, larger than for every local null, for which the two rotations left the same figure mass: differences 0.0000 to +0.0003).' The square drop ratios follow in the Results only if their denominators pass the CR1 screen of Section 4.3 (|t| >= 4); otherwise they are given with the flag 'unstable' and never in the abstract, highlights or contributions. Null ratios are given only for nulls whose denominator passes the screen, and are described as 'the same change under both rotations', not as a cost." (c) Make the abstract clause of REWRITE_PLAN 2.3 ratio-free in the same way.

### 3. BLOCKER: P2's square restriction leaves out known unfavourable results, and the fallback highlight drops "on squares"

Evidence. `v3/scores/results.json`, alt_minus_mis 20: IIE pooled +0.0259 [+0.0164, +0.0352], triangles +0.0275 [+0.0127, +0.0419]; MSDB pooled +0.0083 [-0.0004, +0.0171], triangles -0.0042 [-0.0123, +0.0031] (MSDB triangle uniform loss +0.0127 [+0.0016, +0.0249], triangle drop ratio 1.33 [0.71, 4.18]). P2 is restricted to squares, where MSDB passes (+0.0208 [+0.0132, +0.0280]). The geometric reason (triangle alternation keeps one mirror axis and two thirds of the notch-sector ink) is real and was in the earlier manuscript (`cbb/main.tex` line 57), but that text was written after these values existed, and Section 1.1 lists only the square values. Section 5.5 then prescribes the highlight "Alternating inducer rotation separates DeepGaze IIE and MSDB from 15 local nulls" (80 characters), which would be false for MSDB pooled over shapes.

Fix. (a) Add to Section 1.1 item 1: "P2 is restricted to squares. Pooled over shapes the alternation difference Delta(20) was +0.0259 [+0.0164, +0.0352] for IIE and +0.0083 [-0.0004, +0.0171] for MSDB; on triangles +0.0275 [+0.0127, +0.0419] and -0.0042 [-0.0123, +0.0031]. The square restriction and its geometric reason were stated in the earlier manuscript, which was written after these values had been computed. The pooled and triangle values are reported next to P2 in the Results (exploratory), not only in the supplement." (b) Section 5.5 and Section 7.2: highlight 4 becomes "On squares, alternating rotation separates DeepGaze IIE and MSDB from 15 local nulls" (84 characters, within 85), used only if P2 is met for both models (see finding 12).

### 4. BLOCKER: the fixed seed sentence misstates the training design

Evidence. Section 7.5: "seeds share the frozen forward pass and the batch order, so they measure head-initialisation variance only". `v3/gpu/train_full.py` lines 6 to 7 and 92 to 99: seed s sets `torch.manual_seed(s)` before each head (initial weights and dropout stream) and `numpy.RandomState(s)` for the batch order (one permutation per epoch). Seeds therefore differ in initialisation, batch order and dropout; what is shared within a seed is the forward pass per batch across readouts (`cbb/main.tex` line 91: "All readouts of one backbone and seed were trained from the same frozen forward pass per batch"). The error comes from REWRITE_PLAN Critic fix 4, which misread `nn/mock_review/meta_review.md` line 46.

Fix. Replace the Section 7.5 sentence by: "Seeds differ in head initialisation, batch order and dropout stream but share the frozen backbone, the training images and the recipe, so they measure the variability of training the head, not variability over backbones, data or recipes; readouts of one seed were trained on the same batches from one forward pass, so readout comparisons within a seed are paired." Correct Critic fix 4 in REWRITE_PLAN the same way when the plan is next edited.

### 5. SHOULD-FIX: the title rule does not require P3 to hold in full

Evidence. Section 5.6 (c) requires Holm significance for every valid cell on the three new title-bearing sets, but the P3 cells enter only through (d) (no opposite-signed interval). Demonstrated in part 7(b): P3 with 8 of 10 cells keeping the sign and every new set passing gives "stronger = True". The cell most likely to fail P3 is SigLIP-L block 24 R3 (percentile-approximate |t| 2.99; it is the largest p in the Holm step-down and must reach p < .05 on its own), so this gap is the realistic one.

Fix. Add to Section 5.6: "(c1) P3 is met in full: all 10 valid non-reference swap cells keep R4's sign after Holm adjustment. If fewer than 10 do and none has an interval of the opposite sign, the scoped title is used." In `title_decision`, append a reason when `p3res["n_keep"] < p3res["n"]`, and make `p3res` mandatory (the P6 step already recomputes P3, RUN_PLAN Step 5).

### 6. SHOULD-FIX: condition (d) is undefined when R4's reference sign is undefined

Evidence. Section 5.6 says that if R4's test fails the reference sign is undefined and the cells "cannot support keeps the sign", but not what (d) does on that set. The code computes `opposite_sign_interval` against the sign of R4's point estimate even when the reference is undefined, so a near-zero R4 with a positive point estimate and a valid readout at -0.06 narrows both titles (part 7(e): reference defined False, R2 opposite_sign_interval True, title None). This matters most for `dinov2ms_L11`, where the neighbouring sweep blocks (-0.029 at block 10, -0.098 at block 12) make a weak R4 plausible.

Fix. Fix one reading now, in the rule and the code. Recommended: "If R4's reference is undefined on a feature set, any valid cell on that set whose 95% WCU interval excludes zero counts as 'an interval of the opposite sign' for condition (d), because the readout then sets a sign that R4 does not; the paper states this." (The alternative, "(d) is not evaluated on that set", is weaker and should then be stated as such.) Add a unit test for both signs of R4's point estimate.

### 7. SHOULD-FIX: skipping or losing `dinov2ms_L11` removes the only route by which it can veto a title

Evidence. Section 5.6 (a) requires completeness only for the three title-bearing sets; `dinov2ms_L11` enters only through (d). If its cells are not scored, (d) cannot fail on it, and `title_decision` checks completeness only for `TITLE_BEARING`. hs[11] is trained in the same A14 job, so a missing hs[11] result is a choice or a failure, not a separate run.

Fix. Section 5.6 (a): "... and A14 was also scored for every cell of `dinov2ms_L11`; a missing `dinov2ms_L11` cell counts as failing (a)." Include `dinov2ms_L11` in the completeness check of `title_decision`.

### 8. SHOULD-FIX: `p6` crashes instead of reporting "not run"

Evidence. Part 7(c) and 7(d): a missing R4 on a new set raises `KeyError: 'R4'` (analyze_re.py line 448), and a cell that has scores but no `mit_*.json` raises `KeyError: 'R2'` (line 456, because `step_primary` adds the head and its CC in one `try`, and the score read succeeds before the CC read fails). Section 5.6 and Section 9 say a missing run or cell "counts as failing (a)" and is "reported as not run"; the code does not implement that.

Fix. In `step_primary`, add a head only when both its three score files and its three `mit` files exist; in `p6`, if R4 is missing set `R4_reference_defined = False`, `complete = False` and mark the set "not run"; a cell without CC gets status "not run". Add two unit tests (the cases of part 7(c) and 7(d) must give the scoped title, not an exception).

### 9. SHOULD-FIX: the accuracy criterion is applied with different rounding in the two codes that apply it

Evidence. Published validity (`v3/gpu/analyze_full.py` lines 88 to 92, copied as `gpu/analyze_gpu.py`): mean of the seed CCs rounded to 4 decimals, threshold `round(R4 mean - 0.03, 4)`, valid if rounded mean >= rounded threshold. `analyze_re.accuracy_valid` compares unrounded means. Section 2.7 says the criterion is the same "for old and new cells alike" but does not say which. A borderline new cell could be valid in one output and invalid in the other (ShapeNet-C R3 cleared by 0.0014, so this is not academic).

Fix. Section 2.7: "Validity is computed exactly as in the published analysis: the mean of the three unrounded seed CCs, rounded to 4 decimals, against round(R4 mean - 0.03, 4)." Change `accuracy_valid` to that computation and print the thresholds now: 0.6982 for `dinov2reg_L12` and 0.7077 for `dinov2reg_L6` (registers R4 means 0.7282 and 0.7377); the `dinov2ms_*` thresholds follow from the new R4.

### 10. SHOULD-FIX: nothing regulates retraining, extra seeds or a late completion of G3 and A14

Evidence. Section 9 forbids changing family membership, thresholds, estimands, categories and the title rule after results are seen, but not rerunning a cell with a longer schedule, another seed or a changed learning rate after its held-out CC or display scores have been read (for example to bring a pointwise readout over the criterion), and it sets no time limit on a run that "does not complete".

Fix. Add to Section 5.6 or 9: "Each G3 and A14 cell is trained once per seed (0, 1, 2) with the fixed recipe. After any held-out CC or display score of a cell has been read, no retraining, extra seed, longer schedule or changed hyperparameter is allowed for that cell or its feature set. A crashed job may be resumed from its saved state or restarted with identical settings, and the restart is logged. The title decision uses the runs completed before first submission; a run completed later can change the title only in a revision, with a logged deviation."

### 11. SHOULD-FIX: P5 categories 1 and 2 are worded more strongly than their tests, and MSDB's rigid share will probably be unstable

Evidence. Category 1 requires (i) alignment-specific loss beyond every null and (ii) C = L_uni - 2 L_rig > 0. That still allows up to half of the loss to come from the orientation change, yet the fixed sentence ends "so it is not an orientation effect". Category 2 is labelled "shrinks", but it is defined by (ii) not being met, which includes an inconclusive C; its sentence says "uniform rotation does not isolate alignment; the alternation test does", although the earlier manuscript itself says the alternating loss "is not attributable to collinearity alone" (`cbb/main.tex` line 198). Category 3 says "so uniform rotation does not isolate alignment" also when (i) fails for lack of precision with a small rigid share. All three sentences quote the rigid share [s] = L_rig / L_uni; MSDB's L_uni (all 60 configurations) is +0.0299 [+0.0155, +0.0450], percentile-approximate |t| 3.97, below the screen of Section 4.3.

Fix. Replace the three sentences of Section 7.2:
- Category 1: "Rotating the whole display by the same angle, which keeps the inducers aligned and turns every edge by that angle, reproduced [L_rig] [CI] of the uniform-rotation loss ([L_uni] [CI]). The part that depends on inducer alignment ([A] [CI]) was larger than for every local null and more than half of the loss, so the loss is not explained by the change in edge orientation alone."
- Category 2 (label "alignment share not established"): "Rotating the whole display reproduced [L_rig] [CI] of the uniform-rotation loss. The part that depends on inducer alignment ([A] [CI]) still separated [model] from all 15 local nulls, but it was not shown to exceed half of the loss, so uniform rotation alone does not isolate alignment; the square alternation test, which matches the size of the edge rotation, is the closer test, with the symmetry caveat stated in Section [x]."
- Category 3: "Rotating the whole display reproduced [L_rig] [CI] of the uniform-rotation loss, and the part that depends on inducer alignment ([A] [CI]) did not separate [model] from [k] of the 15 local nulls, so this control could not confirm that the uniform-rotation loss depends on alignment."
- Add to Section 5.5: "MSDB's uniform loss has percentile-approximate |t| 3.97, so its rigid share is expected to carry the flag 'unstable'; s is then reported only with the flag, and the sentences use the differences."

### 12. SHOULD-FIX: the P2 by P5 outcome space is not covered

Evidence. Section 5.5: with P5 category 2 or 3 "the abstract leads with the square alternation result (P2)" and highlight 4 becomes the alternation highlight. Nothing says what happens if P2 is not met for a model at the same time; the abstract would then lead with a result that failed.

Fix. Add to Section 5.5: "If P2 is not met for both models, highlight 4 is replaced by the strongest highlight that P1 to P6 allow (or dropped), the abstract carries no rotation claim, and the rotation results appear in the Results with the category and P2 wording. If P2 is met for one model only, the highlight and abstract name that model only."

### 13. SHOULD-FIX: the preamble does not record that every P1 to P4 input was in the remote repository since 17 October, and it treats "pushed" and "public" inconsistently

Evidence. `git -C looking-at-nothing log` and reflog: commit 9159a90 (author date 2026-10-17 22:49:00 +0100, pushed 23:01:20 +0100) added `results/v3/scores/raw*/` (stim_v3 and stim_alt per-image scores of IIE, MSDB and the sweep readouts), `raw_nulls.json.xz`, `results.json.xz`, `results/v3/gpu/full/results_full.json`, all 180 readout-swap score files, `results/v2/verify/encoding_nulls.json` and the decomposition results. Section 1.2 calls the new intervals "not known"; strictly, they were not computed, while their inputs were available to anyone with access. I searched every `.py`, `.md`, `.json` and `.tex` file outside `neuro/` for Webb, wild cluster, BCa, Fieller and Holm: none (the only hits are unrelated substrings, in SHA-256 hashes and in the word "webcam"), which supports "not computed". Separately, Section 1.1 item 5 and `RF_RULE_CORRECTIONS.md` (C1, C2, C5) say the sweep results were "public since" commit a533d4b on 9 October, while item 4 says local records "do not show whether the repository was public" on 15 October; the evidence for both dates is the same kind (local reflog of a push).

Fix. Add to Section 1.1: "15. Remote record. Every per-image score that P1 to P4 use, and `results.json`, `results_full.json` and `encoding_nulls.json`, were pushed to the remote repository in commit 9159a90 (17 October 2026, pushed 23:01:20 +0100). 'Not known' in Section 1.2 therefore means 'not computed': no script in the project computed a wild cluster bootstrap, BCa, Fieller or Holm statistic before this rule (search of all .py, .md, .json and .tex files outside neuro/, 18 November 2026)." Replace "public since commit a533d4b" (item 5) and "public repository"/"they were public" (RF_RULE_CORRECTIONS C1, C2, C5) by "pushed to the remote repository in commit a533d4b (9 October 2026)", unless evidence of public visibility on that date is added; use the same wording standard as item 4.

### 14. SHOULD-FIX: the preamble does not say that the null family, the alternating displays and R1n were introduced after the version-2 results

Evidence. `v2/DEVIATIONS.md` section "Perception-science verifier (PV): post hoc analyses outside the plan", written 15 October 2026 "after all section A to E results had been seen", defines the 15 local nulls (PV1). The alternating-rotation displays (v3) and R1n (`v3/NOTES.md` line 210) came after that; the earlier manuscript says the pilot used "readouts R1 to R4 (no R1n)" and that "R1 failed it on all four feature sets there" (`cbb/main.tex` line 91). None of the three is in `v2/ANALYSIS_PLAN_v2.md`. P1, P2 and P5 test against this null family, P2 uses the alternating displays, and P3 and the title count R1n.

Fix. Add to Section 1.1: "16. Post hoc elements that the primary contrasts use. The 15 local nulls were defined on 15 October 2026 after every result of the version-2 plan had been seen (v2/DEVIATIONS.md, PV1); the alternating-rotation displays and R1n were introduced afterwards, R1n after a pilot in which R1 failed the accuracy criterion on all four feature sets. None of them is in v2/ANALYSIS_PLAN_v2.md." One sentence in the Methods should carry the same content.

### 15. SHOULD-FIX: values that bear on P6 are known and should be listed

Evidence. Section 0.2 says P6 tests outcomes "that nobody had seen", which is true of the new cells, but these related values are known and shape expectations: xSoyelNet-ML, which reads the same two tensors, gave -0.024 [-0.035, -0.013] against control180 and +0.030 [+0.019, +0.041] against control90 (`cbb/numbers.tex` lines 279 to 285); in DeepGaze MSDB the Shapley shares of DINOv2 block 11 and block 7 are 0.30 [0.19, 0.43] and 0.12 [0.08, 0.17] (`cbb/numbers.tex`, from `v2/decompose/results.json`); the sweep readouts bracket hs[7] (+0.082 at block 6, +0.035 at block 8) and hs[11] (-0.029 at block 10, -0.098 at block 12); block 11 fails the pre-specified token rule (interior +0.304 [0.230, 0.377] against a background upper bound of +0.281) and passes the near-exterior one; the registers R4 cells (+0.111 and +0.047) are known; and the plain pointwise readouts failed the criterion at block 6.

Fix. Add these as Section 1.1 item 17, with the sentence: "P6 cells are unseen; these related values were known when the rule and the title condition were written." Also note there that xSoyelNet-ML is one of the author's commercial models (the competing-interest statement covers it).

### 16. SHOULD-FIX: decisions use the WCR p-value while intervals are WCU percentile-t; precedence is not stated

Evidence. Section 4.1 tests with WCR (null imposed) and reports WCU intervals (unrestricted). They are different procedures and can disagree near the boundary (a WCU interval excluding zero with WCR p > .05, or the reverse). The title condition (c) says "keeps R4's sign with an interval excluding zero (Holm-adjusted p < .05, point estimate with R4's sign)", which mixes the two; the code uses the Holm-adjusted WCR p only. In my simulations the interval obtained by inverting the WCR test has the same coverage as WCU (G = 12: 0.955 against 0.958 symmetric; G = 6: 0.960 against 0.960), so the choice is not about accuracy but about consistency.

Fix. Add to Section 4.1: "Decisions (P1 to P3, P5, P6 and the title rule) use the WCR p-value. The WCU interval is the reported interval estimate. When the WCR decision at .05 and the WCU interval disagree on whether zero is excluded, both are reported and the WCR p decides." In Section 5.6 (c) write "keeps R4's sign (Holm-adjusted WCR p < .05 and point estimate on R4's side)" and drop "with an interval excluding zero". Optionally report the WCR-inverted interval in the supplement (an implementation is `wcr_inverted_ci` in `review_rule/independent_check.py`).

### 17. SHOULD-FIX: say plainly that no method is nominal under strong skew with 6 to 12 clusters

Evidence. `independent_check_output.txt` part 3 (DGPs different from `test_analyze_re.py`): symmetric heteroskedastic clusters with Laplace noise, WCU coverage 0.958 (G = 12) and 0.960 (G = 6), WCR size 0.045 and 0.040, two-stage percentile 0.922 and 0.847; one cluster with three times the spread, WCU 0.940 and 0.953, size 0.055 and 0.073; centred lognormal cluster effects (strong right skew), WCU 0.825 and 0.847, WCR-inverted 0.840 and 0.840, CR1 t(G-1) 0.833 and 0.840, two-stage percentile 0.807 and 0.707, WCR size 0.160 at both G. The rule reports only the test file's milder skewed case (size 0.090 and 0.113). The IUT claims of P1, P2 and P5 take each component at nominal .05, so under strong skew they are liberal.

Fix. Replace the last part of Section 1.1 item 13 by: "Under strongly skewed cluster effects no interval method we tested held nominal coverage with 6 or 12 clusters (coverage 0.83 to 0.90 in two independent stress cases; WCR size 0.09 to 0.16); the wild cluster intervals were never worse than the two-stage percentile intervals." Add to Section 4.1 a fixed diagnostic that changes no decision: "For every primary linear contrast the sample skewness of its cluster means is reported next to the result."

### 18. SHOULD-FIX: the validity-margin count depends on an unstated grid, and one count is off by one

Evidence. Section 1.1 item 4 and the fixed wording of Section 7.5: "over margins 0 to 0.06 the valid set changes at 11 distinct margins". From the unrounded seed CCs (`v3/gpu/full/scores/mit_*.json`) the 16 non-R4 cells have 16 distinct gaps (0.001998 to 0.044558); at four decimals 14 distinct values; on a 0.001 grid 11 changes; on the 0.005 grid that Section 6 item 7 prescribes, 6 changes (`review_rule/verify_known.py` and the inline check in this review). Item 4 also says "the 15 non-R4 cells" and lists 16 values (0.0153 twice; 0.0423 is also two cells: DINOv2-B block 6 R1n and SigLIP-L R1).

Fix. Item 4: "the CC gaps of the 16 non-R4 cells ... 0.0423 (two cells) ...". Section 7.5 and item 4: "the valid set changes 16 times between margins 0.0020 and 0.0446 (once per cell), and sign agreement among valid cells is 100% at every margin".

### 19. SHOULD-FIX: two P6 wordings overreach and two outcomes have no wording

Evidence. Section 7.3, positive `dinov2ms_L7L11`: "the negative 180 degree contrast of xSoyelNet-ML then reflects its loss, aspect handling or test-time averaging, not the tensors". The swap varies the readout under one loss, data set and recipe; xSoyelNet-ML differs in more than the three listed factors, so a positive swap result shows only that these tensors can give a positive contrast under this recipe. There is no fixed wording for "R4 reference undefined on a set" or for "no (or fewer than the required) valid non-R4 cells on a set".

Fix. Positive case: "Under this recipe the two tensors that xSoyelNet-ML reads gave a positive contrast with every valid readout, so xSoyelNet-ML's negative 180 degree contrast comes from something other than the tensors, for example its loss, training data, aspect handling or test-time averaging; this study does not separate them." Add: "R4 undefined on [set]: 'On [set], R4 gave no reliable contrast ([E] [CI]), so no readout comparison on it bears on the sign.'" and "Too few valid cells on [set]: 'On [set], [k] of 4 readouts met the accuracy criterion, too few for a readout comparison; the scoped title is used.'"

### 20. SHOULD-FIX: the scoring scripts have no in-code guard against running before the push

Evidence. Section 8.1: "No model or null is scored on any stimulus for this re-analysis ... before" the push, and "`analyze_re.py` refuses to run any step otherwise". `run_gpu_re.sh` also checks. `scoring/score_rigid.py`, `scoring/local_nulls_rigid.py` and `gpu/eval_re.py` (which RUN_PLAN Steps 3 R1 to R6 call directly on `stim_rigid`) contain no such check (grep for `REANALYSIS_RULE.sha256` and `PUBLIC_PUSH` finds only `run_gpu_re.sh`). The promise of Section 8.1 is procedural for exactly the new-data family (P5).

Fix. Copy `require_public()` (or a ten-line equivalent that checks both files exist and the rule hash matches) into `scoring/common_rigid.py` and `gpu/eval_re.py`, called before any display is scored; the stim_v3 pipeline checks of Section 3.2 run after it. This changes hashed files, so do it before `fix_rule.py`.

### 21. SHOULD-FIX: say that the P3 and P6 intervals hold the trained heads fixed

Evidence. P3 and P6 average seeds per configuration and resample configurations only (Section 2.2), so their intervals reflect stimulus sampling, not training-run variability. The receptive-field wording carries the equivalent sentence ("intervals reflect stimulus sampling only"); the readout claims do not.

Fix. Add to Section 7.5: "Intervals for P3 and P6 treat the configurations as the sampling unit and hold the three trained heads of each cell fixed (seed-averaged); seed-to-seed variation enters only through the per-seed counts and the exploratory two-level seed bootstrap."

### 22. NOTE: the random-draw p formula differs between rule text, docstring and code

Evidence. Section 4.1: "p = (1 + number of draws at least as extreme) / 10,000". Docstring (analyze_re.py line 18): "p = (1 + #{|t*| >= |t|}) / (B + 1)". Code (lines 99 to 101): p = 2 (1 + min(k_up, k_lo)) / (B + 1), the equal-tailed version, slightly more conservative (smallest p 0.0002).

Fix. State the code's formula in Section 4.1 and the docstring: "p = 2 (1 + min(#{t* >= t}, #{t* <= t})) / (B + 1)".

### 23. NOTE: "at most 2 x 3^6 / 6^6 = 0.031" is an empirical regularity, not a bound

Evidence. Section 1.1 item 13 and Section 4.1. In my search over 3,000 random six-cluster datasets (361 with t > 5) the largest enumerated p was 0.0123; constructed cases with one cluster near zero gave p 0.019 to 0.024 (at t 2.7 to 3.8). No counterexample was found, but no proof is given.

Fix. Write "in our simulations at most 0.031" instead of "at most 0.031".

### 24. NOTE: Fieller and the denominator check are only partly implemented

Evidence. Section 4.2 promises a Fieller interval "next to every ratio" and a BCa-versus-Fieller rule; `p4` computes Fieller for the readout's ratio only (line 393), not for R4's ratio, and a Fieller interval does not exist for a difference of ratios, so the disagreement rule cannot apply to the P4 estimand. Section 4.3's condition "reported only when the 95% WCU interval of its denominator excludes 0" is not checked in `p4` or `p5`.

Fix. Compute Fieller for both ratios of each P4 contrast; state in Section 4.2 that the BCa-versus-Fieller rule applies to single ratios, not to ratio differences; add the WCU denominator check to `p4` and `p5` (flag "not reportable").

### 25. NOTE: Holm is conservative for the conjunctive claims, and P3 and P6 use different family granularity

Evidence. "All valid cells keep the sign" is an intersection claim, for which the IUT at .05 per cell already controls the error; Holm is needed only for "k of n" counts. P3 is one family of 10 across four feature sets, P6 is one family per feature set (at most 4). Both choices are defensible and are fixed.

Fix. One sentence in Section 4.4: "Holm is used for the counts of P3 and P6; for the claim that every valid cell keeps the sign it is conservative." Optionally say why P6 families are per feature set (each new feature set is a separate claim; the title condition combines them as an intersection).

### 26. NOTE: condition (d) uses unadjusted intervals across up to 26 cells

Evidence. A valid cell with a true contrast near zero has about a 2.5% chance of an interval on the opposite side; across many such cells the chance that (d) fires falsely grows. The error falls against the stronger claim, so it is acceptable.

Fix. One sentence in Section 5.6: "Condition (d) uses unadjusted 95% intervals; this makes the narrowing more likely, not less."

### 27. NOTE: exploratory A10 with 4 clusters

Evidence. `g4_check_output.txt`: WCU coverage 0.970 (symmetric), 0.957 (one wide cluster), 0.847 (skewed); WCR size 0.023, 0.020, 0.137; smallest p with 6^4 = 1,296 vectors is 6 / 1,296 = 0.0046.

Fix. Label the A10 intervals in the supplement "4 clusters; approximate" and state the smallest attainable p.

### 28. NOTE: Critic fix 12 is applied without the ASPP reason

Evidence. Fix 12 asks for the narrow ERF50 range "with the shared ASPP readout as an untested reason"; Section 7.7 gives the range (238.5 to 325.5 px, factor 1.36; both verified against `nn/results/summary.json`) but not the reason.

Fix. Add to Section 7.7: "a plausible but untested reason for the narrow range is that every sweep readout shares the ASPP design".

### 29. NOTE: the layer-arm sentence is inaccurate as worded

Evidence. Section 1.1 item 10: negative readouts lie "between positive neighbours in two of the three networks". DINOv2-B's negative eligible readouts are its last two blocks (10 and 12, after positive blocks 4 to 8); only SigLIP-L block 12 lies between positive eligible blocks (8 and 20).

Fix. "3 of the 13 eligible readouts are negative: DINOv2-B blocks 10 and 12, the last two blocks after positive blocks 4 to 8, and SigLIP-L block 12, between positive blocks 8 and 20."

### 30. NOTE: the lead claim of Section 7.4 should carry the P3 count when P3 is not met in full

Evidence. "No readout that met the accuracy criterion changed the sign" stays literally true when a cell fails to exclude zero, but a reader takes it to mean that every valid readout kept the sign.

Fix. Add: "If P3 gives fewer than 10 of 10, the lead claim adds '([k] of 10 with intervals excluding zero after Holm adjustment; none of the opposite sign)'."

### 31. NOTE: the P4 BCa intervals may be slightly liberal

Evidence. `tests_output.txt`: pairs cluster BCa coverage 0.913 for a ratio of means with 12 clusters (300 datasets, Monte Carlo SE about 0.013); Fieller 0.927.

Fix. One clause in the Methods: "in simulation the BCa interval for a ratio covered 0.91, so P4 counts may be slightly liberal".

### 32. NOTE: P5 when a pipeline check fails

Evidence. Section 3.2 requires the CPU scoring of IIE and MSDB to reproduce published stim_v3 values to 1e-4 before any rigid display is scored. If MSDB on the CPU misses that tolerance, P5 for MSDB is not computable; Section 5.5 implies, but does not say, that this is "not category 1".

Fix. "If a decision-bearing model cannot pass its pipeline check, its P5 is reported as not computable and treated as not in category 1 for the abstract rule. A tolerance may be changed, with a logged deviation, only before any rigid display has been scored."

### 33. NOTE: two exploratory items keep open options although Section 6 says "fixed definitions"

Evidence. Section 6 item 6: MSI-Net "rank only unless its float maps are regenerated"; item 9: "Optional point-spread probe".

Fix. Decide now (for example: MSI-Net float maps are regenerated on the CPU; the point-spread probe is run, or is not run), or state that the choice will be made before any result of that item is computed and logged.

### 34. NOTE: verified, no action needed (numbers)

- Section 1.1 item 1 (17 rows: E, L20, square Delta20, square D(20)) and item 2 (20 swap rows: CC, gap, E with CI, seed means, 336 and 448 px, E vs control90, U20, A20): every number matches `v3/scores/results.json` and `v3/gpu/full/results_full.json` (`compare_tables.py`: 0 mismatches).
- P1 and P2 point differences (+0.0508, +0.0306; +0.0239, +0.0205); 10 of 10 valid cells with R4's sign; 30 of 30 seed point estimates; 29 of 30 per-seed intervals (SigLIP-L R3 seed 1 [-0.021, +0.184]); 59 of 60 overall; 6 of 6 and 18 of 18 below criterion; 20 of 20 cells; control90 sign agreement in all 20 cells; ShapeNet-C R3 clearance 0.0014.
- Section 5.4 table: all 18 point differences and every denominator |t| (6.61, 7.12, ..., 5.80, 4.82); excluded SigLIP-L R3 contrasts (|t| 2.99; differences +0.198 and +0.444).
- Item 5 sweep values (recomputed from `sweep/<bb>/sweep_results.json`); item 7 registers values (`REGISTERS_RESULTS.md`); item 9 token verdicts under both nulls, including DINOv2-B block 12 +0.305 [0.225, 0.388] against +0.267 and SigLIP-L 24 0.523 [0.448, 0.598] against 0.429 [0.375, 0.482]; the 12 of 18 agreement count and the six named blocks; item 10 (3 of 13); item 11 (rho -0.062 stored, -0.023 corrected; within-network -0.03, +0.14, +0.43 unchanged); Section 0.5 RF wording (verbatim from `nn/PROPERTY_RESULTS.md` line 13); plain thresholds 0.6955 and 0.7122; ERF50 range.
- Item 13: every coverage, size and p figure equals `tests_output.txt`; that file was produced by the current `analyze_re.py` (hash match).
- 0.03 margin: commit 37a04f0 at 15:54:11, pushed 15:54:13 +0100; the pilot started after it (cache files from 16:00 on 15 October), so "before any readout-swap training" holds.

### 35. NOTE: verified, no action needed (receptive-field corrections and code)

- `RF_RULE_CORRECTIONS.md`: rule hash 8e59fe87... matches `nn/ANALYSIS_RULE.sha256`; preamble lines quoted correctly; sweep file times (8 October 18:58:01, 21:36:18, 23:31:49; Ehrenstein 9 October 07:30:54 to 07:46:50); git blob hashes of the three `sweep_results.json` files equal in a533d4b, in the current tree and on disk; `nn/analyze_nn.py` lines 265 to 267 quoted correctly (stored L20 0.005198 for DINOv2-B L2, stored E rounded to -0.0); file times 07:53:35 (rule), 08:01:09 (analyze_nn.py), 09:30:09 (summary.json) UTC; RUN_LOG 08:08:00 to 09:30:09. The only issue is the "public" wording (finding 13).
- `analyze_re.py`: Webb weights (mean 0, variance 1, fourth moment 7/6), the 6^6 enumeration (identical set, 2 x 3^6 / 6^6 = 0.03125 sign-constant, smallest p 6 / 46,656), CR1 (G / (G - 1), K = 1), WCR with the null imposed, WCU percentile-t, the two-stage indices, BCa (bias, delete-one-cluster acceleration, adjusted levels), BCa p by inversion (0.00730 against 0.00723 by bisection), Fieller with t(G - 1), Holm, the P4 list, the P5 algebra (A = L_uni - L_rig = E_rig - E_mis) and the exhaustive P5 category assignment are correct. Score layouts: stim_v3 and stim_alt keys do not overlap in the readout or DeepGaze files, so the merges in `head_scores` and `survey_scores` are safe.

## Critic fixes 2 to 12, 16, 18, 22, 24, 25

| fix | where in the rule | status |
|---|---|---|
| 2 lead sentence | 7.4 | applied |
| 3 registers reporting | 1.1 item 7, 7.4 | applied |
| 4 seed runs | 7.5, 6.3 | applied with a false design statement (finding 4) |
| 5 below-criterion sentence | 7.5 | applied |
| 6 token analysis, both nulls, registers tokens required | 1.1 item 9, 6.8, 7.6 | applied |
| 7 margin wording, 0.0014 clearance | 1.1 item 4, 5.4, 7.5 | applied; count depends on an unstated grid and "15 cells" should be 16 (finding 18) |
| 8 P4 option (a), 18 contrasts, stable leading example | 5.4, 7.1 | applied; numbers verified |
| 9 honesty sentences | 0.2, 0.3 | applied |
| 10 dependencies, definitions in the rule | 6.4, 6.6, 6.10, 8.1 | applied |
| 11 enumeration for 6 clusters, smallest p, unit test | 4.1, 1.1 item 13 | applied ("at most 0.031" is empirical, finding 23) |
| 12 RF reporting | 6.13, 6.18, 7.7 | applied except the ASPP reason (finding 28) |
| 16 C7 fallback | 5.5, 7.2 | applied; highlight lacks "on squares" and the P2 condition (findings 3, 12) |
| 18 A15(c) conditional | 6.12 | applied |
| 22 hard-rule gates | 8.1 to 8.3 | applied in the rule; not enforced in the scoring scripts (finding 20) |
| 24 never "pre-registered"; DeepGaze MSDB naming | header, 2.4, 7.7 | applied (no em-dash, no en-dash, no banned name in the rule) |
| 25 12 clusters, no "20 cells" | 2.1, 7.7 | applied |
