# Corrections to the receptive-field analysis rule (`nn/ANALYSIS_RULE.md`)

- Dated 18 November 2026. Author: Hamit Soyel. Revised the same day after the statistics review of the re-analysis rule (finding 13): "public" is replaced by "pushed to the remote repository" wherever the only evidence is the local record of a push, because local records do not show whether the repository was publicly visible on those dates.
- `nn/ANALYSIS_RULE.md` is frozen (SHA-256 `8e59fe87433a101aba97c99bca8e37a9e36faf08b47c9a70b1c6a141de3b403a`, recorded 2026-11-17T07:53:35Z in `nn/ANALYSIS_RULE.sha256`; re-checked today, unchanged). It is not edited. This note records the factual errors found in its preamble and in material that repeats them, with the correct statement and the evidence for each. It is intended to be published next to the rule (for example as `nn/DEVIATIONS.md` or `nn/RULE_CORRECTIONS.md` in the public repository; the copy is the author's action).
- None of these corrections changes the rule's decision. Eligibility for the primary test depends only on the aligned effect, the decision (INCONCLUSIVE, case I-d) does not involve secondary item 7, and the decision wording stays verbatim.
- Timing context, for completeness (not an error in the rule): the rule and its hash were not public before scoring. `nn/` was never committed; the last push of `Deep-c-Vision-LTD/looking-at-nothing` before scoring was 2026-10-18 05:48:37 +0100 (commit 6ea8ddc). The rule's timing therefore rests on local file records (rule and hash written 07:53:35 UTC, three open readings fixed in `nn/analyze_nn.py` at 08:01:09 UTC by its file time, scoring 08:08:00 to 09:30:09 UTC on 17 November 2026). In paper text it is "a fixed analysis rule, not public before scoring"; it is not a preregistration.

## C1. "BEFORE any of the nine unscored layer-sweep readouts was scored on the displays" (lines 3 to 4)

Wrong for the aligned contrast. Correct statement: all 18 layer-sweep readouts had been scored on the illusory, 180 degree control and 90 degree control displays on 8 October 2026, when they were trained. What the nine readouts of the rule's "not yet scored" column lacked before 17 November were the scores of the version-3 pipeline: the uniform rotations (mis10, mis20, mis30) and the real, real-filled and Ehrenstein measures on `v2/stim_v3`, the alternating rotations on `v3/stim_alt`, the core measures, and every receptive field.

Evidence:
1. `sweep.py evaluate()` (project root, lines 97 to 134) scored every sweep readout on the support-ratio 0.3, 0.5 and 0.7 illusory, control (= control180) and control90 images of `stim_v2` at 336, 448, 560 and 672 px and saved per-image r in `sweep/<bb>/sweep_results.json`.
2. File times of those files (local, +0100): `sweep/dinov2/sweep_results.json` 2026-10-08 18:58:01; `sweep/clip/sweep_results.json` 2026-10-08 21:36:18; `sweep/siglip/sweep_results.json` 2026-10-08 23:31:49. The Ehrenstein files (`ehr_results.json`) were written 2026-10-09 07:30:54 to 07:46:50.
3. The same six files were committed and pushed to the remote repository (`Deep-c-Vision-LTD/looking-at-nothing` on GitHub) in commit a533d4b6ebc29d861bf892e6d0e34545d86ff7e4 (author date 2026-10-09 20:16:02 +0100; pushed 2026-10-09 20:16:27 +0100 by the local reflog of `refs/remotes/origin/main`) as `results/layers/<bb>/{sweep,ehr}_results.json`. The git blob hashes of the three `sweep_results.json` files in a533d4b equal those of the files on disk today and of `legacy/results/layers/<bb>/sweep_results.json` in the current tree of the remote repository (DINOv2 2bf7a2a5f50bf261274e4929768ea0083e128220, CLIP 83709ce6d624e24bf66376f85ea674c9a33b4c0a, SigLIP 935409834ffb257e65592e8c5b4616e97058a983; `git rev-parse a533d4b:<path>` against `git hash-object <file>`, checked 18 November 2026).
4. The images are the present displays: the md5 of each of the 180 `stim_v2/img` images used there (2 shapes x 3 support ratios x 2 polarities x 5 positions x {illusory, control, control90}) equals the md5 of the corresponding `v2/stim_v3/img` image (illusory, control180, control90): 180 of 180 identical (checked 18 November 2026; for example `square_sr30_dark_p0_illusory.png` 4b377521c1cde08ec2b0a01860f1e946 in both sets).

## C2. "Nothing is known about the nine unscored readouts on the displays" (line 10)

Wrong. Correct statement: the aligned effects of the nine readouts against the 180 degree and 90 degree controls were known (on disk since 8 October 2026 and pushed to the remote repository on 9 October 2026); their rotation, core, real-figure and Ehrenstein scores in the version-3 pipeline and every receptive field were not. Because the rule's eligibility for the primary test depends only on the aligned effect, the eligible set (13 readouts) was predictable from numbers already on disk.

Evidence: the size-averaged (336 and 448 px) aligned effect against control180 recomputed today from the pushed `sweep_results.json` files equals, to the four decimals printed, the value reported as "newly scored" in `nn/PROPERTY_RESULTS.md`:

| readout | from `sweep_results.json` (8 Oct) | `nn/PROPERTY_RESULTS.md` (17 Nov) |
|---|---|---|
| DINOv2-B L4 | +0.0507 | +0.0507 |
| DINOv2-B L6 | +0.0824 | +0.0824 |
| DINOv2-B L10 | -0.0285 | -0.0285 |
| CLIP-L L8 | +0.0409 | +0.0409 |
| CLIP-L L16 | +0.0176 | +0.0176 |
| CLIP-L L20 | -0.0101 | -0.0101 |
| SigLIP-L L12 | -0.0206 | -0.0206 |
| SigLIP-L L16 | +0.0019 | +0.0019 |
| SigLIP-L L20 | +0.0473 | +0.0473 |

(The nine Appendix B readouts agree as well: DINOv2-B L2 -0.0000, L8 +0.0347, L12 -0.0975.)

## C3. Table "scored on the displays / not yet scored" (lines 24 to 28)

The column "not yet scored" (DINOv2-B L4, L6, L10; CLIP-L L8, L16, L20; SigLIP-L L12, L16, L20) should read "not yet scored in the version-3 pipeline (rotation, core, real-figure, Ehrenstein) and on the alternating displays; aligned contrast scored 8 October 2026". Evidence as C1 and C2.

## C4. "9 newly scored" and the "new" source labels (`nn/PROPERTY_RESULTS.md`, line 3 and the Source column of Section 2)

The labels refer to the version-3 pipeline run of 17 November 2026, not to the first scoring of these readouts on the displays. Correct reading: "9 readouts scored in the version-3 pipeline on 17 November 2026 (aligned contrast first computed 8 October 2026)". Evidence as C1 and C2. `nn/PROPERTY_RESULTS.md` is a generated report outside `neuro/`; it is not edited here.

## C5. The manuscript sentence that carries the same error (`cbb/main.tex` line 231, Appendix B)

"the other nine blocks of the 18-block sweep were trained but not scored on them" is wrong in two ways: the aligned effects of all 18 blocks were computed (8 October), and they had been pushed to the remote repository (9 October). Replacement text for the Neurocomputing version is in `neuro/REWRITE_PLAN.md` Section 3.4 and `neuro/understand/manuscript_map.md` PROV-1 and PROV-2.

## C6. Sign of L20 for DINOv2-B L2 (secondary item 7; `nn/analyze_nn.py` lines 265 to 267)

The rule (Section 7, item 7) defines L20 = sign(E) x mean figure-mass loss r(illusory) - r(mis20). The code reads

```python
sgn = np.sign(o["aligned"]["mean"]) or 1.0
l20 = X(parts, "illusory") - X(parts, "mis20")
rec["L20"] = float(sgn * l20.mean())
```

`o["aligned"]["mean"]` is E rounded to four decimals. For DINOv2-B L2 the unrounded E is -0.0000288 (mean over the 60 configurations of the size-averaged per-configuration r_inside of `v3/scores/raw_v3maps/dinov2_L2_s{336,448}.json`), the rounded value is -0.0, `np.sign(-0.0)` is a zero and therefore false, and `or 1.0` gives +1. The stored L20 is +0.0052 (`nn/results/summary.json`, 0.005198); under the rule it is -0.0052. No other readout is affected (CLIP-L L4, stored E -0.0003, has the correct sign).

Effect on secondary item 7 (Spearman rho between ERF50 and L20, all 18 readouts): stored rho = -0.06 (95% CI [-0.18, +0.13], permutation p = .807); with the corrected sign rho = -0.023 (recomputed 18 November 2026 from the unrounded L20 values in `summary.json`). Within-network rho values are unchanged (DINOv2-B -0.03, CLIP-L +0.14, SigLIP-L +0.43), because L2 keeps its rank within DINOv2-B. The stimulus-resampling CI and the 100,000-permutation p of the corrected value are recomputed with a copy of the rule's code in `neuro/reanalysis/` after `REANALYSIS_RULE.md` is public (RUN_PLAN step S6); both values are reported, the corrected one marked as a logged departure from the code (not from the rule). Item 7 is never decision-bearing (rule Section 7).

## C7. Reporting label "figure-insensitive" (rule Section 8 guardrail)

Not a factual error in the preamble, recorded here because it is a departure in reporting: the guardrail calls DINOv2 block 12 "figure-insensitive". The registers control and the real-figure contrasts show that saliency read from block 12 of the public DINOv2-B checkpoint puts less mass inside real filled (-0.077 [-0.110, -0.045]) and real outlined figures (-0.094 [-0.123, -0.064]) than inside the 180 degree control, and that a DINOv2-B trained with registers favours both (+0.091 and +0.053). The paper therefore describes that block as "inducer-favouring (public checkpoint)". The guardrail's purpose (no "reversal" or "suppression" language) is kept.

## What the corrected preamble would have said

"Disclosure of what is already known when this rule is written: the effects and rotation ratios of the nine readouts in Appendix B are published (`cbb/tab_sweep.tex`, from `v3/scores/results.json`) and were read while writing this rule. The aligned effects of all 18 readouts against the 180 and 90 degree controls were computed on 8 October 2026 and pushed to the remote repository on 9 October 2026 (commit a533d4b); they determine eligibility for the primary test. The rotation, alternating-rotation, core, real-figure and Ehrenstein scores of the other nine readouts in the version-3 pipeline are not known, and no ERF of any readout or block has been computed. Held-out MIT1003 CC values (`v2/anatomy/anatomy.json`) are known."
