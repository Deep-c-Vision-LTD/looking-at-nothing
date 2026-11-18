# Analysis rule: does rotation sensitivity track the effective receptive field of the block read?

Written 2026-11-17 (UTC; exact time and SHA-256 in `nn/ANALYSIS_RULE.sha256`) BEFORE any of the nine unscored
layer-sweep readouts was scored on the displays and BEFORE any effective receptive field (ERF) was computed. It fixes the
measures, the ERF definition, the primary test, the secondary analyses, the decision wording for every outcome and the
reporting rule. It is a fixed analysis rule for an internal follow-up, not a registered report.

Disclosure of what is already known when this rule is written: the effects and rotation ratios of the nine scored
readouts are published in Appendix B of the manuscript (`cbb/tab_sweep.tex`, from `v3/scores/results.json`) and were
read while writing this rule. Nothing is known about the nine unscored readouts on the displays, and no ERF of any
readout or block has been computed. Their held-out MIT1003 CC values (`v2/anatomy/anatomy.json`) are known.

Any departure from this file is logged in `nn/DEVIATIONS.md` with date, reason and effect, and the result under the rule
as written is reported next to it.

## 1. Readouts (18, all from `sweep.py`, seed 0)

One readout of the R4 design (`sweep.Readout`: ASPP with dilations 1, 3, 6, 12 plus a global-average-pooling branch,
128 channels, 1 x 1 head, learned centre bias that is switched off on the displays) per block, trained jointly per
backbone on the 12,803 sweep training images at 336 px for 6 epochs (loss KL + 2(1 - CC), AdamW 5e-4, cosine), final
epoch. Checkpoints: `sweep/<bb>/readout_L<L>.pt`. Block L = Hugging Face `hidden_states[L]` (output of block L, 1-based,
before any final layer normalisation); CLS token dropped for DINOv2 and CLIP.

| backbone (HF id) | patch | blocks | scored on the displays | not yet scored |
|---|---|---|---|---|
| DINOv2-B (`facebook/dinov2-base`) | 14 | 12 | L2, L8, L12 | L4, L6, L10 |
| CLIP-L/14 (`openai/clip-vit-large-patch14`) | 14 | 24 | L4, L12, L24 | L8, L16, L20 |
| SigLIP-L/16 (`google/siglip-large-patch16-256`) | 16 | 24 | L4, L8, L24 | L12, L16, L20 |

Stored scores of the nine scored readouts (read only): `v3/scores/raw_v3maps/<bb>_L<L>_s<336|448>.json` (stim_v3
conditions, measured on the saved float16 maps), `v3/scores/raw/<tag>.json` (stim_alt), `v2/scores/raw/<tag>.json`
(v2-style measures on all 740 stim_v3 images), summaries in `v3/scores/results.json`.

## 2. Displays, scoring pipeline and measures (identical to the manuscript)

Displays: `v2/stim_v3` (Kanizsa: 60 configurations = square/triangle x support ratio 0.3/0.5/0.7 x dark/light x 5
positions, 9 conditions: illusory, control180, control90, real, real_filled, realC, mis10, mis20, mis30; plus 200
Ehrenstein images that are scored but not analysed here) and `v3/stim_alt` (360 alternating-rotation images: 60
configurations x 10/20/30 deg x phases a, b). AOIs: `v3/stim_alt/aoi/<config>_aoi.npz` (identical to the stim_v3 AOIs,
with core30 and core45 added).

Pipeline per image and readout (exactly `v2/scores/score_v3.py` and `v3/scores/score_alt.py`, CPU, 3 threads): RGB
display -> `sweep.prep(rgb, s)` (cv2 INTER_AREA resize to s x s, backbone mean/std) for s = 336 and 448 ->
`sweep.feats` -> readout with `use_cb = False` -> sigmoid -> `cv2.resize` to 1024 x 768 (INTER_LINEAR) -> map divided by
its maximum and rounded to float16 (`common_alt.q16`) -> `common_alt.measures`. stim_v3 Kanizsa measures come from the
q16 map, identical to what `v3/scores/measure_v3maps.py` computed from the saved v2 float16 maps.

Measures (map clipped at 0): r = M(inside) / (M(inside) + M(inducer discs)); core r (D = 30 px) = M(core30) /
(M(core30) + M(discs)); core share = M(core30) / M(all). Per configuration the values at 336 and 448 px are averaged
(size average), as in the manuscript.

Effects and ratios (inside r unless stated; `v3/scores/analyze_alt.analyse`, imported unchanged):
- Kanizsa effect E = mean over the 60 configurations of r(illusory) - r(control180); 95% CI from 10,000 cluster
  resamples over the 12 shape x support ratio x polarity cells with positions resampled within cells, using the v2
  resample indices (`v2/scores/analyze_v3.IDX["kanizsa"]`, seed 2026).
- A readout is "eligible" for ratio-based measures if the 95% CI of E excludes 0 (the manuscript's rule).
- Uniform rotation ratio U(theta) = E(mis theta vs control180) / E(illusory vs control180), ratio of means.
- Alternating ratio A(theta): the same with alt theta = mean of phases a and b.
- Drop ratio D(theta) = (m(illusory) - m(alt theta)) / (m(illusory) - m(mis theta)), computed where the uniform-loss CI
  excludes 0. Figure-mass loss = r(illusory) - r(mis theta).
- Support-ratio slope = OLS slope of the per-configuration effect on support ratio.

Rotation sensitivity (primary dependent variable): S20 = 1 - U(20), inside r, pooled over shapes, size average. It
measures loss of the effect under a 20 deg uniform rotation of either sign of effect (as in the manuscript, rotation
sensitivity marks non-local sensitivity to inducer alignment, not figural priority).

## 3. Local nulls and the readout blur scale

The 15 local nulls (Harris corner, ink, Sobel edge energy, Gaussian sigma 8/16/25/40/60 px) are image-only and already
scored on all 900 relevant images in `v3/scores/raw_nulls.json` (code `v3/scores/local_nulls_alt.py`, maps verbatim
from `v2/verify/local_nulls.py`); they are read, not recomputed.

The sweep readouts have no explicit output blur; no blur scale was set or estimated for them in the manuscript (the
"model's own blur scale" there refers to DeepGaze IIE and MSDB, read from their checkpoints). Rule: the architectural
output blur of a readout at input size s is the standard deviation of the bilinear upsampling kernel from its token
grid, Delta / sqrt(6) per axis (Delta = token spacing in display px: 1024/g horizontally, 768/g vertically, g = s / patch),
summarised as the geometric mean over the two axes. This is 15.1 px (336) and 11.3 px (448) for patch 14 and 17.2 px
(336) and 12.9 px (448) for patch 16, so for every sweep readout the nulls at its blur scale are sigma = 8, 16 and 25 px
(the bracket), with sigma = 16 px as the single reference. Null comparisons are descriptive and per readout: E against
the sigma-16 corner null (and the 8 to 25 bracket); U(20) and A(20) against the smallest corner-null ratio in the
bracket; figure-mass loss at 20 and 30 deg against the largest null loss in the bracket; square D(20) against the
null range; support-ratio slope against the corner nulls in the bracket.

## 4. Effective receptive field (ERF)

Computed on the CPU (no training, no new data), for every one of the 18 readouts and the block it reads, at s = 336
and 448.

Image set (fixed): 24 displays = 12 configurations x {illusory, control180}. One configuration per design cell: cells
ordered as in `v3/scores/analyze_alt.Design().keys` (sorted by str of (shape, support ratio, polarity)); cell i uses
position p = i mod 5, so both shapes, all support ratios, both polarities and all five positions occur.

Output locations per display (display coordinates, from the configuration's AOI file): (a) inside set, 5 locations:
the centroid of the inside AOI and the centroids of its four quadrants split at that centroid; (b) inducer set: the
centroid of each connected component (8-connectivity) of the inducer-disc AOI (4 for squares, 3 for triangles; the
count is asserted). Primary ERF uses all locations of a display (8 or 9), each display weighted equally.

Readout ERF. For location (X, Y) the output pixel is (i, j) = (round(v), round(u)), u = (X + 0.5) s / 1024 - 0.5,
v = (Y + 0.5) s / 768 - 0.5 (clipped to the s x s grid). Target y = pre-sigmoid readout output at (i, j) (the s x s
bilinearly upsampled logit map of `sweep.Readout`, centre bias off). Gradient magnitude per model-input pixel p:
G(p) = sum over colour channels c of |dy / dx_c(p)| / std_c (x = normalised input, so this is the gradient with respect
to raw pixel intensity), weights w(p) = G(p) / sum G. Distances d(p) are Euclidean in display pixels between the centre
of p and the centre of output pixel (i, j), both mapped to display coordinates ((col + 0.5) 1024 / s - 0.5,
(row + 0.5) 768 / s - 0.5). Radial cumulative mass CDF(r) = sum of w(p) over d(p) <= r for r = 0, 1, ..., 1280 px.
CDFs are averaged over the locations of a display, then over the 24 displays. ERF50(s) and ERF90(s) = smallest r with
mean CDF >= 0.50 and >= 0.90. Primary predictor: ERF50 = mean of ERF50(336) and ERF50(448), in display pixels.
(The INTER_AREA resize from the display to the model input is not differentiated; it adds at most about 1.5 display
px of spread and is ignored.) Because the pre-sigmoid and post-sigmoid maps differ only by a per-location factor,
per-location normalisation makes the choice irrelevant.

Backbone-only ERF of the read block (comparison, not a predictor in the primary test): target y = q . h_L[t], where
h_L is `hidden_states[L]` (patch tokens only), t is the token containing (u, v) (row floor(v / patch), column
floor(u / patch), clipped), and q is a standard normal vector of the hidden size drawn with
`torch.Generator().manual_seed(1000003 * display_index + 1009 * location_index + L)` (one fresh projection per display,
location and block; same q at both sizes). Same G, w, distance (from the token centre) and CDF summary, giving
bbERF50 and bbERF90.

Also reported: ERF50/90 per size, in display px and in model-input px; per-display ERF50 (spread over displays);
inside-set-only and inducer-set-only ERFs.

## 5. Primary test

Set: the eligible readouts among the 18 (Section 2), n of them. Statistic: Spearman rho (average ranks for ties)
between ERF50 and S20 across the n readouts.
- Uncertainty from the stimuli: for each of the 10,000 v2 cluster resamples, recompute S20 of every eligible readout
  from the resampled means (same resample for all readouts, size-averaged per-configuration values, eligibility fixed)
  and rho with ERF50 held fixed; 95% percentile CI and the share of resamples with rho > 0.
- Association test: two-sided permutation p for rho (all n! permutations if n <= 9, otherwise 100,000 random
  permutations, numpy RandomState(2026)); p = share with |rho_perm| >= |rho_obs| - 1e-12.
- Within each backbone with at least 3 eligible readouts: rho_b (descriptive). Pooled within-backbone statistic
  T = sum_b n_b rho_b / sum_b n_b over those backbones; two-sided stratified permutation p (ERF50 permuted within
  backbone, 100,000 permutations, RandomState(2026)).
- Backbone identity versus ERF (eligible readouts): R2_bb = between-backbone share of the variance of S20 (one-way,
  backbone means); R2_erf = R2 of the OLS fit S20 ~ log2(ERF50); R2_both = R2 of S20 ~ log2(ERF50) + backbone dummies;
  partial R2 of ERF given backbone = (R2_both - R2_bb) / (1 - R2_bb), with the sign of the ERF coefficient in the joint
  model; adjusted R2 for each model (adj = 1 - (1 - R2)(n - 1)/(n - k - 1), k = number of predictors).

## 6. Decision rule (applies to the primary test only; wording fixed now)

Let rho, CI, p, T, p_T, rho_b, R2 values be as in Section 5.

SUPPORTS if n >= 8 and all of:
- S1: rho >= +0.50, p < 0.05 and the lower CI bound > 0;
- S2: T > 0, p_T < 0.10, and rho_b > 0 in at least two backbones with >= 3 eligible readouts (if fewer than two
  backbones qualify, S2 fails);
- S3: adjusted R2_erf >= adjusted R2_bb, and in the joint model the ERF coefficient is positive with partial R2 >= 0.25.

Wording: "Across the n layer-sweep readouts with a reliable aligned effect, loss of the Kanizsa effect under a 20 deg
inducer rotation increased with the effective receptive field of the readout (Spearman rho = .., 95% CI [..],
permutation p = ..), within networks (pooled within-network rho = .., p = ..) as well as across them, and the receptive
field accounted for at least as much variance as network identity (adjusted R2 .. vs ..). In this single-seed sweep,
rotation sensitivity therefore follows the spatial extent over which the readout integrates more closely than the
network it reads, as the receptive-field account predicts. Within each network receptive field and depth rise
together, so these data do not separate receptive field from depth."

CONTRADICTS if n >= 8 and either:
- C1: rho < 0 with p < 0.05; or
- C2: rho <= +0.10, T <= 0, and adjusted R2_bb - adjusted R2_erf >= 0.20.

Wording: "Loss of the Kanizsa effect under a 20 deg inducer rotation did not increase with the effective receptive
field of the readout (Spearman rho = .., 95% CI [..], permutation p = ..; pooled within-network rho = ..); network
identity accounted for more of its variance (adjusted R2 .. vs ..). The layer sweep therefore does not support the
proposal that rotation sensitivity follows the receptive field of the block read rather than the network." (C1 adds:
"Rotation sensitivity was larger for readouts with smaller receptive fields.")

INCONCLUSIVE otherwise. Wording by case:
- I-a (n < 8): "Only n of the 18 readouts had an aligned effect whose interval excluded zero, too few to test whether
  rotation sensitivity follows the receptive field; we report the values descriptively."
- I-b (S1 met, S2 not met): "Rotation sensitivity rose with receptive field across readouts (rho = ..) but not
  consistently within networks (pooled within-network rho = .., p = ..), so differences between networks, which
  confound training objective, data and architecture, could carry the association."
- I-c (S1 and S2 met, S3 not met): "Rotation sensitivity rose with receptive field across and within networks, but the
  receptive field did not explain more of its variance than network identity (adjusted R2 .. vs ..)."
- I-d (any other pattern): "The layer sweep neither supports nor rules out the receptive-field account (rho = ..,
  95% CI [..], p = ..; pooled within-network rho = ..; adjusted R2 for receptive field .. and for network identity ..)."

Inverted U (guardrail): wording that rotation sensitivity "first rises and then falls" or peaks at an intermediate
receptive field may be used only if all of: (i) in the OLS fit S20 ~ log2(ERF50) + log2(ERF50)^2 over the eligible
readouts the quadratic coefficient is negative and its 95% CI over the 10,000 resamples (ERF fixed) excludes 0;
(ii) the fitted vertex lies inside the observed log2(ERF50) range with at least two eligible readouts on each side;
(iii) in at least two of the three backbones the largest S20 among eligible blocks is at a block that is neither the
shallowest nor the deepest eligible block of that backbone. Otherwise the report states "no evidence for a peak at an
intermediate receptive field" and makes no inverted-U claim.

## 7. Secondary and sensitivity analyses (reported in full, never decision-bearing)

1. Same statistics (Section 5) with S30 = 1 - U(30), with alternating sensitivity 1 - A(20), and with the square-only
   drop ratio D(20) (readouts whose square uniform-loss CI excludes 0).
2. Same statistics with ERF90, bbERF50 and bbERF90 as predictor; with the inside-set-only and inducer-set-only ERF50.
3. Per size: S20 from the 336-only and 448-only analyses against ERF50(336) and ERF50(448).
4. Positive-effect eligible readouts only.
5. Fixation-accuracy filter: readouts with held-out MIT1003 CC >= (best CC of the same backbone) - 0.05.
6. All 18 readouts with the point ratio U(20) whether or not eligible (ratios of near-zero effects are unstable; flagged).
7. Ratio-free alignment sensitivity for all 18: L20 = sign(E) x mean figure-mass loss r(illusory) - r(mis20), Spearman
   with ERF50 (all 18, and within backbone).
8. Competing descriptors over the eligible readouts: R2 and Spearman rho for relative depth (L / number of blocks) and
   for training-objective family (DINOv2 self-supervised vs CLIP and SigLIP image-text, two groups); rho between
   ERF50 and relative depth across all 18 (collinearity).
9. For each of the 9 newly scored readouts, the full row of Appendix B (CC, E with CI, U(20), A(20), D(20), slope),
   plus core r and core share effects, control90 effect, real and real_filled against control180, U(10), U(30),
   A(10), A(30), and the null comparisons of Section 3. The 9 scored readouts are re-read from `v3/scores` (not
   rescored) apart from the one sanity rescore.

## 8. Reporting rule

- Every outcome is reported whatever its direction: the decision category with its fixed wording, all numbers of
  Sections 5 and 7, and the per-readout table for all 18 readouts. No readout, block, size or location set is dropped
  after the results are seen.
- The readouts are single-seed (seed 0); there is no seed variance. All results are descriptive of these 18 readouts.
  Confidence intervals reflect stimulus sampling only.
- ERF and depth are collinear within each network; any statement must say so. Backbone identity confounds training
  objective with data, architecture size and patch size; "training objective" is not isolated by this design.
- Manuscript guardrails stay in force: no "reversal" or "suppression" language for DINOv2 block 12 (it is
  figure-insensitive); no inverted-U claim unless Section 6 allows it; no human-likeness claim from any index; local
  structure can produce the inside contrast, so E alone is not evidence of figural processing.
- The sanity rescore of one stored readout must reproduce the stored per-image r values to floating-point noise
  (max |difference| < 1e-4) and the stored E, U(20), U(30), A(20), D(20) and slope to the printed precision before any
  new result is interpreted.
