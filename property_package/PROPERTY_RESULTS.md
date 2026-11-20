# Property package: does rotation sensitivity track the effective receptive field of the block read?

Results of `nn/ANALYSIS_RULE.md` applied to the 18 layer-sweep readouts (9 stored Appendix B readouts + 9 newly scored). Generated 2026-11-17T09:47:01Z by `nn/property_results.py` from the outputs of `nn/run_all.sh` (2026-11-17, 08:08 to 09:30 UTC).

Rule SHA-256 `8e59fe87433a101aba97c99bca8e37a9e36faf08b47c9a70b1c6a141de3b403a`, identical to the value recorded in `nn/ANALYSIS_RULE.sha256` (ANALYSIS_RULE.md recorded_utc 2026-11-17T07:53:35Z); the rule file is read-only and was not changed.

Scope. Single seed (seed 0), no seed variance; all results describe these 18 readouts only. Confidence intervals reflect stimulus sampling only (10,000 cluster resamples, v2 indices, seed 2026). ERF and depth are collinear within each network. Backbone identity confounds training objective with data, architecture size and patch size, so 'training objective' is not isolated by this design.

## 1. Decision under the rule (primary test)

**INCONCLUSIVE, case I-d.** Pre-written wording that applies:

> The layer sweep neither supports nor rules out the receptive-field account (rho = -0.37, 95% CI [-0.62, -0.14], p = .218; pooled within-network rho = -0.27; adjusted R2 for receptive field 0.18 and for network identity -0.10).

Criteria (Section 6):

| criterion | requirement | observed | met |
|---|---|---|---|
| n | >= 8 eligible readouts for S or C | n = 13 | yes |
| S1 | rho >= +0.50, p < .05, lower CI > 0 | rho = -0.366, p = .218, CI [-0.624, -0.140] | no |
| S2 | T > 0, p_T < .10, rho_b > 0 in >= 2 backbones with >= 3 eligible | T = -0.269, p_T = .448; rho_b: clip -0.50, dinov2 -0.30, siglip -0.10 (0 positive) | no |
| S3 | adj R2_erf >= adj R2_bb, ERF coefficient > 0 in joint model, partial R2 >= .25 | adj R2_erf 0.179 vs adj R2_bb -0.097; joint coefficient -1.384; partial R2 0.289 | no |
| C1 | rho < 0 with p < .05 | rho = -0.366, p = .218 | no |
| C2 | rho <= +0.10, T <= 0, adj R2_bb - adj R2_erf >= .20 | -0.366, -0.269, -0.276 | no |

Exact statistics (eligible readouts, ERF50 = readout ERF50, outcome S20 = 1 - U(20), inside r, pooled over shapes, size average):

- n = 13: DINOv2-B L4, DINOv2-B L6, DINOv2-B L8, DINOv2-B L10, DINOv2-B L12, CLIP-L L8, CLIP-L L12, CLIP-L L16, SigLIP-L L4, SigLIP-L L8, SigLIP-L L12, SigLIP-L L20, SigLIP-L L24.
- Spearman rho = -0.3659; 95% percentile CI over 10,000 stimulus resamples (ERF50 fixed, eligibility fixed) [-0.6245, -0.1403]; share of resamples with rho > 0 = 0.0014.
- Two-sided permutation p = 0.2179 (100000 random permutations, RandomState(2026); Monte Carlo SE about 0.0013).
- Within backbone: clip rho_b = -0.500 (n 3), dinov2 rho_b = -0.300 (n 5), siglip rho_b = -0.100 (n 5); pooled T = -0.2692, stratified permutation p = 0.4478 (100,000 permutations within backbone).
- Variance: R2_bb = 0.086 (adjusted -0.097); R2_erf (S20 ~ log2 ERF50) = 0.247 (adjusted 0.179, slope -1.328 per doubling); R2_both = 0.350 (adjusted 0.134); partial R2 of ERF given backbone = 0.289, ERF coefficient in the joint model -1.384 (negative).
- The stimulus-resampling CI of rho excludes 0 while the permutation p does not. The CI holds ERF50 fixed and only asks how stable rho is to the choice of displays; the rule's association test is the permutation test, so S1 and C1 are not met, C2 is not met either, and the case is I-d.

Inverted-U guardrail (Section 6):

- (i) quadratic coefficient of S20 ~ log2(ERF50) + log2(ERF50)^2 = -0.807, 95% CI [-10.551, +11.296]: not met.
- (ii) fitted vertex at ERF50 = 158.4 px, outside the observed range (238.5 to 325.5 px), 0 readouts left / 13 right: not met.
- (iii) largest S20 at an interior eligible block: DINOv2-B yes (largest S20 at L10, an interior eligible block); CLIP-L/14 no (largest S20 at L8, the shallowest eligible block); SigLIP-L/16 yes (largest S20 at L20, an interior eligible block): met on its own.
- Statement required by the rule: **no evidence for a peak at an intermediate receptive field.** No inverted-U claim is made.

## 2. Per-readout table (all 18)

Effects are inside r, illusory minus control180, mean over 60 configurations, size average of 336 and 448 px, 95% cluster-bootstrap CI. Ratios: U(theta) uniform rotation ratio, A(20) alternating ratio (computed only for eligible readouts, n.e. otherwise); D(20) alternating drop ratio (computed where the uniform-loss CI excludes 0). S20 = 1 - U(20). ERF50/ERF90: readout ERF in display px (size average; 336 / 448 in brackets); bbERF50: block-token ERF50. CC: held-out MIT1003 CC from the sweep (`v2/anatomy/anatomy.json`). Source: stored = re-read from `v3/scores` (not rescored); new = scored in this package.

| backbone | block | source | CC | E vs control180 [95% CI] | eligible | U(20) | U(30) | A(20) | D(20) | slope [95% CI] | S20 | ERF50 (336 / 448) | ERF90 | bbERF50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DINOv2-B | 2 | stored | 0.626 | -0.000 [-0.017, +0.016] | no | n.e. | n.e. | n.e. | n.e. | -0.080 [-0.183, +0.020] | n.e. | 244.0 (263 / 225) | 516.0 | 30.5 |
| DINOv2-B | 4 | new | 0.708 | +0.051 [+0.043, +0.059] | yes | 0.74 [0.65, 0.83] | 0.58 [0.47, 0.70] | 0.76 [0.67, 0.86] | 0.90 [0.72, 1.14] | +0.022 [-0.036, +0.081] | +0.26 | 287.5 (311 / 264) | 560.5 | 74.0 |
| DINOv2-B | 6 | new | 0.743 | +0.082 [+0.062, +0.102] | yes | 0.65 [0.52, 0.76] | 0.33 [0.14, 0.48] | 0.91 [0.81, 1.00] | 0.25 [-0.01, 0.42] | +0.164 [+0.075, +0.247] | +0.35 | 300.5 (321 / 280) | 574.5 | 140.0 |
| DINOv2-B | 8 | stored | 0.746 | +0.035 [+0.023, +0.046] | yes | 1.57 [1.36, 1.94] | 1.22 [1.04, 1.46] | 0.96 [0.87, 1.10] | -0.06 [-0.32, 0.13] | +0.007 [-0.073, +0.088] | -0.57 | 312.5 (332 / 293) | 575.5 | 178.0 |
| DINOv2-B | 10 | new | 0.734 | -0.029 [-0.045, -0.013] | yes | 0.24 [-1.09, 0.80] | 0.51 [-0.20, 0.99] | 0.96 [0.71, 1.25] | 0.05 [-0.51, 0.46] | -0.061 [-0.179, +0.054] | +0.76 | 294.5 (329 / 260) | 570.0 | 214.0 |
| DINOv2-B | 12 | stored | 0.731 | -0.098 [-0.126, -0.070] | yes | 0.56 [0.41, 0.68] | 0.41 [0.27, 0.53] | 0.79 [0.71, 0.87] | 0.48 [0.28, 0.68] | -0.240 [-0.362, -0.130] | +0.44 | 288.5 (320 / 257) | 556.0 | 239.5 |
| CLIP-L/14 | 4 | stored | 0.657 | -0.000 [-0.017, +0.017] | no | n.e. | n.e. | n.e. | 0.45 [-0.47, 0.83] | -0.064 [-0.177, +0.043] | n.e. | 259.0 (276 / 242) | 533.5 | 77.0 |
| CLIP-L/14 | 8 | new | 0.737 | +0.041 [+0.021, +0.061] | yes | 0.22 [-0.72, 0.57] | -0.17 [-1.44, 0.28] | 0.63 [0.11, 0.84] | 0.47 [0.33, 0.59] | +0.034 [-0.097, +0.161] | +0.78 | 238.5 (256 / 221) | 532.5 | 138.0 |
| CLIP-L/14 | 12 | stored | 0.760 | +0.060 [+0.048, +0.073] | yes | 0.90 [0.73, 1.04] | 0.58 [0.32, 0.81] | 0.87 [0.76, 0.97] | n.e. | +0.007 [-0.072, +0.095] | +0.10 | 318.0 (332 / 304) | 580.0 | 182.5 |
| CLIP-L/14 | 16 | new | 0.741 | +0.018 [+0.005, +0.030] | yes | 1.16 [0.79, 2.36] | 1.01 [0.56, 2.56] | 1.31 [0.57, 2.42] | n.e. | +0.075 [+0.008, +0.142] | -0.16 | 317.5 (335 / 300) | 566.5 | 258.5 |
| CLIP-L/14 | 20 | new | 0.726 | -0.010 [-0.024, +0.003] | no | n.e. | n.e. | n.e. | n.e. | +0.046 [-0.014, +0.103] | n.e. | 322.5 (338 / 307) | 564.5 | 287.5 |
| CLIP-L/14 | 24 | stored | 0.718 | +0.014 [-0.012, +0.042] | no | n.e. | n.e. | n.e. | n.e. | +0.228 [+0.107, +0.351] | n.e. | 319.5 (334 / 305) | 560.0 | 314.0 |
| SigLIP-L/16 | 4 | stored | 0.691 | +0.020 [+0.009, +0.029] | yes | 0.46 [-0.11, 0.89] | 0.38 [-0.28, 0.86] | 1.14 [0.91, 1.59] | -0.27 [-2.74, 0.13] | -0.030 [-0.104, +0.035] | +0.54 | 249.0 (259 / 239) | 542.0 | 116.0 |
| SigLIP-L/16 | 8 | stored | 0.757 | +0.055 [+0.042, +0.069] | yes | 0.55 [0.32, 0.76] | 0.24 [-0.04, 0.44] | 0.89 [0.74, 1.02] | 0.24 [-0.08, 0.43] | +0.071 [-0.008, +0.151] | +0.45 | 277.5 (284 / 271) | 561.5 | 155.5 |
| SigLIP-L/16 | 12 | new | 0.748 | -0.021 [-0.033, -0.008] | yes | 0.42 [-0.74, 0.83] | 0.58 [-0.36, 1.09] | 0.76 [0.03, 1.26] | 0.41 [-0.95, 0.89] | -0.013 [-0.089, +0.070] | +0.58 | 318.0 (326 / 310) | 562.0 | 203.5 |
| SigLIP-L/16 | 16 | new | 0.737 | +0.002 [-0.012, +0.016] | no | n.e. | n.e. | n.e. | n.e. | -0.107 [-0.181, -0.035] | n.e. | 313.0 (323 / 303) | 554.0 | 240.0 |
| SigLIP-L/16 | 20 | new | 0.728 | +0.047 [+0.020, +0.074] | yes | 0.41 [-0.49, 0.82] | -0.23 [-1.95, 0.36] | 0.22 [-1.97, 0.90] | 1.32 [0.31, 3.68] | -0.118 [-0.277, +0.042] | +0.59 | 318.5 (326 / 311) | 556.0 | 252.0 |
| SigLIP-L/16 | 24 | stored | 0.721 | +0.145 [+0.122, +0.170] | yes | 0.84 [0.71, 0.97] | 0.50 [0.41, 0.62] | 0.70 [0.60, 0.80] | 1.86 [0.90, 7.65] | -0.004 [-0.177, +0.166] | +0.16 | 325.5 (333 / 318) | 554.5 | 288.5 |

The nine stored rows equal Appendix B (`cbb/tab_sweep.tex`, from `v3/scores/results.json`) exactly; the check is listed in Section 8.

### 2a. Local nulls at the readout's blur scale (all 18)

Architectural output blur (SD of the bilinear kernel, Delta/sqrt(6), geometric mean of the two axes): patch 14 15.1 px (336) and 11.3 px (448); patch 16 17.2 px and 12.9 px. For every readout the nulls at its blur scale are therefore sigma = 8, 16 and 25 px (bracket), sigma = 16 px the reference. Null values in the bracket: corner E = +0.176, +0.211, +0.237 (sigma 8 / 16 / 25); smallest corner U(20) = 1.030, smallest corner A(20) = 1.028; largest null figure-mass loss over corner, ink and edge nulls in the bracket = -0.0065 (20 deg) and +0.0003 (30 deg); null square D(20) range 1.0003 to 1.0175; corner slopes -0.040, -0.114, -0.152.

| readout | E (corner s16 E = +0.211) | U(20) CI upper < smallest corner U(20) (1.03)? | A(20) CI upper < smallest corner A(20) (1.03)? | loss 20 deg [95% CI] (null max -0.0065) | loss 30 deg [95% CI] (null max +0.0003) | square D(20) (null 1.000 to 1.018) | slope (corner -0.040 / -0.114 / -0.152) |
|---|---|---|---|---|---|---|---|
| DINOv2-B L2 | -0.000 | n.e. | n.e. | +0.0052 [-0.0010, +0.0118] | +0.0040 [-0.0034, +0.0118] | n.e. | -0.080 |
| DINOv2-B L4 | +0.051 | yes (0.834) | yes (0.859) | +0.0133 [+0.0080, +0.0190] | +0.0211 [+0.0142, +0.0283] | 0.92 | +0.022 |
| DINOv2-B L6 | +0.082 | yes (0.759) | yes (1.004) | +0.0288 [+0.0182, +0.0409] | +0.0550 [+0.0424, +0.0686] | 0.32 | +0.164 |
| DINOv2-B L8 | +0.035 | no (1.940) | no (1.104) | -0.0197 [-0.0249, -0.0142] | -0.0078 [-0.0139, -0.0014] | 0.19 | +0.007 |
| DINOv2-B L10 | -0.029 | yes (0.803) | no (1.251) | -0.0216 [-0.0369, -0.0065] | -0.0139 [-0.0283, -0.0002] | n.e. | -0.061 |
| DINOv2-B L12 | -0.098 | yes (0.682) | yes (0.868) | -0.0430 [-0.0535, -0.0318] | -0.0577 [-0.0716, -0.0443] | 0.86 | -0.240 |
| CLIP-L L4 | -0.000 | n.e. | n.e. | +0.0058 [+0.0011, +0.0109] | +0.0047 [-0.0016, +0.0111] | n.e. | -0.064 |
| CLIP-L L8 | +0.041 | yes (0.567) | yes (0.841) | +0.0317 [+0.0253, +0.0380] | +0.0477 [+0.0414, +0.0546] | 0.53 | +0.034 |
| CLIP-L L12 | +0.060 | no (1.043) | yes (0.967) | +0.0063 [-0.0026, +0.0153] | +0.0254 [+0.0112, +0.0402] | 0.80 | +0.007 |
| CLIP-L L16 | +0.018 | no (2.363) | no (2.424) | -0.0027 [-0.0105, +0.0047] | -0.0002 [-0.0111, +0.0100] | n.e. | +0.075 |
| CLIP-L L20 | -0.010 | n.e. | n.e. | -0.0120 [-0.0238, +0.0006] | -0.0196 [-0.0326, -0.0056] | n.e. | +0.046 |
| CLIP-L L24 | +0.014 | n.e. | n.e. | +0.0104 [-0.0083, +0.0279] | -0.0015 [-0.0226, +0.0191] | n.e. | +0.228 |
| SigLIP-L L4 | +0.020 | yes (0.890) | no (1.586) | +0.0106 [+0.0019, +0.0197] | +0.0124 [+0.0023, +0.0226] | -0.10 | -0.030 |
| SigLIP-L L8 | +0.055 | yes (0.756) | yes (1.024) | +0.0248 [+0.0135, +0.0355] | +0.0421 [+0.0309, +0.0526] | 0.33 | +0.071 |
| SigLIP-L L12 | -0.021 | yes (0.833) | no (1.259) | -0.0120 [-0.0200, -0.0040] | -0.0087 [-0.0185, +0.0019] | n.e. | -0.013 |
| SigLIP-L L16 | +0.002 | n.e. | n.e. | -0.0019 [-0.0193, +0.0146] | +0.0238 [+0.0049, +0.0424] | 1.53 | -0.107 |
| SigLIP-L L20 | +0.047 | yes (0.821) | yes (0.900) | +0.0279 [+0.0083, +0.0473] | +0.0582 [+0.0358, +0.0817] | 2.21 | -0.118 |
| SigLIP-L L24 | +0.145 | yes (0.968) | yes (0.797) | +0.0235 [+0.0046, +0.0420] | +0.0721 [+0.0500, +0.0937] | n.e. | -0.004 |

Reading of the null comparisons (descriptive, per readout). The largest aligned effect of any readout in magnitude (0.145) is below the sigma-16 corner null's (+0.211), and the corner nulls lose nothing under rotation (U(20) and A(20) above 1 at every sigma in the bracket). Local structure can produce the inside contrast, so E alone is not evidence of figural processing.

### 2b. Newly scored readouts: remaining Appendix B style measures (Section 7, item 9)

| readout | core r effect | core share effect | illusory vs control90 | real vs control180 | real_filled vs control180 | U(10) | A(10) | A(30) | D(10) | D(30) | square D(20) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DINOv2-B L4 | +0.034 [+0.027, +0.042] | +0.014 [+0.011, +0.018] | +0.046 [+0.037, +0.055] | +0.133 [+0.113, +0.154] | +0.145 [+0.127, +0.162] | 0.90 [0.85, 0.95] | 0.94 [0.89, 0.99] | 0.62 [0.50, 0.74] | 0.62 [0.21, 0.91] | 0.92 [0.75, 1.09] | 0.92 [0.80, 1.04] |
| DINOv2-B L6 | +0.079 [+0.070, +0.088] | +0.040 [+0.034, +0.045] | +0.110 [+0.088, +0.130] | +0.162 [+0.149, +0.176] | +0.134 [+0.123, +0.145] | 0.85 [0.77, 0.93] | 1.00 [0.94, 1.09] | 0.67 [0.49, 0.80] | -0.01 [-1.22, 0.31] | 0.49 [0.37, 0.61] | 0.32 [-0.28, 0.52] |
| DINOv2-B L10 | -0.029 [-0.045, -0.015] | -0.019 [-0.028, -0.010] | +0.032 [+0.013, +0.053] | +0.001 [-0.017, +0.019] | -0.017 [-0.039, +0.006] | 0.73 [-0.23, 1.22] | 1.04 [0.77, 1.35] | 0.99 [0.74, 1.41] | n.e. | 0.03 [-1.93, 1.02] | n.e. |
| CLIP-L L8 | +0.039 [+0.024, +0.054] | +0.019 [+0.012, +0.027] | +0.076 [+0.057, +0.097] | +0.145 [+0.121, +0.171] | +0.147 [+0.122, +0.172] | 0.68 [0.18, 0.87] | 0.89 [0.65, 1.01] | 0.42 [-0.27, 0.68] | 0.35 [-0.06, 0.56] | 0.50 [0.40, 0.59] | 0.53 [0.38, 0.64] |
| CLIP-L L16 | +0.011 [-0.002, +0.023] | -0.003 [-0.010, +0.004] | +0.031 [+0.016, +0.046] | +0.117 [+0.087, +0.144] | +0.128 [+0.105, +0.148] | 1.35 [1.00, 2.63] | 1.70 [1.21, 3.11] | 0.97 [-0.05, 2.08] | 1.99 [0.08, 13.54] | n.e. | n.e. |
| CLIP-L L20 | -0.014 [-0.029, -0.000] | -0.016 [-0.025, -0.008] | -0.002 [-0.017, +0.012] | +0.087 [+0.061, +0.111] | +0.093 [+0.062, +0.126] | n.e. | n.e. | n.e. | n.e. | 1.62 [1.00, 4.78] | n.e. |
| SigLIP-L L12 | -0.023 [-0.036, -0.010] | -0.012 [-0.020, -0.005] | +0.002 [-0.007, +0.009] | +0.063 [+0.043, +0.081] | +0.076 [+0.047, +0.104] | 0.60 [-0.10, 0.84] | 0.95 [0.55, 1.36] | 0.80 [0.19, 1.38] | 0.11 [-1.42, 0.69] | n.e. | n.e. |
| SigLIP-L L16 | +0.007 [-0.007, +0.023] | -0.006 [-0.013, -0.000] | +0.056 [+0.039, +0.073] | +0.117 [+0.094, +0.139] | +0.221 [+0.191, +0.247] | n.e. | n.e. | n.e. | n.e. | -0.44 [-5.27, 0.32] | 1.53 [0.52, 9.67] |
| SigLIP-L L20 | +0.050 [+0.024, +0.077] | +0.027 [+0.012, +0.042] | +0.083 [+0.054, +0.112] | +0.081 [+0.046, +0.120] | +0.164 [+0.139, +0.190] | 0.75 [0.39, 0.95] | 0.44 [-0.60, 0.75] | 0.24 [-2.07, 1.02] | 2.21 [1.03, 10.38] | 0.62 [-0.02, 1.13] | 2.21 [1.17, 7.91] |

### 2c. ERF details (Section 4, 'also reported')

Readout ERF and block-token ERF per input size, in display px and model-input px; per-display ERF50 over the 24 displays (median and range); inside-set-only and inducer-set-only ERF50 (size average).

| readout | ERF50 336 / 448 (display) | ERF90 336 / 448 (display) | ERF50 336 / 448 (model px) | per-display ERF50 336: median (min to max) | per-display ERF50 448: median (min to max) | ERF50 inside / inducer | bbERF50 336 / 448 | bbERF90 (size avg) | bbERF50 inside / inducer |
|---|---|---|---|---|---|---|---|---|---|
| DINOv2-B L2 | 263 / 225 | 545 / 487 | 93 / 112 | 263 (227 to 302) | 223 (200 to 244) | 250.5 / 233.0 | 32 / 29 | 485.5 | 39.0 / 24.0 |
| DINOv2-B L4 | 311 / 264 | 597 / 524 | 116 / 132 | 310 (282 to 347) | 258 (237 to 353) | 290.0 / 285.0 | 85 / 63 | 469.0 | 50.5 / 141.5 |
| DINOv2-B L6 | 321 / 280 | 597 / 552 | 121 / 143 | 327 (257 to 375) | 284 (238 to 333) | 289.5 / 315.5 | 154 / 126 | 443.5 | 115.5 / 174.0 |
| DINOv2-B L8 | 332 / 293 | 598 / 553 | 125 / 152 | 330 (287 to 377) | 284 (252 to 338) | 304.5 / 326.5 | 192 / 164 | 479.0 | 150.5 / 212.0 |
| DINOv2-B L10 | 329 / 260 | 596 / 544 | 125 / 132 | 330 (291 to 370) | 257 (230 to 299) | 292.5 / 298.5 | 236 / 192 | 525.5 | 183.0 / 254.0 |
| DINOv2-B L12 | 320 / 257 | 587 / 525 | 121 / 130 | 316 (264 to 387) | 257 (214 to 314) | 284.0 / 295.5 | 268 / 211 | 546.0 | 213.5 / 272.0 |
| CLIP-L L4 | 276 / 242 | 562 / 505 | 103 / 120 | 275 (246 to 306) | 242 (225 to 274) | 259.0 / 257.5 | 76 / 78 | 489.0 | 71.0 / 83.0 |
| CLIP-L L8 | 256 / 221 | 573 / 492 | 94 / 110 | 250 (240 to 314) | 221 (208 to 228) | 235.0 / 241.5 | 141 / 135 | 456.5 | 122.0 / 166.0 |
| CLIP-L L12 | 332 / 304 | 611 / 549 | 125 / 162 | 332 (312 to 373) | 303 (284 to 359) | 320.0 / 315.5 | 189 / 176 | 512.5 | 154.5 / 225.0 |
| CLIP-L L16 | 335 / 300 | 595 / 538 | 127 / 159 | 336 (312 to 375) | 298 (272 to 355) | 318.0 / 317.0 | 269 / 248 | 550.0 | 217.5 / 305.5 |
| CLIP-L L20 | 338 / 307 | 597 / 532 | 127 / 162 | 342 (297 to 375) | 302 (273 to 371) | 326.0 / 316.0 | 298 / 277 | 556.5 | 254.5 / 325.5 |
| CLIP-L L24 | 334 / 305 | 586 / 534 | 126 / 160 | 338 (305 to 372) | 308 (266 to 364) | 323.0 / 314.5 | 327 / 301 | 560.0 | 284.5 / 352.5 |
| SigLIP-L L4 | 259 / 239 | 549 / 535 | 98 / 119 | 260 (228 to 291) | 238 (215 to 263) | 254.0 / 242.5 | 126 / 106 | 466.0 | 100.0 / 140.0 |
| SigLIP-L L8 | 284 / 271 | 572 / 551 | 104 / 135 | 284 (242 to 309) | 264 (252 to 295) | 277.0 / 277.5 | 165 / 146 | 473.0 | 131.5 / 189.0 |
| SigLIP-L L12 | 326 / 310 | 569 / 555 | 123 / 158 | 325 (275 to 359) | 310 (285 to 334) | 314.0 / 326.5 | 217 / 190 | 511.0 | 174.5 / 241.0 |
| SigLIP-L L16 | 323 / 303 | 562 / 546 | 121 / 153 | 326 (270 to 354) | 304 (279 to 325) | 315.0 / 310.5 | 254 / 226 | 524.5 | 211.5 / 278.5 |
| SigLIP-L L20 | 326 / 311 | 562 / 550 | 122 / 159 | 327 (263 to 358) | 314 (276 to 334) | 320.5 / 314.5 | 264 / 240 | 531.0 | 221.5 / 291.5 |
| SigLIP-L L24 | 333 / 318 | 558 / 551 | 125 / 165 | 334 (291 to 364) | 322 (285 to 340) | 326.0 / 325.0 | 299 / 278 | 544.0 | 262.0 / 325.5 |

Ranges over the 18 readouts: readout ERF50 238.5 to 325.5 px; block-token ERF50 30.5 to 314.0 px. Spearman rho with relative depth (all 18): readout ERF50 +0.76, block-token ERF50 +0.90; readout vs block-token ERF50 +0.86.

## 3. Secondary and sensitivity analyses (Section 7; reported in full, never decision-bearing)

Same statistics as Section 5 unless stated. CI = 95% percentile interval over 10,000 stimulus resamples with the predictor fixed; p = two-sided permutation p (100,000 random permutations, RandomState(2026)); T = pooled within-backbone rho with stratified permutation p; R2 columns: predictor (log2) / backbone / both; partial = partial R2 of the predictor given backbone, with the sign of its joint-model coefficient.

| item | analysis | n | rho [95% CI] | perm p | within-backbone rho | T (p) | R2 x / bb / both | partial R2 (joint coef sign) |
|---|---|---|---|---|---|---|---|---|
| 1 | S30 = 1 - U(30) vs ERF50 | 13 | -0.27 [-0.45, -0.02] | .374 | clip -0.50 (n 3), dinov2 -0.10 (n 5), siglip -0.10 (n 5) | -0.19 (p .601) | 0.20 / 0.14 / 0.35 | 0.24 (-) |
| 1 | alternating sensitivity 1 - A(20) vs ERF50 | 13 | +0.23 [-0.05, +0.47] | .438 | clip -0.50 (n 3), dinov2 -0.90 (n 5), siglip +0.90 (n 5) | -0.12 (p .774) | 0.01 / 0.10 / 0.11 | 0.01 (+) |
| 1 | square-only D(20) vs ERF50 (square uniform-loss CI excludes 0) | 10 | +0.50 [+0.32, +0.66] | .143 | clip n/a (n 2), dinov2 -1.00 (n 4), siglip +1.00 (n 4) | +0.00 (p 1.000) | 0.28 / 0.09 / 0.42 | 0.37 (+) |
| 2 | S20 vs readout ERF90 | 13 | -0.51 [-0.67, -0.21] | .078 | clip -0.50 (n 3), dinov2 -0.50 (n 5), siglip +0.30 (n 5) | -0.19 (p .602) | 0.30 / 0.09 / 0.32 | 0.26 (-) |
| 2 | S20 vs block-token ERF50 | 13 | -0.16 [-0.45, +0.08] | .604 | clip -1.00 (n 3), dinov2 +0.60 (n 5), siglip -0.10 (n 5) | -0.04 (p .954) | 0.01 / 0.09 / 0.12 | 0.03 (-) |
| 2 | S20 vs block-token ERF90 | 13 | -0.25 [-0.51, -0.01] | .402 | clip -1.00 (n 3), dinov2 +0.50 (n 5), siglip -0.10 (n 5) | -0.08 (p .860) | 0.01 / 0.09 / 0.10 | 0.02 (-) |
| 2 | S20 vs inside-set-only ERF50 | 13 | -0.41 [-0.66, -0.23] | .164 | clip -0.50 (n 3), dinov2 -0.40 (n 5), siglip -0.10 (n 5) | -0.31 (p .379) | 0.23 / 0.09 / 0.35 | 0.29 (-) |
| 2 | S20 vs inducer-set-only ERF50 | 13 | -0.55 [-0.83, -0.25] | .053 | clip -1.00 (n 3), dinov2 -0.30 (n 5), siglip +0.10 (n 5) | -0.31 (p .377) | 0.26 / 0.09 / 0.35 | 0.29 (-) |
| 3 | 336 px only: S20(336) vs ERF50(336) | 12 | -0.47 [-0.77, -0.26] | .126 | clip n/a (n 2), dinov2 -0.10 (n 5), siglip -0.41 (n 5) | -0.26 (p .494) | 0.08 / 0.02 / 0.08 | 0.07 (-) |
| 3 | 448 px only: S20(448) vs ERF50(448) | 12 | -0.31 [-0.52, -0.06] | .330 | clip -0.50 (n 3), dinov2 -0.80 (n 5), siglip -0.40 (n 4) | -0.59 (p .082) | 0.21 / 0.12 / 0.65 | 0.60 (-) |
| 4 | positive-effect eligible readouts only | 10 | -0.49 [-0.72, -0.27] | .155 | clip -0.50 (n 3), dinov2 -0.50 (n 3), siglip -0.40 (n 4) | -0.46 (p .256) | 0.39 / 0.22 / 0.55 | 0.42 (-) |
| 5 | CC filter (CC >= best CC of backbone - 0.05) | 12 | -0.35 [-0.58, -0.13] | .261 | clip -0.50 (n 3), dinov2 -0.30 (n 5), siglip -0.20 (n 4) | -0.32 (p .361) | 0.24 / 0.06 / 0.42 | 0.38 (-) |
| 6 | all 18, S20 point values regardless of eligibility (unstable: flagged) | 18 | +0.21 none (point values) | .407 | clip +0.43 (n 6), dinov2 +0.26 (n 6), siglip +0.09 (n 6) | +0.26 (p .350) | 0.22 / 0.11 / 0.29 | 0.20 (+) |
| 7 | all 18, ratio-free L20 = sign(E) x [r(illusory) - r(mis20)] | 18 | -0.06 [-0.18, +0.13] | .807 | clip +0.14 (n 6), dinov2 -0.03 (n 6), siglip +0.43 (n 6) | +0.18 (p .519) | 0.01 / 0.05 / 0.07 | 0.02 (-) |

- S30 = 1 - U(30) vs ERF50: readouts DINOv2-B L4, DINOv2-B L6, DINOv2-B L8, DINOv2-B L10, DINOv2-B L12, CLIP-L L8, CLIP-L L12, CLIP-L L16, SigLIP-L L4, SigLIP-L L8, SigLIP-L L12, SigLIP-L L20, SigLIP-L L24.
- alternating sensitivity 1 - A(20) vs ERF50: readouts DINOv2-B L4, DINOv2-B L6, DINOv2-B L8, DINOv2-B L10, DINOv2-B L12, CLIP-L L8, CLIP-L L12, CLIP-L L16, SigLIP-L L4, SigLIP-L L8, SigLIP-L L12, SigLIP-L L20, SigLIP-L L24.
- square-only D(20) vs ERF50 (square uniform-loss CI excludes 0): readouts DINOv2-B L4, DINOv2-B L6, DINOv2-B L8, DINOv2-B L12, CLIP-L L8, CLIP-L L12, SigLIP-L L4, SigLIP-L L8, SigLIP-L L16, SigLIP-L L20.
- S20 vs readout ERF90: readouts DINOv2-B L4, DINOv2-B L6, DINOv2-B L8, DINOv2-B L10, DINOv2-B L12, CLIP-L L8, CLIP-L L12, CLIP-L L16, SigLIP-L L4, SigLIP-L L8, SigLIP-L L12, SigLIP-L L20, SigLIP-L L24.
- S20 vs block-token ERF50: readouts DINOv2-B L4, DINOv2-B L6, DINOv2-B L8, DINOv2-B L10, DINOv2-B L12, CLIP-L L8, CLIP-L L12, CLIP-L L16, SigLIP-L L4, SigLIP-L L8, SigLIP-L L12, SigLIP-L L20, SigLIP-L L24.
- S20 vs block-token ERF90: readouts DINOv2-B L4, DINOv2-B L6, DINOv2-B L8, DINOv2-B L10, DINOv2-B L12, CLIP-L L8, CLIP-L L12, CLIP-L L16, SigLIP-L L4, SigLIP-L L8, SigLIP-L L12, SigLIP-L L20, SigLIP-L L24.
- S20 vs inside-set-only ERF50: readouts DINOv2-B L4, DINOv2-B L6, DINOv2-B L8, DINOv2-B L10, DINOv2-B L12, CLIP-L L8, CLIP-L L12, CLIP-L L16, SigLIP-L L4, SigLIP-L L8, SigLIP-L L12, SigLIP-L L20, SigLIP-L L24.
- S20 vs inducer-set-only ERF50: readouts DINOv2-B L4, DINOv2-B L6, DINOv2-B L8, DINOv2-B L10, DINOv2-B L12, CLIP-L L8, CLIP-L L12, CLIP-L L16, SigLIP-L L4, SigLIP-L L8, SigLIP-L L12, SigLIP-L L20, SigLIP-L L24.
- 336 px only: S20(336) vs ERF50(336): readouts DINOv2-B L4, DINOv2-B L6, DINOv2-B L8, DINOv2-B L10, DINOv2-B L12, CLIP-L L8, CLIP-L L12, SigLIP-L L4, SigLIP-L L8, SigLIP-L L12, SigLIP-L L20, SigLIP-L L24.
- 448 px only: S20(448) vs ERF50(448): readouts DINOv2-B L4, DINOv2-B L6, DINOv2-B L8, DINOv2-B L10, DINOv2-B L12, CLIP-L L8, CLIP-L L12, CLIP-L L16, SigLIP-L L8, SigLIP-L L12, SigLIP-L L20, SigLIP-L L24.
- positive-effect eligible readouts only: readouts DINOv2-B L4, DINOv2-B L6, DINOv2-B L8, CLIP-L L8, CLIP-L L12, CLIP-L L16, SigLIP-L L4, SigLIP-L L8, SigLIP-L L20, SigLIP-L L24.
- CC filter (CC >= best CC of backbone - 0.05): readouts DINOv2-B L4, DINOv2-B L6, DINOv2-B L8, DINOv2-B L10, DINOv2-B L12, CLIP-L L8, CLIP-L L12, CLIP-L L16, SigLIP-L L8, SigLIP-L L12, SigLIP-L L20, SigLIP-L L24.
- all 18, S20 point values regardless of eligibility (unstable: flagged): readouts DINOv2-B L2, DINOv2-B L4, DINOv2-B L6, DINOv2-B L8, DINOv2-B L10, DINOv2-B L12, CLIP-L L4, CLIP-L L8, CLIP-L L12, CLIP-L L16, CLIP-L L20, CLIP-L L24, SigLIP-L L4, SigLIP-L L8, SigLIP-L L12, SigLIP-L L16, SigLIP-L L20, SigLIP-L L24.
- all 18, ratio-free L20 = sign(E) x [r(illusory) - r(mis20)]: readouts DINOv2-B L2, DINOv2-B L4, DINOv2-B L6, DINOv2-B L8, DINOv2-B L10, DINOv2-B L12, CLIP-L L4, CLIP-L L8, CLIP-L L12, CLIP-L L16, CLIP-L L20, CLIP-L L24, SigLIP-L L4, SigLIP-L L8, SigLIP-L L12, SigLIP-L L16, SigLIP-L L20, SigLIP-L L24.

Item 6 note: S20 point values of the readouts without a reliable effect are extreme (DINOv2-B L2 -180.3, CLIP-L L4 -16.8), because their aligned effects are near 0; the row is unstable by construction (rule item 6).

Item 8, competing descriptors over the 13 eligible readouts: relative depth (L / number of blocks) rho = -0.18, R2 = 0.022; training-objective family (DINOv2 self-supervised vs CLIP and SigLIP image-text) rho = +0.17, R2 = 0.031. Collinearity across all 18 readouts: rho(ERF50, relative depth) = +0.76.

Item 3 reading. The per-size rows above use eligibility determined at each size (n = 12 at 336 px, 12 at 448 px). With the size-averaged eligibility set instead (n = 13 at both sizes; not part of the rule, shown only because item 3 can be read either way): 336 px rho = -0.55 [-0.72, -0.07], p = .054, T = -0.31 (p .343); 448 px rho = -0.40 [-0.57, -0.08], p = .175, T = -0.58 (p .078).

## 4. Figure

`nn/figs/erf_vs_rotation.pdf` and `.png`: S20 against readout ERF50 for the 13 eligible readouts, colour and marker shape by backbone, each point labelled with its block, vertical bars = 95% stimulus-resampling CI of S20; dashed line S20 = 0 (no loss), dotted line S20 = 1 (effect abolished). The five readouts without a reliable aligned effect are listed under the plot, not drawn, because S20 is undefined for them.

## 5. What the numbers say (descriptive, within the rule's limits)

- Under the rule the result is INCONCLUSIVE (I-d): rotation sensitivity did not rise with the readout ERF50. The point estimates go the other way (rho = -0.37 across readouts, negative within all three networks), but the permutation test gives p = .218, so C1 is not met; network identity explained less variance than ERF (adjusted R2 -0.10 vs 0.18), so C2 is not met either.
- The primary predictor spans a narrow range (238.5 to 325.5 px, a factor of 1.36) and is not monotonic in depth within any network. A plausible reason, not tested here, is the readout design shared by all 18 readouts (ASPP with a dilation-12 branch and a global-pooling branch). The block-token ERF50 spans far more (30.5 to 314.0 px) and tracks depth (rho +0.90); against it S20 gives rho = -0.16 (p = .604), within-network rho clip -1.00, dinov2 +0.60, siglip -0.10. This is a secondary analysis and does not change the decision.
- Secondary analyses with permutation p < .05: positive association none; negative association none. The square-only drop ratio (n = 10) has rho = +0.50 (p = .143) with within-network rho dinov2 -1.00, siglip +1.00 and pooled T = +0.00, so its positive overall rho is not reproduced within networks.
- These are single-seed readouts; nothing here bears on human perception, and the illusion index is not read as human-likeness.

## 6. Deviations from the rule

None. No readout, block, size, display or location set was dropped or added; no measure, threshold, seed or permutation count was changed; `nn/DEVIATIONS.md` was not needed and does not exist.

Readings of wording that the rule leaves open (the first three were fixed in `nn/analyze_nn.py` at 08:01 UTC, before the new readouts were scored from 08:08 UTC, and were not changed after the results were seen):

- Section 7 item 3 (per size) is run with eligibility determined at that size (the 336-only and 448-only analyses each apply the Section 2 eligibility rule), as implemented in nn/analyze_nn.py before the results were seen. The alternative reading (size-averaged eligibility set, n = 13) is given under per_size_alternative_reading for transparency; neither is decision-bearing.
- Section 7 item 6 is reported as S20 = 1 - U(20) point values (the rho with U(20) itself has the opposite sign); summary.json labels it all18_point_U20.
- Section 3 'nulls in the bracket' = the corner, ink and edge nulls at sigma 8, 16 and 25 px; U(20), A(20) and slope comparisons use the corner nulls only, as the rule states.

Completions made here (rule items that `nn/results/summary.json` did not yet contain; computed exactly as the rule states):

- Section 7 item 8: summary.json lacked the Spearman rho for the training-objective family; it is computed here (secondary.competing.rho_objective_family).
- Section 4 'also reported' ERF details (per size in display and model-input px, per-display spread, inside-only and inducer-only ERFs) are tabulated in Section 2c from `nn/results/erf/<bb>.json`.

Report-level corrections relative to `nn/results/report.md`:

- Cosmetic only, no number changed: unrounded adjusted R2, partial R2 and vertex values are rounded here; the row labelled `all18_point_U20` there holds S20 point values and is labelled accordingly here.

## 7. Provenance and run facts

- Interpreter `$LAN_SAL_SRC/.venv/bin/python` (Python 3.12, torch 2.12.0). All jobs on the CPU, one heavy job at a time, under `caffeinate -i`; offline Hugging Face cache only (HF_HUB_OFFLINE=1); no downloads.
- Scoring of the 9 new readouts: 740 stim_v3 + 360 stim_alt images, 336 and 448 px, pipeline of `v2/scores/score_v3.py` + `v3/scores/score_alt.py` (`nn/score_nn.py`, 3 threads). ERF: 24 displays x 2 sizes x 8 or 9 locations x 18 readouts, readout and block-token targets (`nn/erf_nn.py`).
- Sanity rescore (rule Section 8): DINOv2-B L12 rescored twice (before and after the main run): all 1,100 images x 2 sizes bit-identical to the stored files in all three score files (max |difference| 0.0), and all 357 summary numbers (E, CIs, U, A, D, slope, per size) identical.

## 8. Checks run by `nn/property_results.py`

40 checks, 0 failed. Every decision-bearing number was re-derived with code written separately from `nn/analyze_nn.py` (same procedure, bootstrap indices and seeds) and matched; the nine stored readouts reproduce `v3/scores/results.json`.

- PASS: rule SHA-256 matches nn/ANALYSIS_RULE.sha256 (8e59fe87433a101aba97c99bca8e37a9e36faf08b47c9a70b1c6a141de3b403a (ANALYSIS_RULE.md recorded_utc 2026-11-17T07:53:35Z))
- PASS: summary.json: no missing inputs, real ERFs, rule unchanged at analysis time
- PASS: DINOv2-B L2: E, CI, eligibility, CC, S20 point, ERF50 re-derived = summary.json (E -0.0000 [-0.0167, +0.0164] eligible False)
- PASS: DINOv2-B L2: summary.json equals stored v3/scores/results.json (E, U, A, D at 10/20/30, slope)
- PASS: DINOv2-B L4: E, CI, eligibility, CC, S20 point, ERF50 re-derived = summary.json (E +0.0507 [+0.0426, +0.0591] eligible True)
- PASS: DINOv2-B L6: E, CI, eligibility, CC, S20 point, ERF50 re-derived = summary.json (E +0.0824 [+0.0624, +0.1023] eligible True)
- PASS: DINOv2-B L8: E, CI, eligibility, CC, S20 point, ERF50 re-derived = summary.json (E +0.0347 [+0.0234, +0.0465] eligible True)
- PASS: DINOv2-B L8: summary.json equals stored v3/scores/results.json (E, U, A, D at 10/20/30, slope)
- PASS: DINOv2-B L10: E, CI, eligibility, CC, S20 point, ERF50 re-derived = summary.json (E -0.0285 [-0.0455, -0.0134] eligible True)
- PASS: DINOv2-B L12: E, CI, eligibility, CC, S20 point, ERF50 re-derived = summary.json (E -0.0975 [-0.1260, -0.0703] eligible True)
- PASS: DINOv2-B L12: summary.json equals stored v3/scores/results.json (E, U, A, D at 10/20/30, slope)
- PASS: CLIP-L L4: E, CI, eligibility, CC, S20 point, ERF50 re-derived = summary.json (E -0.0003 [-0.0173, +0.0167] eligible False)
- PASS: CLIP-L L4: summary.json equals stored v3/scores/results.json (E, U, A, D at 10/20/30, slope)
- PASS: CLIP-L L8: E, CI, eligibility, CC, S20 point, ERF50 re-derived = summary.json (E +0.0409 [+0.0212, +0.0611] eligible True)
- PASS: CLIP-L L12: E, CI, eligibility, CC, S20 point, ERF50 re-derived = summary.json (E +0.0605 [+0.0480, +0.0732] eligible True)
- PASS: CLIP-L L12: summary.json equals stored v3/scores/results.json (E, U, A, D at 10/20/30, slope)
- PASS: CLIP-L L16: E, CI, eligibility, CC, S20 point, ERF50 re-derived = summary.json (E +0.0176 [+0.0052, +0.0299] eligible True)
- PASS: CLIP-L L20: E, CI, eligibility, CC, S20 point, ERF50 re-derived = summary.json (E -0.0101 [-0.0243, +0.0026] eligible False)
- PASS: CLIP-L L24: E, CI, eligibility, CC, S20 point, ERF50 re-derived = summary.json (E +0.0137 [-0.0120, +0.0423] eligible False)
- PASS: CLIP-L L24: summary.json equals stored v3/scores/results.json (E, U, A, D at 10/20/30, slope)
- PASS: SigLIP-L L4: E, CI, eligibility, CC, S20 point, ERF50 re-derived = summary.json (E +0.0198 [+0.0093, +0.0289] eligible True)
- PASS: SigLIP-L L4: summary.json equals stored v3/scores/results.json (E, U, A, D at 10/20/30, slope)
- PASS: SigLIP-L L8: E, CI, eligibility, CC, S20 point, ERF50 re-derived = summary.json (E +0.0551 [+0.0417, +0.0685] eligible True)
- PASS: SigLIP-L L8: summary.json equals stored v3/scores/results.json (E, U, A, D at 10/20/30, slope)
- PASS: SigLIP-L L12: E, CI, eligibility, CC, S20 point, ERF50 re-derived = summary.json (E -0.0206 [-0.0331, -0.0083] eligible True)
- PASS: SigLIP-L L16: E, CI, eligibility, CC, S20 point, ERF50 re-derived = summary.json (E +0.0019 [-0.0122, +0.0156] eligible False)
- PASS: SigLIP-L L20: E, CI, eligibility, CC, S20 point, ERF50 re-derived = summary.json (E +0.0473 [+0.0200, +0.0739] eligible True)
- PASS: SigLIP-L L24: E, CI, eligibility, CC, S20 point, ERF50 re-derived = summary.json (E +0.1447 [+0.1221, +0.1702] eligible True)
- PASS: SigLIP-L L24: summary.json equals stored v3/scores/results.json (E, U, A, D at 10/20/30, slope)
- PASS: primary test re-derived independently = summary.json (rho, CI, perm p, T, p_T, R2s, decision, inverted-U checks) (rho -0.3659 CI [-0.6245, -0.1403] p 0.2179; T -0.2692 p_T 0.4478; INCONCLUSIVE I-d; boot NaN 0)
- PASS: secondary S30 re-derived = summary.json
- PASS: secondary alt_S20 re-derived = summary.json
- PASS: secondary S20 vs ERF90 re-derived = summary.json
- PASS: secondary S20 vs bbERF50 re-derived = summary.json
- PASS: secondary S20 vs bbERF90 re-derived = summary.json
- PASS: secondary S20 vs ERF50_inside re-derived = summary.json
- PASS: secondary S20 vs ERF50_ind re-derived = summary.json
- PASS: item 8 relative-depth rho re-derived = summary.json
- PASS: secondary size336: eligibility set per size = readouts whose 336-px aligned CI excludes 0 (n = 12)
- PASS: secondary size448: eligibility set per size = readouts whose 448-px aligned CI excludes 0 (n = 12)

## 9. Files

- `$LAN_PROJECT_ROOT/nn/results/property_results.json`
- `$LAN_PROJECT_ROOT/nn/PROPERTY_RESULTS.md`
- `$LAN_PROJECT_ROOT/nn/figs/erf_vs_rotation.pdf`
- `$LAN_PROJECT_ROOT/nn/figs/erf_vs_rotation.png`
- `$LAN_PROJECT_ROOT/nn/property_results.py`
- `$LAN_PROJECT_ROOT/nn/results/summary.json`
- `$LAN_PROJECT_ROOT/nn/results/report.md`
- `$LAN_PROJECT_ROOT/nn/results/erf/{dinov2,clip,siglip}.json`
- `$LAN_PROJECT_ROOT/nn/results/raw_v3maps, raw_alt, raw_v2style (9 new readouts x 2 sizes)`
- `$LAN_PROJECT_ROOT/nn/ANALYSIS_RULE.md, nn/ANALYSIS_RULE.sha256, nn/RUN_LOG.md`
