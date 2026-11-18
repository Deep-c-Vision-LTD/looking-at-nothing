# Registers control: DINOv2-B R4 readouts, blocks 12 and 6

Generated 2026-11-17 15:40:24 by nn/registers/report_reg.py from nn/registers/full (registers) and v3/gpu/full (published plain run).

## Design

- Backbone: facebook/dinov2-with-registers-base (Darcet et al. 2024), 86.58 M parameters, 4 register tokens, local Hugging Face cache, frozen. Features: hidden_states[12] and hidden_states[6] from one forward pass, before the final layer norm, as for plain DINOv2-B. The token sequence is [CLS, 4 registers, 24 x 24 patches] at 336 px; CLS and the 4 registers are dropped and the patch tokens are reshaped to the grid, as the plain pipeline drops CLS.
- Readout and recipe: R4 (full ASPP with global pooling, the sweep readout), the published readout-swap training code copied unchanged (v3/gpu/train_full.py): the same 12,803 images (SALICON 10,000, CAT2000 2,000, MIT1003 803), 336 px, 6 epochs, batch 32, AdamW (learning rate 5e-4, weight decay 1e-4, cosine), loss KL + 2(1 - CC), no augmentation, final epoch, seeds 0, 1, 2 (the same initial weights, batch order and dropout stream structure as the plain R4 heads of each seed). Features computed on the fly in float32 with the backbone in the loop, on MPS, as the published run. A feature cache was not used: 12,803 images x 576 tokens x 768 channels x 2 blocks is 22.7 GB in float16 and 45.3 GB in float32, against 43 GB free at the start, and a float16 cache would also depart from the published float32 recipe.
- Scoring: the published evaluation code copied unchanged (v3/gpu/eval_full.py): held-out MIT1003 (last 200 images, 336 px, centre bias on), v2/stim_v3 (540 Kanizsa, 200 Ehrenstein displays) and v3/stim_alt (360 alternating-rotation displays) at 336 and 448 px, centre bias off. Statistics: the published analysis copied unchanged (v3/gpu/analyze_full.py): cluster bootstrap 10,000 over the 20 shape x support ratio x polarity cells with positions resampled within cells (60 configurations), RandomState(2026), 95% percentile CI; seeds averaged per configuration for the pooled rows; size avg = mean of the 336 and 448 px effects.
- Measures: Kanizsa effect = r(illusory) - r(control180) on the inside-figure AOI (r = mass inside / (mass inside + mass on the inducers)); core = the same contrast on figure pixels more than 30 px from the inducers (r_core30); rotation ratios on r_inside: ratio_uni(t) = effect(uniform rotation t) / effect(illusory), ratio_alt(t) = effect(alternating rotation t) / effect(illusory); alternating drop = drop_ratio(t) = (m(illusory) - m(alt t)) / (m(illusory) - m(mis t)), the share of the uniform-rotation loss that alternating rotation costs. All effects against control180.
- Pipeline check: the copied evaluation, run on the published plain checkpoint dinov2_L12_R4_s0, reproduced the published score files bit for bit (MIT1003 CC per image, 16,280 stim_v3 and 7,920 stim_alt values; max |diff| 0.0).

## Verdict

- Block 12: plain DINOv2-B R4 Kanizsa effect -0.114 [-0.146, -0.083] (negative); with registers +0.047 [+0.027, +0.069] (positive); per seed with registers: positive, positive, positive; core with registers +0.041 [+0.024, +0.059] (plain -0.100 [-0.127, -0.075]). The block-12 sign does not survive registers.
  Rotation profile with registers: ratio_uni20 0.96 [0.79, 1.16] (plain 0.72 [0.62, 0.81]), drop_ratio20 1.18 [-9.57, 9.42] (den. CI includes 0) (plain 0.51 [0.24, 0.80]); reference: the corner nulls (positive effects; ratio_uni20) and all local nulls (drop_ratio20), ratio_uni20 1.00 to 1.09, drop_ratio20 0.83 to 0.95. Upper CI of ratio_uni20 not below the null minimum; upper CI of drop_ratio20 not below the null minimum.
- Block 6: plain DINOv2-B R4 Kanizsa effect +0.090 [+0.069, +0.110] (positive); with registers +0.111 [+0.094, +0.128] (positive); per seed with registers: positive, positive, positive; core with registers +0.102 [+0.088, +0.115] (plain +0.082 [+0.071, +0.093]). The block-6 sign survives registers.
  Rotation profile with registers: ratio_uni20 0.55 [0.41, 0.67] (plain 0.59 [0.43, 0.73]), drop_ratio20 0.50 [0.36, 0.64] (plain 0.12 [-0.08, 0.26]); reference: the corner nulls (positive effects; ratio_uni20) and all local nulls (drop_ratio20), ratio_uni20 1.00 to 1.09, drop_ratio20 0.83 to 0.95. Upper CI of ratio_uni20 below the null minimum; upper CI of drop_ratio20 below the null minimum.

### Reading

- The block-12 sign does not survive registers. With registers the block-12 R4 readout puts more mass in the illusory figure than in the 180 degree control in every seed (+0.065, +0.037, +0.040; pooled +0.047 [+0.027, +0.069]), in the core (+0.041 [+0.024, +0.059]), against the 90 degree control (+0.067 [+0.048, +0.086]) and for the real filled figure (+0.091 [+0.050, +0.134]) and outline (+0.053 [+0.017, +0.091]), where the plain readout was negative on all five (-0.114 [-0.146, -0.083], -0.100 [-0.127, -0.075], -0.096 [-0.122, -0.071], -0.077 [-0.110, -0.045], -0.094 [-0.123, -0.064]). The flip holds at 336 and at 448 px separately.
- The block-6 sign survives registers: +0.111 [+0.094, +0.128] against +0.090 [+0.069, +0.110], positive in every seed, with a similar uniform-rotation ratio (0.55 [0.41, 0.67] against 0.59 [0.43, 0.73]), below the corner nulls (1.00 to 1.09). Its alternating drop is larger with registers (0.50 [0.36, 0.64] against 0.12 [-0.08, 0.26]), and both lie below the local nulls (0.83 to 0.95).
- The positive block-12 effect with registers is small (less than half of block 6) and shows no alignment signature: a uniform 20 degree rotation leaves 0.96 [0.79, 1.16] of it, an interval that includes 1 and overlaps the corner nulls, and the alternating drop is not estimable (the uniform loss is not different from zero; 1.18 [-9.57, 9.42] (den. CI includes 0)). Under the section B rule it is labelled 'inward-notch cue'. The Ehrenstein effect with registers is -0.022 [-0.039, -0.006] (plain +0.002 [-0.013, +0.014]).
- Held-out MIT1003 CC does not separate the two backbones: block 12 0.728 with registers against 0.726 plain, block 6 0.738 against 0.742; all six registers readouts pass the R4 validity threshold.
- The flip is not traced to high-norm tokens inside the displays. Plain DINOv2-B block 12 has no patch token above twice the image median on the Kanizsa displays at 336 or 448 px (largest max/median ratio 2.00); such tokens appear on natural images, mostly at 448 px (in 75% of the MIT1003 images at 448 px and 8% at 336 px, at most 0.07% of tokens), and the plain negative effect is present at both input sizes (-0.123 [-0.155, -0.092] at 336 px, -0.105 [-0.138, -0.073] at 448 px). facebook/dinov2-with-registers-base is a separately trained network, not the plain weights with registers added, so this control changes the checkpoint as well as the token mechanism. What it shows is that the negative block-12 sign is specific to the plain DINOv2-B checkpoint: a DINOv2-B trained with registers gives a weak, rotation-insensitive positive effect at block 12, and at block 6 a positive, rotation-sensitive effect like that of the plain checkpoint.

## Held-out MIT1003 CC and validity

Validity rule of the readout swap: CC >= CC(R4 on the same features) - 0.03. Applied within the registers features the rule is met by R4 by construction (R4 is its own reference), so the substantive check is against the published plain DINOv2-B R4 on the same block (threshold = plain R4 mean CC - 0.03).

| block | seed | plain R4 CC | registers R4 CC | difference | threshold (plain R4 - 0.03) | registers valid | final training loss plain / registers |
|---|---|---|---|---|---|---|---|
| 12 | 0 | 0.7256 | 0.7275 | +0.0019 | 0.6955 | yes | 0.2790 / 0.2908 |
| 12 | 1 | 0.7272 | 0.7266 | -0.0006 | 0.6955 | yes | 0.2817 / 0.2941 |
| 12 | 2 | 0.7238 | 0.7306 | +0.0068 | 0.6955 | yes | 0.2819 / 0.2931 |
| 12 | mean | 0.7255 | 0.7282 | +0.0027 | 0.6955 | yes | |
| 6 | 0 | 0.7390 | 0.7373 | -0.0017 | 0.7122 | yes | 0.3099 / 0.3009 |
| 6 | 1 | 0.7406 | 0.7357 | -0.0049 | 0.7122 | yes | 0.3166 / 0.3052 |
| 6 | 2 | 0.7469 | 0.7402 | -0.0067 | 0.7122 | yes | 0.3157 / 0.3077 |
| 6 | mean | 0.7422 | 0.7377 | -0.0045 | 0.7122 | yes | |

## Kanizsa effect vs control180 (inside AOI, size avg), per seed and pooled

Per-seed CIs: cluster bootstrap over the 60 configurations of that seed. Pooled: seeds averaged per configuration, then the same bootstrap.

| block | seed | plain DINOv2-B R4 | registers R4 | plain core (r_core30) | registers core (r_core30) |
|---|---|---|---|---|---|
| 12 | 0 | -0.126 [-0.165, -0.088] | +0.065 [+0.046, +0.086] | -0.109 [-0.142, -0.078] | +0.059 [+0.042, +0.077] |
| 12 | 1 | -0.106 [-0.137, -0.078] | +0.037 [+0.008, +0.067] | -0.094 [-0.121, -0.070] | +0.027 [+0.003, +0.052] |
| 12 | 2 | -0.111 [-0.139, -0.082] | +0.040 [+0.023, +0.057] | -0.098 [-0.120, -0.075] | +0.038 [+0.021, +0.056] |
| 12 | pooled | -0.114 [-0.146, -0.083] p < .001 | +0.047 [+0.027, +0.069] p < .001 | -0.100 [-0.127, -0.075] | +0.041 [+0.024, +0.059] |
| 6 | 0 | +0.115 [+0.089, +0.139] | +0.113 [+0.095, +0.130] | +0.105 [+0.092, +0.117] | +0.101 [+0.088, +0.113] |
| 6 | 1 | +0.088 [+0.066, +0.109] | +0.102 [+0.085, +0.120] | +0.081 [+0.069, +0.093] | +0.095 [+0.080, +0.109] |
| 6 | 2 | +0.067 [+0.051, +0.082] | +0.118 [+0.100, +0.136] | +0.061 [+0.050, +0.073] | +0.109 [+0.094, +0.124] |
| 6 | pooled | +0.090 [+0.069, +0.110] p < .001 | +0.111 [+0.094, +0.128] p < .001 | +0.082 [+0.071, +0.093] | +0.102 [+0.088, +0.115] |

### Other contrasts (pooled over seeds; size avg unless a size is stated; inside AOI unless core)

| block | contrast | plain DINOv2-B R4 | registers R4 |
|---|---|---|---|
| 12 | Kanizsa illusory - control180, 336 px only | -0.123 [-0.155, -0.092] | +0.046 [+0.024, +0.071] |
| 12 | Kanizsa illusory - control180, 448 px only | -0.105 [-0.138, -0.073] | +0.048 [+0.028, +0.069] |
| 12 | Kanizsa core (r_core30), 336 px only | -0.116 [-0.147, -0.088] | +0.044 [+0.027, +0.063] |
| 12 | Kanizsa core (r_core30), 448 px only | -0.084 [-0.108, -0.061] | +0.039 [+0.018, +0.059] |
| 12 | Kanizsa illusory - control90 | -0.096 [-0.122, -0.071] | +0.067 [+0.048, +0.086] |
| 12 | real filled - control180 | -0.077 [-0.110, -0.045] | +0.091 [+0.050, +0.134] |
| 12 | real outline - control180 | -0.094 [-0.123, -0.064] | +0.053 [+0.017, +0.091] |
| 12 | Ehrenstein illusory - control_end | +0.002 [-0.013, +0.014] | -0.022 [-0.039, -0.006] |
| 6 | Kanizsa illusory - control180, 336 px only | +0.078 [+0.058, +0.097] | +0.108 [+0.089, +0.128] |
| 6 | Kanizsa illusory - control180, 448 px only | +0.101 [+0.080, +0.123] | +0.115 [+0.097, +0.131] |
| 6 | Kanizsa core (r_core30), 336 px only | +0.073 [+0.059, +0.086] | +0.105 [+0.091, +0.119] |
| 6 | Kanizsa core (r_core30), 448 px only | +0.092 [+0.080, +0.103] | +0.098 [+0.082, +0.114] |
| 6 | Kanizsa illusory - control90 | +0.121 [+0.100, +0.141] | +0.123 [+0.099, +0.148] |
| 6 | real filled - control180 | +0.130 [+0.120, +0.140] | +0.170 [+0.156, +0.184] |
| 6 | real outline - control180 | +0.173 [+0.163, +0.183] | +0.169 [+0.153, +0.185] |
| 6 | Ehrenstein illusory - control_end | +0.109 [+0.092, +0.127] | +0.105 [+0.090, +0.122] |

## Rotation ratios and alternating drop (stim_alt, r_inside, theta 20, size avg)

Local-null reference (v3/scores/results.json, as quoted in the paper): the 10 ink and edge-energy nulls, whose aligned effects are also negative, give ratio_uni20 0.77 to 0.94 and drop_ratio20 0.83 to 0.86 (10 nulls with a measurable uniform drop). The 5 corner nulls, whose aligned effects are positive, give ratio_uni20 1.00 to 1.09 (5 with ratios computed), and all local nulls with a measurable uniform drop give drop_ratio20 0.83 to 0.95 (13 nulls; all shapes pooled). A ratio_uni below the null range means the effect depends on inducer alignment more than the local nulls; a drop ratio below the null range means alternating rotation costs less of the uniform loss than local image statistics predict.

| block | seed | ratio_uni20 plain | ratio_uni20 registers | ratio_alt20 plain | ratio_alt20 registers | drop_ratio20 plain | drop_ratio20 registers |
|---|---|---|---|---|---|---|---|
| 12 | 0 | 0.72 [0.61, 0.82] | 0.77 [0.61, 0.91] | 0.89 [0.82, 0.97] | 0.83 [0.63, 0.98] | 0.38 [0.09, 0.71] | 0.75 [0.12, 1.41] |
| 12 | 1 | 0.71 [0.63, 0.79] | 1.04 [0.76, 1.64] | 0.85 [0.78, 0.92] | 0.87 [0.50, 1.14] | 0.52 [0.26, 0.75] | -3.29 [-17.86, 18.65] (den. CI includes 0) |
| 12 | 2 | 0.72 [0.61, 0.82] | 1.18 [0.98, 1.55] | 0.82 [0.76, 0.89] | 1.21 [1.03, 1.59] | 0.63 [0.38, 0.99] | 1.21 [-3.35, 6.51] (den. CI includes 0) |
| 12 | pooled | 0.72 [0.62, 0.81] | 0.96 [0.79, 1.16] | 0.86 [0.79, 0.93] | 0.95 [0.75, 1.13] | 0.51 [0.24, 0.80] | 1.18 [-9.57, 9.42] (den. CI includes 0) |
| 6 | 0 | 0.61 [0.48, 0.73] | 0.58 [0.44, 0.70] | 0.93 [0.86, 0.99] | 0.78 [0.65, 0.88] | 0.18 [0.02, 0.31] | 0.53 [0.36, 0.68] |
| 6 | 1 | 0.47 [0.25, 0.65] | 0.36 [0.17, 0.53] | 0.89 [0.76, 0.99] | 0.69 [0.56, 0.81] | 0.22 [0.02, 0.34] | 0.48 [0.36, 0.59] |
| 6 | 2 | 0.71 [0.53, 0.85] | 0.69 [0.59, 0.78] | 1.06 [0.97, 1.18] | 0.85 [0.76, 0.92] | -0.22 [-1.09, 0.07] | 0.50 [0.33, 0.68] |
| 6 | pooled | 0.59 [0.43, 0.73] | 0.55 [0.41, 0.67] | 0.95 [0.86, 1.02] | 0.78 [0.66, 0.87] | 0.12 [-0.08, 0.26] | 0.50 [0.36, 0.64] |

For comparison, the other published R4 readouts (pooled over seeds, r_inside, theta 20): siglip_L24: Kanizsa effect +0.143 [+0.119, +0.171], ratio_uni20 0.81 [0.67, 0.95], ratio_alt20 0.63 [0.48, 0.80], drop_ratio20 1.97 [0.95, 6.12]; iie_shapenetc: Kanizsa effect +0.082 [+0.050, +0.117], ratio_uni20 0.10 [-0.11, 0.27], ratio_alt20 0.49 [0.32, 0.66], drop_ratio20 0.56 [0.42, 0.67].

### All angles (pooled over seeds)

| block | backbone | ratio_uni10 | ratio_uni30 | ratio_alt10 | ratio_alt30 | drop_ratio30 | alt20 - mis20 (r_inside) |
|---|---|---|---|---|---|---|---|
| 12 | plain | 0.92 [0.83, 1.01] | 0.52 [0.40, 0.61] | 0.98 [0.92, 1.05] | 0.73 [0.65, 0.81] | 0.57 [0.37, 0.77] | -0.016 [-0.026, -0.005] |
| 12 | registers | 1.15 [1.02, 1.41] | 0.60 [0.18, 0.84] | 1.12 [0.99, 1.38] | 0.79 [0.45, 1.00] | 0.53 [-0.02, 0.87] | -0.000 [-0.007, +0.007] |
| 6 | plain | 0.82 [0.71, 0.92] | 0.27 [0.08, 0.42] | 1.04 [0.99, 1.10] | 0.71 [0.57, 0.82] | 0.39 [0.30, 0.48] | +0.032 [+0.022, +0.043] |
| 6 | registers | 0.82 [0.74, 0.90] | 0.28 [0.12, 0.42] | 0.93 [0.88, 0.98] | 0.57 [0.40, 0.71] | 0.60 [0.49, 0.70] | +0.025 [+0.016, +0.034] |

Section B rule (uniform ratio(20) labelled only when the aligned CI is above 0): block 12 plain: not eligible; block 12 registers: eligible, 0.96 [0.79, 1.16] inward-notch cue; block 6 plain: eligible, 0.59 [0.43, 0.73] intermediate; block 6 registers: eligible, 0.55 [0.41, 0.67] intermediate.

## Token-norm diagnostic (CPU, the readouts' input sizes)

Patch-token L2 norms (CLS and registers excluded) on 24 Kanizsa displays (illusory and control180, one per shape x support ratio x polarity) and 24 held-out MIT1003 images, at 336 and 448 px. Per image, the ratio of the largest patch norm to the median patch norm; outlier = norm above twice the image's median (high-norm artefact tokens in DINOv2 are typically several times the median).

| backbone | block | px | image set | median of max/median | largest max/median | share of tokens > 2 x median | share of images with such a token |
|---|---|---|---|---|---|---|---|
| plain | 12 | 336 | kanizsa | 1.72 | 1.90 | 0.00000 | 0.000 |
| plain | 12 | 336 | mit1003 | 1.53 | 2.36 | 0.00014 | 0.083 |
| plain | 12 | 448 | kanizsa | 1.62 | 2.00 | 0.00000 | 0.000 |
| plain | 12 | 448 | mit1003 | 2.79 | 3.43 | 0.00073 | 0.750 |
| plain | 6 | 336 | kanizsa | 1.61 | 1.71 | 0.00000 | 0.000 |
| plain | 6 | 336 | mit1003 | 1.48 | 1.94 | 0.00000 | 0.000 |
| plain | 6 | 448 | kanizsa | 1.65 | 1.73 | 0.00000 | 0.000 |
| plain | 6 | 448 | mit1003 | 1.56 | 1.89 | 0.00000 | 0.000 |
| registers | 12 | 336 | kanizsa | 1.44 | 1.57 | 0.00000 | 0.000 |
| registers | 12 | 336 | mit1003 | 1.26 | 1.42 | 0.00000 | 0.000 |
| registers | 12 | 448 | kanizsa | 1.43 | 1.62 | 0.00000 | 0.000 |
| registers | 12 | 448 | mit1003 | 1.30 | 1.42 | 0.00000 | 0.000 |
| registers | 6 | 336 | kanizsa | 1.28 | 1.32 | 0.00000 | 0.000 |
| registers | 6 | 336 | mit1003 | 1.32 | 1.49 | 0.00000 | 0.000 |
| registers | 6 | 448 | kanizsa | 1.26 | 1.28 | 0.00000 | 0.000 |
| registers | 6 | 448 | mit1003 | 1.34 | 1.47 | 0.00000 | 0.000 |

## Run record and files

- 2026-11-17 13:50:49 done train_dinov2reg_R4_s0 2830s
- 2026-11-17 13:54:22 done eval_dinov2reg_s0 146s
- 2026-11-17 14:42:20 done train_dinov2reg_R4_s1 2811s
- 2026-11-17 14:46:06 done eval_dinov2reg_s1 158s
- 2026-11-17 15:33:51 done train_dinov2reg_R4_s2 2798s
- 2026-11-17 15:37:24 done eval_dinov2reg_s2 146s
- 2026-11-17 15:38:35 done eval_dinov2reg_final 3s
- 2026-11-17 15:39:04 finished; failed steps: none; trained heads: 6 of 6
- Interpreter: ~/Downloads/saliency_research/.venv/bin/python (torch 2.12, transformers 5.12.1), offline Hugging Face cache, device MPS, one GPU job at a time (gpu_idle.sh checks before every step), caffeinate -i. No latency timing was run.
- Code: nn/registers/regcommon.py (adds the backbone), train_reg.py, eval_reg.py, analyze_reg.py (copies of v3/gpu/train_full.py, eval_full.py, analyze_full.py with only the import changed), report_reg.py, norm_diag.py, check_eval.py, run_reg.sh. Nothing outside nn/registers was written.
- Outputs: nn/registers/full/ (ckpt/*.pt and training records, scores/, maps/stim_alt/, results_full.json, report_full.md), nn/registers/registers_results.json (all numbers in this file, per seed and pooled, and every local null), nn/registers/diag/norm_diag.json, nn/registers/check/eval_check.json.
