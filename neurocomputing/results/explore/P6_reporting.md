# P6 cells: rule 5.6 reporting items and rule Section 6 items 1, 3 and 5 (`report_p6_cells.py`)

**EXPLORATORY AND REPORTING ONLY.** Nothing here is decision-bearing. P6 and the title decision are those of `out/primary_P6.json` (reproduced, gate G2). Linear quantities: mean and 95% WCU interval (12 clusters); ratios: value, BCa interval and flag; continuity: the published two-stage percentile interval. All values unrounded in `P6_reporting.json`; 4 decimals here.

- Code SHA-256: `explore/report_p6_cells.py` 5db1fc3bc45f4e4820f28d16eb070ef319979760167c70384b7dbaabaaec7665; `explore/excommon.py` d168232289be46a9e883385aeee279165e1c7924879c07c5f03204b3b38a397a; `explore/explore_saved.py` 95236f1daacac26606f90294226f2394e1e73c9c6bacd4bb19d31148f01dfb7a.
- Rule SHA-256 (`REANALYSIS_RULE.md`): b03f397b6b1808e7055f0d6db8f5685c0e6a2850261eeb3d8525dddbeca73a38. Public commit (`PUBLIC_PUSH.txt`): f67c7df98c27136be8ff3db4eea57d1973579602.
- Run 2026-11-20T04:01:45Z to 2026-11-20T04:01:51Z (6.2 s compute). require_public: 33 hashed files and every stimulus verified.

## 0. Reproduction gates (all passed; any failure would have stopped the run before writing)

- G1 reused code unchanged (explore_saved.py and excommon.py equal the hashes in explore_saved.json): passed.
- G2 out/primary_P6.json (2026-11-20T01:20:26Z) reproduced in full (P3, P6, title): 578 leaves, 430 numeric, max |difference| 0.0, tolerance 1e-12: passed.
- G3 per unit through this script's loader, against primary_P6.json at 1e-12: CC mean, threshold, validity, status, E mean, WCU interval, WCR p, se, t and cluster-mean skewness (R4 rows: CC mean, sign, reference defined); per set Holm p, keeps-sign and opposite-sign flags and n_valid; the rule-fixed registers R4 means, thresholds and plain thresholds exact. primary_P6.json stores neither per-seed CCs nor seed point estimates: the seed point estimates enter G3 only through their mean equalling E, and both are checked one by one only at 4 decimals in G4 (mit_cc_seed, E_seed_mean). 290 checks, 0 failed.
- G4 published-method continuity values (gpu/g3, gpu/a14 and nn/registers results_full.json, 4 decimals): mit_cc_seed 60 of 60; mit_cc_mean 20 of 20; E_mean 20 of 20; E_percentile_ci 40 of 40; E_seed_mean 60 of 60; E_seed_percentile_ci 120 of 120; E_per_size 80 of 80; E_per_size_percentile_ci 80 of 80; E_vs_control90 20 of 20; E_vs_control90_percentile_ci 40 of 40; alt_effect_illusory 20 of 20; profile_U_A 120 of 120; profile_D 39 of 39; U20_A20_value 40 of 40; U20_A20_percentile_ci 80 of 80; item5_f_core30 40 of 40; item5_f_core30_percentile_ci 40 of 40; validity_flag_a14 8 of 8.
- G5 registers R4 items 1, 3 and 5 against explore_saved.json: dinov2reg_L12 R4 523 numeric leaves, max |difference| 0.0; dinov2reg_L6 R4 563 numeric leaves, max |difference| 0.0.
- Internal: rule 5.6 per-size E, U20 and A20 equal the item 1 values (100 records, max |difference| 0.0).

## 1. Rule 5.6 items per feature set

CC per seed and mean: held-out MIT1003 CC; validity by `analyze_re.accuracy_valid`. E = illusory minus control180 (size average); Holm p within the set over its valid non-R4 cells. U20 and A20: BCa; `unstable` = denominator |t| < 4; `not reportable` = denominator WCU interval includes 0 (rule 4.3). Each set has three tables: the rule 5.6 items; seed point estimates, WCR p, the P6 flags, the cluster-mean skewness of E (rule 4.1, diagnostic) and the plain-threshold comparison; and each ratio next to the difference it summarises with its Fieller interval (rules 4.2 and 4.3).

### dinov2reg_L6 (G3; DINOv2-B with registers, block 6; R4 = registers heads of 17 November 2026)

R4 reference sign +1, reference defined: yes; valid non-R4 cells 2; keeping the sign after Holm 2.

| cell | validity | CC per seed | mean CC | threshold | E [WCU] | Holm p | E 336 [WCU] | E 448 [WCU] | E vs control90 [WCU] | U20 [BCa] | A20 [BCa] | E [published percentile] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1n | below criterion (flag: reported, not counted) | 0.7004 / 0.7005 / 0.6992 | 0.7000 | 0.7077 | +0.0672 [+0.0531, +0.0817] | n/a (below criterion) | +0.0704 [+0.0587, +0.0821] | +0.0640 [+0.0455, +0.0832] | +0.0849 [+0.0635, +0.1065] | +0.5300 [+0.2214, +0.7140] | +0.9280 [+0.8575, +0.9901] | [+0.0548, +0.0820] |
| R1 | below criterion (flag: reported, not counted) | 0.6861 / 0.6923 / 0.6911 | 0.6899 | 0.7077 | +0.0384 [+0.0215, +0.0560] | n/a (below criterion) | +0.0467 [+0.0321, +0.0617] | +0.0300 [+0.0101, +0.0504] | +0.0543 [+0.0353, +0.0740] | +0.6033 [+0.1979, +0.7708] | +1.1287 [+1.0198, +1.3415] | [+0.0236, +0.0531] |
| R2 | valid | 0.7207 / 0.7213 / 0.7255 | 0.7225 | 0.7077 | +0.1065 [+0.0879, +0.1250] | 0.0004 | +0.1065 [+0.0837, +0.1285] | +0.1064 [+0.0885, +0.1236] | +0.1189 [+0.0934, +0.1443] | +0.6099 [+0.4869, +0.7106] | +0.8262 [+0.7240, +0.9075] | [+0.0900, +0.1228] |
| R3 | valid | 0.7194 / 0.7244 / 0.7259 | 0.7233 | 0.7077 | +0.0979 [+0.0673, +0.1282] | 0.0004 | +0.1183 [+0.0882, +0.1485] | +0.0774 [+0.0419, +0.1140] | +0.1020 [+0.0732, +0.1305] | +0.6064 [+0.4276, +0.7278] | +0.8938 [+0.7932, +0.9756] | [+0.0719, +0.1237] |
| R4 | reference (R4) | 0.7373 / 0.7357 / 0.7402 | 0.7377 | n/a (reference) | +0.1114 [+0.0921, +0.1304] | n/a (reference) | +0.1081 [+0.0868, +0.1297] | +0.1147 [+0.0956, +0.1333] | +0.1232 [+0.0941, +0.1521] | +0.5533 [+0.4114, +0.6711] | +0.7757 [+0.6665, +0.8723] | [+0.0939, +0.1281] |

| cell | seed point estimates 0 / 1 / 2 | WCR p | skewness of the cluster means of E | keeps R4's sign | interval of the opposite sign | CC against the plain DINOv2-B R4 threshold (never decision-bearing) |
|---|---|---|---|---|---|---|
| R1n | +0.0637 / +0.0537 / +0.0842 | 0.0002 | +1.423 | n/a (below criterion) | no (below criterion; not counted) | 0.7000 against 0.7122 (dinov2_L6): below |
| R1 | +0.0334 / +0.0371 / +0.0446 | 0.0006 | +0.121 | n/a (below criterion) | no (below criterion; not counted) | 0.6899 against 0.7122 (dinov2_L6): below |
| R2 | +0.1096 / +0.1032 / +0.1067 | 0.0002 | -0.143 | yes | no | 0.7225 against 0.7122 (dinov2_L6): at or above |
| R3 | +0.1109 / +0.0989 / +0.0838 | 0.0002 | +0.160 | yes | no | 0.7233 against 0.7122 (dinov2_L6): at or above |
| R4 | +0.1134 / +0.1025 / +0.1183 | 0.0002 | -0.390 | n/a | n/a | n/a |

| cell | L20 [WCU] | U20 [BCa] | U20 Fieller | L_alt20 [WCU] | A20 [BCa] | A20 Fieller | abs(t) of E |
|---|---|---|---|---|---|---|---|
| R1n | +0.0316 [+0.0174, +0.0454] | +0.5300 [+0.2214, +0.7140] | [+0.2014, +0.7655] | +0.0048 [+0.0003, +0.0094] | +0.9280 [+0.8575, +0.9901] | [+0.8613, +0.9940] | 9.67 |
| R1 | +0.0152 [+0.0104, +0.0199] | +0.6033 [+0.1979, +0.7708] | [+0.1172, +0.7990] | -0.0049 [-0.0090, -0.0006] | +1.1287 [+1.0198, +1.3415] | [+1.0165, +1.4005] | 5.01 |
| R2 | +0.0415 [+0.0310, +0.0518] | +0.6099 [+0.4869, +0.7106] | [+0.4603, +0.7263] | +0.0185 [+0.0102, +0.0268] | +0.8262 [+0.7240, +0.9075] | [+0.7009, +0.9167] | 12.68 |
| R3 | +0.0385 [+0.0297, +0.0470] | +0.6064 [+0.4276, +0.7278] | [+0.3853, +0.7369] | +0.0104 [+0.0016, +0.0189] | +0.8938 [+0.7932, +0.9756] | [+0.7743, +0.9829] | 7.36 |
| R4 | +0.0498 [+0.0352, +0.0641] | +0.5533 [+0.4114, +0.6711] | [+0.3846, +0.6931] | +0.0250 [+0.0141, +0.0355] | +0.7757 [+0.6665, +0.8723] | [+0.6360, +0.8818] | 12.87 |

### dinov2reg_L12 (G3; DINOv2-B with registers, block 12; R4 = registers heads of 17 November 2026)

R4 reference sign +1, reference defined: yes; valid non-R4 cells 4; keeping the sign after Holm 0.

| cell | validity | CC per seed | mean CC | threshold | E [WCU] | Holm p | E 336 [WCU] | E 448 [WCU] | E vs control90 [WCU] | U20 [BCa] | A20 [BCa] | E [published percentile] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1n | valid | 0.7092 / 0.7057 / 0.7084 | 0.7078 | 0.6982 | +0.0033 [-0.0458, +0.0537] | 1 | +0.0084 [-0.0400, +0.0590] | -0.0018 [-0.0509, +0.0492] | +0.0440 [+0.0001, +0.0891] | not reportable (denominator interval includes 0) | not reportable (denominator interval includes 0) | [-0.0399, +0.0426] |
| R1 | valid | 0.7034 / 0.7076 / 0.7055 | 0.7055 | 0.6982 | -0.0351 [-0.0688, -0.0007] | 0.1616 | -0.0309 [-0.0629, +0.0020] | -0.0393 [-0.0748, -0.0033] | -0.0136 [-0.0474, +0.0211] | +0.1455 [-1.3964, +0.6222] (unstable) | +0.4902 [-0.1591, +0.7687] (unstable) | [-0.0644, -0.0084] |
| R2 | valid | 0.7200 / 0.7206 / 0.7263 | 0.7223 | 0.6982 | +0.0255 [-0.0142, +0.0660] | 0.5622 | +0.0216 [-0.0172, +0.0607] | +0.0293 [-0.0117, +0.0715] | +0.0518 [+0.0162, +0.0869] | not reportable (denominator interval includes 0) | not reportable (denominator interval includes 0) | [-0.0079, +0.0590] |
| R3 | valid | 0.7262 / 0.7235 / 0.7295 | 0.7264 | 0.6982 | +0.0034 [-0.0393, +0.0470] | 1 | +0.0019 [-0.0326, +0.0377] | +0.0049 [-0.0460, +0.0570] | +0.0463 [+0.0029, +0.0921] | not reportable (denominator interval includes 0) | not reportable (denominator interval includes 0) | [-0.0330, +0.0396] |
| R4 | reference (R4) | 0.7275 / 0.7266 / 0.7306 | 0.7282 | n/a (reference) | +0.0473 [+0.0228, +0.0723] | n/a (reference) | +0.0463 [+0.0190, +0.0742] | +0.0483 [+0.0250, +0.0721] | +0.0667 [+0.0445, +0.0890] | +0.9553 [+0.7994, +1.1717] | +0.9471 [+0.7335, +1.1101] | [+0.0270, +0.0694] |

| cell | seed point estimates 0 / 1 / 2 | WCR p | skewness of the cluster means of E | keeps R4's sign | interval of the opposite sign | CC against the plain DINOv2-B R4 threshold (never decision-bearing) |
|---|---|---|---|---|---|---|
| R1n | -0.0224 / +0.0316 / +0.0006 | 0.8598 | -0.475 | no | no | 0.7078 against 0.6955 (dinov2_L12): at or above |
| R1 | -0.0329 / -0.0874 / +0.0151 | 0.0404 | -0.495 | no | yes | 0.7055 against 0.6955 (dinov2_L12): at or above |
| R2 | +0.0185 / +0.0190 / +0.0388 | 0.1874 | +0.198 | no | no | 0.7223 against 0.6955 (dinov2_L12): at or above |
| R3 | +0.0300 / -0.0163 / -0.0035 | 0.844 | -0.175 | no | no | 0.7264 against 0.6955 (dinov2_L12): at or above |
| R4 | +0.0652 / +0.0366 / +0.0400 | 0.0008 | +0.605 | n/a | n/a | n/a |

| cell | L20 [WCU] | U20 [BCa] | U20 Fieller | L_alt20 [WCU] | A20 [BCa] | A20 Fieller | abs(t) of E |
|---|---|---|---|---|---|---|---|
| R1n | -0.0109 [-0.0291, +0.0073] | not reportable (denominator interval includes 0) | nr | -0.0109 [-0.0236, +0.0021] | not reportable (denominator interval includes 0) | nr | 0.15 |
| R1 | -0.0300 [-0.0467, -0.0135] | +0.1455 [-1.3964, +0.6222] (unstable) | [-11.3420, +0.6362] | -0.0179 [-0.0291, -0.0066] | +0.4902 [-0.1591, +0.7687] (unstable) | [-4.5560, +0.7492] | 2.34 |
| R2 | -0.0075 [-0.0165, +0.0014] | not reportable (denominator interval includes 0) | nr | -0.0012 [-0.0049, +0.0027] | not reportable (denominator interval includes 0) | nr | 1.43 |
| R3 | -0.0045 [-0.0228, +0.0133] | not reportable (denominator interval includes 0) | nr | -0.0026 [-0.0150, +0.0103] | not reportable (denominator interval includes 0) | nr | 0.18 |
| R4 | +0.0021 [-0.0070, +0.0112] | +0.9553 [+0.7994, +1.1717] | [+0.7634, +1.2011] | +0.0025 [-0.0061, +0.0111] | +0.9471 [+0.7335, +1.1101] | [+0.7214, +1.1534] | 4.22 |

### dinov2ms_L7L11 (A14; public DINOv2-B blocks 7 and 11 concatenated; R4 trained in A14)

R4 reference sign -1, reference defined: yes; valid non-R4 cells 4; keeping the sign after Holm 2.

| cell | validity | CC per seed | mean CC | threshold | E [WCU] | Holm p | E 336 [WCU] | E 448 [WCU] | E vs control90 [WCU] | U20 [BCa] | A20 [BCa] | E [published percentile] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1n | valid | 0.7262 / 0.7284 / 0.7292 | 0.7279 | 0.7151 | +0.0121 [-0.0013, +0.0255] | 0.0674 | +0.0023 [-0.0126, +0.0175] | +0.0219 [+0.0084, +0.0354] | +0.0259 [+0.0082, +0.0437] | not reportable (denominator interval includes 0) | not reportable (denominator interval includes 0) | [-0.0003, +0.0243] |
| R1 | valid | 0.7230 / 0.7240 / 0.7243 | 0.7238 | 0.7151 | +0.0198 [+0.0063, +0.0343] | 0.0272 | +0.0140 [-0.0003, +0.0288] | +0.0256 [+0.0118, +0.0400] | +0.0452 [+0.0290, +0.0620] | +1.2123 [+0.7946, +2.1666] (unstable) | +1.0415 [+0.7451, +1.9901] (unstable) | [+0.0065, +0.0310] |
| R2 | valid | 0.7366 / 0.7361 / 0.7371 | 0.7366 | 0.7151 | -0.0188 [-0.0299, -0.0078] | 0.0138 | -0.0248 [-0.0365, -0.0131] | -0.0129 [-0.0254, -0.0005] | +0.0262 [+0.0164, +0.0360] | +0.6068 [+0.1508, +1.0546] (unstable) | +1.3226 [+1.0009, +1.8847] (unstable) | [-0.0285, -0.0091] |
| R3 | valid | 0.7376 / 0.7338 / 0.7359 | 0.7358 | 0.7151 | -0.0450 [-0.0641, -0.0254] | 0.0032 | -0.0513 [-0.0767, -0.0256] | -0.0386 [-0.0546, -0.0221] | -0.0027 [-0.0134, +0.0079] | +0.9251 [+0.7159, +1.1257] | +1.0688 [+0.8723, +1.2694] | [-0.0614, -0.0281] |
| R4 | reference (R4) | 0.7438 / 0.7467 / 0.7448 | 0.7451 | n/a (reference) | -0.0225 [-0.0347, -0.0103] | n/a (reference) | -0.0198 [-0.0331, -0.0064] | -0.0251 [-0.0375, -0.0127] | +0.0246 [+0.0137, +0.0358] | +0.6042 [+0.1255, +0.8705] | +0.9472 [+0.7687, +1.1464] | [-0.0334, -0.0124] |

| cell | seed point estimates 0 / 1 / 2 | WCR p | skewness of the cluster means of E | keeps R4's sign | interval of the opposite sign | CC against the plain DINOv2-B R4 threshold (never decision-bearing) |
|---|---|---|---|---|---|---|
| R1n | +0.0016 / +0.0168 / +0.0180 | 0.0674 | -0.139 | no | no | n/a |
| R1 | +0.0459 / +0.0338 / -0.0204 | 0.0136 | -1.064 | no | yes | n/a |
| R2 | -0.0114 / -0.0214 / -0.0236 | 0.0046 | -0.002 | yes | no | n/a |
| R3 | -0.0072 / -0.0593 / -0.0684 | 0.0008 | +0.316 | yes | no | n/a |
| R4 | -0.0185 / -0.0284 / -0.0205 | 0.0018 | -0.425 | n/a | n/a | n/a |

| cell | L20 [WCU] | U20 [BCa] | U20 Fieller | L_alt20 [WCU] | A20 [BCa] | A20 Fieller | abs(t) of E |
|---|---|---|---|---|---|---|---|
| R1n | +0.0096 [-0.0068, +0.0255] | not reportable (denominator interval includes 0) | nr | +0.0064 [-0.0045, +0.0175] | not reportable (denominator interval includes 0) | nr | 1.98 |
| R1 | -0.0042 [-0.0137, +0.0053] | +1.2123 [+0.7946, +2.1666] (unstable) | [+0.7838, +2.6543] | -0.0008 [-0.0085, +0.0071] | +1.0415 [+0.7451, +1.9901] (unstable) | [+0.7095, +2.0985] | 3.09 |
| R2 | -0.0074 [-0.0171, +0.0022] | +0.6068 [+0.1508, +1.0546] (unstable) | [+0.0470, +1.1669] | +0.0061 [+0.0004, +0.0119] | +1.3226 [+1.0009, +1.8847] (unstable) | [+1.0112, +2.1738] | 3.73 |
| R3 | -0.0034 [-0.0133, +0.0063] | +0.9251 [+0.7159, +1.1257] | [+0.6791, +1.1492] | +0.0031 [-0.0061, +0.0120] | +1.0688 [+0.8723, +1.2694] | [+0.8788, +1.3635] | 5.16 |
| R4 | -0.0089 [-0.0145, -0.0033] | +0.6042 [+0.1255, +0.8705] | [-0.1114, +0.8830] | -0.0012 [-0.0052, +0.0029] | +0.9472 [+0.7687, +1.1464] | [+0.7349, +1.1589] | 4.08 |

### dinov2ms_L11 (A14; public DINOv2-B block 11 alone; R4 trained in A14)

R4 reference sign -1, reference defined: yes; valid non-R4 cells 4; keeping the sign after Holm 4.

| cell | validity | CC per seed | mean CC | threshold | E [WCU] | Holm p | E 336 [WCU] | E 448 [WCU] | E vs control90 [WCU] | U20 [BCa] | A20 [BCa] | E [published percentile] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1n | valid | 0.7145 / 0.7130 / 0.7165 | 0.7147 | 0.7009 | -0.0532 [-0.0688, -0.0373] | 0.0016 | -0.0571 [-0.0743, -0.0394] | -0.0493 [-0.0640, -0.0341] | -0.0167 [-0.0281, -0.0048] | +0.2593 [+0.0574, +0.4028] | +0.7554 [+0.5978, +0.8500] | [-0.0673, -0.0397] |
| R1 | valid | 0.7078 / 0.7101 / 0.7083 | 0.7087 | 0.7009 | -0.0315 [-0.0450, -0.0179] | 0.0016 | -0.0309 [-0.0446, -0.0168] | -0.0322 [-0.0457, -0.0188] | -0.0030 [-0.0154, +0.0094] | -0.0882 [-0.5618, +0.1618] | +0.5494 [+0.2562, +0.7241] | [-0.0438, -0.0199] |
| R2 | valid | 0.7252 / 0.7276 / 0.7227 | 0.7252 | 0.7009 | -0.0803 [-0.1014, -0.0595] | 0.0016 | -0.0748 [-0.0973, -0.0524] | -0.0859 [-0.1083, -0.0641] | -0.0396 [-0.0529, -0.0260] | +0.6446 [+0.5343, +0.7536] | +1.0953 [+1.0251, +1.1870] | [-0.0982, -0.0626] |
| R3 | valid | 0.7271 / 0.7272 / 0.7269 | 0.7271 | 0.7009 | -0.0920 [-0.1275, -0.0558] | 0.0016 | -0.0919 [-0.1284, -0.0549] | -0.0920 [-0.1272, -0.0557] | -0.0447 [-0.0713, -0.0177] | +0.6179 [+0.5466, +0.6645] | +1.0218 [+0.9470, +1.1350] | [-0.1210, -0.0617] |
| R4 | reference (R4) | 0.7298 / 0.7284 / 0.7345 | 0.7309 | n/a (reference) | -0.0948 [-0.1241, -0.0654] | n/a (reference) | -0.0851 [-0.1174, -0.0529] | -0.1045 [-0.1325, -0.0762] | -0.0524 [-0.0735, -0.0310] | +0.6661 [+0.5649, +0.7481] | +0.9426 [+0.8844, +1.0016] | [-0.1205, -0.0706] |

| cell | seed point estimates 0 / 1 / 2 | WCR p | skewness of the cluster means of E | keeps R4's sign | interval of the opposite sign | CC against the plain DINOv2-B R4 threshold (never decision-bearing) |
|---|---|---|---|---|---|---|
| R1n | -0.0174 / -0.0423 / -0.0999 | 0.0004 | -0.298 | yes | no | n/a |
| R1 | -0.0223 / -0.0335 / -0.0388 | 0.0008 | -0.292 | yes | no | n/a |
| R2 | -0.0845 / -0.0752 / -0.0813 | 0.0004 | +0.116 | yes | no | n/a |
| R3 | -0.0609 / -0.0935 / -0.1214 | 0.0006 | +0.338 | yes | no | n/a |
| R4 | -0.0974 / -0.0998 / -0.0871 | 0.0004 | -0.202 | n/a | n/a | n/a |

| cell | L20 [WCU] | U20 [BCa] | U20 Fieller | L_alt20 [WCU] | A20 [BCa] | A20 Fieller | abs(t) of E |
|---|---|---|---|---|---|---|---|
| R1n | -0.0394 [-0.0433, -0.0355] | +0.2593 [+0.0574, +0.4028] | [-0.0016, +0.4050] | -0.0130 [-0.0192, -0.0068] | +0.7554 [+0.5978, +0.8500] | [+0.5948, +0.8750] | 7.42 |
| R1 | -0.0343 [-0.0408, -0.0278] | -0.0882 [-0.5618, +0.1618] | [-0.7084, +0.1779] | -0.0142 [-0.0227, -0.0055] | +0.5494 [+0.2562, +0.7241] | [+0.2499, +0.7814] | 5.11 |
| R2 | -0.0286 [-0.0413, -0.0161] | +0.6446 [+0.5343, +0.7536] | [+0.5180, +0.7745] | +0.0077 [+0.0019, +0.0136] | +1.0953 [+1.0251, +1.1870] | [+1.0201, +1.2055] | 8.56 |
| R3 | -0.0351 [-0.0490, -0.0217] | +0.6179 [+0.5466, +0.6645] | [+0.5558, +0.6716] | +0.0020 [-0.0065, +0.0106] | +1.0218 [+0.9470, +1.1350] | [+0.9420, +1.1618] | 5.84 |
| R4 | -0.0316 [-0.0429, -0.0204] | +0.6661 [+0.5649, +0.7481] | [+0.5486, +0.7618] | -0.0054 [-0.0125, +0.0018] | +0.9426 [+0.8844, +1.0016] | [+0.8845, +1.0197] | 7.23 |

## 2. Rule Section 6 item 1 (A4.1 per size), exploratory

| cell | size | E [WCU], p | L20 [WCU] | U20 [BCa] | A20 [BCa] | D20 [BCa] |
|---|---|---|---|---|---|---|
| dinov2reg_L6 R1n | 336 | +0.0704 [+0.0587, +0.0821], p 0.0004 | +0.0300 [+0.0112, +0.0487] | +0.5734 [+0.2831, +0.7861] | +0.9665 [+0.8636, +1.1051] | +0.0785 [-0.3466, +0.2910] (unstable) |
| dinov2reg_L6 R1n | 448 | +0.0640 [+0.0455, +0.0832], p 0.0004 | +0.0331 [+0.0198, +0.0460] | +0.4823 [+0.1284, +0.6824] | +0.8856 [+0.7654, +0.9849] | +0.2210 [+0.0363, +0.3449] |
| dinov2reg_L6 R1 | 336 | +0.0467 [+0.0321, +0.0617], p 0.0002 | +0.0174 [+0.0098, +0.0248] | +0.6273 [+0.3515, +0.7885] | +1.1023 [+0.9758, +1.3225] | -0.2745 [-0.7611, +0.0611] |
| dinov2reg_L6 R1 | 448 | +0.0300 [+0.0101, +0.0504], p 0.008 | +0.0130 [+0.0071, +0.0188] | +0.5661 [-0.2854, +0.8090] (unstable) | +1.1699 [+1.0276, +1.6671] (unstable) | -0.3915 [-0.9189, -0.0287] |
| dinov2reg_L6 R2 | 336 | +0.1065 [+0.0837, +0.1285], p 0.0002 | +0.0394 [+0.0281, +0.0506] | +0.6301 [+0.5162, +0.7153] | +0.8546 [+0.7717, +0.9305] | +0.3931 [+0.2195, +0.6280] |
| dinov2reg_L6 R2 | 448 | +0.1064 [+0.0885, +0.1236], p 0.0002 | +0.0437 [+0.0317, +0.0556] | +0.5895 [+0.4305, +0.7116] | +0.7978 [+0.6679, +0.8912] | +0.4925 [+0.3078, +0.6575] |
| dinov2reg_L6 R3 | 336 | +0.1183 [+0.0882, +0.1485], p 0.0002 | +0.0419 [+0.0261, +0.0573] | +0.6459 [+0.5041, +0.7588] | +0.8928 [+0.8128, +0.9659] | +0.3028 [+0.1045, +0.5265] |
| dinov2reg_L6 R3 | 448 | +0.0774 [+0.0419, +0.1140], p 0.0006 | +0.0351 [+0.0262, +0.0441] | +0.5462 [+0.2316, +0.7111] | +0.8953 [+0.7352, +1.0245] | +0.2306 [-0.0597, +0.4536] |
| dinov2reg_L6 R4 | 336 | +0.1081 [+0.0868, +0.1297], p 0.0002 | +0.0433 [+0.0293, +0.0566] | +0.5998 [+0.4771, +0.7020] | +0.8255 [+0.7355, +0.9060] | +0.4360 [+0.2802, +0.6003] |
| dinov2reg_L6 R4 | 448 | +0.1147 [+0.0956, +0.1333], p 0.0002 | +0.0563 [+0.0399, +0.0724] | +0.5095 [+0.3297, +0.6488] | +0.7287 [+0.5898, +0.8458] | +0.5532 [+0.4144, +0.6789] |
| dinov2reg_L12 R1n | 336 | +0.0084 [-0.0400, +0.0590], p 0.691 | -0.0033 [-0.0201, +0.0135] | not reportable (denominator interval includes 0) | not reportable (denominator interval includes 0) | not computed (uniform-loss interval includes 0) |
| dinov2reg_L12 R1n | 448 | -0.0018 [-0.0509, +0.0492], p 0.9594 | -0.0186 [-0.0381, +0.0011] | not reportable (denominator interval includes 0) | not reportable (denominator interval includes 0) | not computed (uniform-loss interval includes 0) |
| dinov2reg_L12 R1 | 336 | -0.0309 [-0.0629, +0.0020], p 0.059 | -0.0256 [-0.0416, -0.0099] | not reportable (denominator interval includes 0) | not reportable (denominator interval includes 0) | +0.5076 [+0.3005, +0.8016] (unstable) |
| dinov2reg_L12 R1 | 448 | -0.0393 [-0.0748, -0.0033], p 0.0298 | -0.0344 [-0.0517, -0.0168] | +0.1250 [-1.2652, +0.5867] (unstable) | +0.4199 [-0.1427, +0.6986] (unstable) | +0.6630 [+0.4617, +1.0979] |
| dinov2reg_L12 R2 | 336 | +0.0216 [-0.0172, +0.0607], p 0.2402 | -0.0048 [-0.0135, +0.0037] | not reportable (denominator interval includes 0) | not reportable (denominator interval includes 0) | not computed (uniform-loss interval includes 0) |
| dinov2reg_L12 R2 | 448 | +0.0293 [-0.0117, +0.0715], p 0.1434 | -0.0102 [-0.0204, -4.86e-05] | not reportable (denominator interval includes 0) | not reportable (denominator interval includes 0) | +0.2121 [-0.6917, +1.0940] (unstable) |
| dinov2reg_L12 R3 | 336 | +0.0019 [-0.0326, +0.0377], p 0.8888 | -0.0035 [-0.0199, +0.0125] | not reportable (denominator interval includes 0) | not reportable (denominator interval includes 0) | not computed (uniform-loss interval includes 0) |
| dinov2reg_L12 R3 | 448 | +0.0049 [-0.0460, +0.0570], p 0.8158 | -0.0055 [-0.0267, +0.0162] | not reportable (denominator interval includes 0) | not reportable (denominator interval includes 0) | not computed (uniform-loss interval includes 0) |
| dinov2reg_L12 R4 | 336 | +0.0463 [+0.0190, +0.0742], p 0.0008 | +0.0055 [-0.0041, +0.0150] | +0.8803 [+0.6886, +1.1124] (unstable) | +0.9082 [+0.6723, +1.0747] (unstable) | not computed (uniform-loss interval includes 0) |
| dinov2reg_L12 R4 | 448 | +0.0483 [+0.0250, +0.0721], p 0.0012 | -0.0013 [-0.0111, +0.0086] | +1.0272 [+0.8474, +1.2774] | +0.9843 [+0.7626, +1.2067] | not computed (uniform-loss interval includes 0) |
| dinov2ms_L7L11 R1n | 336 | +0.0023 [-0.0126, +0.0175], p 0.7174 | +0.0048 [-0.0105, +0.0198] | not reportable (denominator interval includes 0) | not reportable (denominator interval includes 0) | not computed (uniform-loss interval includes 0) |
| dinov2ms_L7L11 R1n | 448 | +0.0219 [+0.0084, +0.0354], p 0.0032 | +0.0145 [-0.0036, +0.0321] | +0.3415 [-0.6302, +0.9366] (unstable) | +0.5860 [+0.1671, +0.9674] (unstable) | not computed (uniform-loss interval includes 0) |
| dinov2ms_L7L11 R1 | 336 | +0.0140 [-0.0003, +0.0288], p 0.0678 | -0.0067 [-0.0159, +0.0024] | not reportable (denominator interval includes 0) | not reportable (denominator interval includes 0) | not computed (uniform-loss interval includes 0) |
| dinov2ms_L7L11 R1 | 448 | +0.0256 [+0.0118, +0.0400], p 0.0024 | -0.0017 [-0.0118, +0.0086] | +1.0661 [+0.6886, +1.5485] | +1.0236 [+0.7662, +1.5172] | not computed (uniform-loss interval includes 0) |
| dinov2ms_L7L11 R2 | 336 | -0.0248 [-0.0365, -0.0131], p 0.0012 | -0.0087 [-0.0179, +0.0002] | +0.6484 [+0.1658, +0.9381] | +1.1682 [+0.9072, +1.4728] | not computed (uniform-loss interval includes 0) |
| dinov2ms_L7L11 R2 | 448 | -0.0129 [-0.0254, -0.0005], p 0.0458 | -0.0061 [-0.0170, +0.0046] | +0.5266 [-0.4185, +1.6795] (unstable) | +1.6199 [+1.0918, +5.8224] (unstable) | not computed (uniform-loss interval includes 0) |
| dinov2ms_L7L11 R3 | 336 | -0.0513 [-0.0767, -0.0256], p 0.0024 | -0.0065 [-0.0150, +0.0018] | +0.8738 [+0.6558, +1.0204] | +1.0702 [+0.8864, +1.2575] | not computed (uniform-loss interval includes 0) |
| dinov2ms_L7L11 R3 | 448 | -0.0386 [-0.0546, -0.0221], p 0.0004 | -0.0003 [-0.0129, +0.0119] | +0.9932 [+0.7190, +1.3069] | +1.0669 [+0.8451, +1.3608] | not computed (uniform-loss interval includes 0) |
| dinov2ms_L7L11 R4 | 336 | -0.0198 [-0.0331, -0.0064], p 0.0072 | -0.0119 [-0.0183, -0.0055] | +0.4001 [-0.6177, +0.6964] (unstable) | +0.9184 [+0.6743, +1.1994] (unstable) | +0.1359 [-0.3806, +0.3849] (unstable) |
| dinov2ms_L7L11 R4 | 448 | -0.0251 [-0.0375, -0.0127], p 0.0008 | -0.0059 [-0.0141, +0.0022] | +0.7653 [+0.3715, +1.1119] | +0.9699 [+0.7742, +1.1380] | not computed (uniform-loss interval includes 0) |
| dinov2ms_L11 R1n | 336 | -0.0571 [-0.0743, -0.0394], p 0.0004 | -0.0392 [-0.0446, -0.0337] | +0.3128 [+0.0599, +0.4357] | +0.7475 [+0.5961, +0.8534] | +0.3674 [+0.2073, +0.5710] |
| dinov2ms_L11 R1n | 448 | -0.0493 [-0.0640, -0.0341], p 0.0004 | -0.0396 [-0.0456, -0.0336] | +0.1974 [-0.0083, +0.3954] | +0.7645 [+0.6039, +0.8990] | +0.2934 [+0.1623, +0.4829] |
| dinov2ms_L11 R1 | 336 | -0.0309 [-0.0446, -0.0168], p 0.0008 | -0.0342 [-0.0393, -0.0290] | -0.1081 [-0.7145, +0.1491] | +0.5344 [+0.1992, +0.7362] | +0.4201 [+0.2241, +0.6045] |
| dinov2ms_L11 R1 | 448 | -0.0322 [-0.0457, -0.0188], p 0.0006 | -0.0344 [-0.0428, -0.0262] | -0.0690 [-0.4639, +0.1849] | +0.5638 [+0.2985, +0.7252] | +0.4080 [+0.2482, +0.5617] |
| dinov2ms_L11 R2 | 336 | -0.0748 [-0.0973, -0.0524], p 0.0004 | -0.0274 [-0.0395, -0.0158] | +0.6334 [+0.4780, +0.7280] | +1.0585 [+0.9642, +1.1788] | -0.1596 [-0.5524, +0.0930] |
| dinov2ms_L11 R2 | 448 | -0.0859 [-0.1083, -0.0641], p 0.0002 | -0.0297 [-0.0459, -0.0138] | +0.6543 [+0.5363, +0.7882] | +1.1274 [+1.0409, +1.2469] | -0.3685 [-1.0247, -0.0884] |
| dinov2ms_L11 R3 | 336 | -0.0919 [-0.1284, -0.0549], p 0.0004 | -0.0317 [-0.0459, -0.0178] | +0.6556 [+0.5636, +0.7244] | +1.0232 [+0.9477, +1.1251] | -0.0675 [-0.3918, +0.1385] |
| dinov2ms_L11 R3 | 448 | -0.0920 [-0.1272, -0.0557], p 0.0006 | -0.0386 [-0.0530, -0.0239] | +0.5801 [+0.5171, +0.6369] | +1.0203 [+0.9385, +1.1754] | -0.0483 [-0.4236, +0.1452] |
| dinov2ms_L11 R4 | 336 | -0.0851 [-0.1174, -0.0529], p 0.0006 | -0.0279 [-0.0398, -0.0159] | +0.6723 [+0.5333, +0.7447] | +0.9427 [+0.8631, +1.0393] | +0.1749 [-0.1214, +0.4223] |
| dinov2ms_L11 R4 | 448 | -0.1045 [-0.1325, -0.0762], p 0.0004 | -0.0354 [-0.0502, -0.0207] | +0.6611 [+0.5495, +0.7645] | +0.9424 [+0.8928, +0.9967] | +0.1698 [+0.0250, +0.3085] |

## 3. Rule Section 6 item 3 (A4.3), exploratory

Per-seed interval counts (seed intervals excluding 0 on the side of R4's sign on the same features; seeds size-averaged):

| feature set | group | seed runs | point estimate with R4's sign | WCU interval on R4's side | published percentile interval on R4's side |
|---|---|---|---|---|---|
| dinov2reg_L6 | valid | 6 | 6 | 6 | 6 |
| dinov2reg_L6 | below_criterion | 6 | 6 | 6 | 6 |
| dinov2reg_L6 | reference_R4 | 3 | 3 | 3 | 3 |
| dinov2reg_L6 | overall | 15 | 15 | 15 | 15 |
| dinov2reg_L12 | valid | 12 | 7 | 1 | 1 |
| dinov2reg_L12 | below_criterion | 0 | 0 | 0 | 0 |
| dinov2reg_L12 | reference_R4 | 3 | 3 | 3 | 3 |
| dinov2reg_L12 | overall | 15 | 10 | 4 | 4 |
| dinov2ms_L7L11 | valid | 12 | 7 | 5 | 6 |
| dinov2ms_L7L11 | below_criterion | 0 | 0 | 0 | 0 |
| dinov2ms_L7L11 | reference_R4 | 3 | 3 | 3 | 3 |
| dinov2ms_L7L11 | overall | 15 | 10 | 8 | 9 |
| dinov2ms_L11 | valid | 12 | 12 | 12 | 12 |
| dinov2ms_L11 | below_criterion | 0 | 0 | 0 | 0 |
| dinov2ms_L11 | reference_R4 | 3 | 3 | 3 | 3 |
| dinov2ms_L11 | overall | 15 | 15 | 15 | 15 |
| all_four_sets | valid | 42 | 32 | 24 | 25 |
| all_four_sets | below_criterion | 6 | 6 | 6 | 6 |
| all_four_sets | reference_R4 | 12 | 12 | 12 | 12 |
| all_four_sets | overall | 60 | 50 | 42 | 43 |
| all_four_sets | new_cells_only | 54 | 44 | 36 | 37 |
- not counted under WCU: dinov2reg_L12 R1n seed 0, -0.0224 [-0.0668, +0.0236]
- not counted under WCU: dinov2reg_L12 R1n seed 1, +0.0316 [-0.0187, +0.0833]
- not counted under WCU: dinov2reg_L12 R1n seed 2, +0.0006 [-0.0532, +0.0557]
- not counted under WCU: dinov2reg_L12 R1 seed 0, -0.0329 [-0.0682, +0.0038]
- not counted under WCU: dinov2reg_L12 R1 seed 1, -0.0874 [-0.1284, -0.0462]
- not counted under WCU: dinov2reg_L12 R1 seed 2, +0.0151 [-0.0126, +0.0440]
- not counted under WCU: dinov2reg_L12 R2 seed 0, +0.0185 [-0.0337, +0.0717]
- not counted under WCU: dinov2reg_L12 R2 seed 1, +0.0190 [-0.0225, +0.0622]
- not counted under WCU: dinov2reg_L12 R3 seed 0, +0.0300 [-0.0189, +0.0807]
- not counted under WCU: dinov2reg_L12 R3 seed 1, -0.0163 [-0.0541, +0.0233]
- not counted under WCU: dinov2reg_L12 R3 seed 2, -0.0035 [-0.0536, +0.0479]
- not counted under WCU: dinov2ms_L7L11 R1n seed 0, +0.0016 [-0.0107, +0.0144]
- not counted under WCU: dinov2ms_L7L11 R1n seed 1, +0.0168 [-0.0059, +0.0399]
- not counted under WCU: dinov2ms_L7L11 R1n seed 2, +0.0180 [+0.0061, +0.0298]
- not counted under WCU: dinov2ms_L7L11 R1 seed 0, +0.0459 [+0.0295, +0.0630]
- not counted under WCU: dinov2ms_L7L11 R1 seed 1, +0.0338 [+0.0220, +0.0461]
- not counted under WCU: dinov2ms_L7L11 R2 seed 0, -0.0114 [-0.0234, +0.0006]
- not counted under WCU: dinov2ms_L7L11 R3 seed 0, -0.0072 [-0.0209, +0.0066]
- not counted under the percentile interval: dinov2reg_L12 R1n seed 0, -0.0224 [-0.0605, +0.0145]
- not counted under the percentile interval: dinov2reg_L12 R1n seed 1, +0.0316 [-0.0130, +0.0714]
- not counted under the percentile interval: dinov2reg_L12 R1n seed 2, +0.0006 [-0.0473, +0.0426]
- not counted under the percentile interval: dinov2reg_L12 R1 seed 0, -0.0329 [-0.0656, -0.0049]
- not counted under the percentile interval: dinov2reg_L12 R1 seed 1, -0.0874 [-0.1219, -0.0527]
- not counted under the percentile interval: dinov2reg_L12 R1 seed 2, +0.0151 [-0.0104, +0.0367]
- not counted under the percentile interval: dinov2reg_L12 R2 seed 0, +0.0185 [-0.0254, +0.0610]
- not counted under the percentile interval: dinov2reg_L12 R2 seed 1, +0.0190 [-0.0150, +0.0557]
- not counted under the percentile interval: dinov2reg_L12 R3 seed 0, +0.0300 [-0.0108, +0.0722]
- not counted under the percentile interval: dinov2reg_L12 R3 seed 1, -0.0163 [-0.0512, +0.0156]
- not counted under the percentile interval: dinov2reg_L12 R3 seed 2, -0.0035 [-0.0471, +0.0377]
- not counted under the percentile interval: dinov2ms_L7L11 R1n seed 0, +0.0016 [-0.0102, +0.0131]
- not counted under the percentile interval: dinov2ms_L7L11 R1n seed 1, +0.0168 [-0.0025, +0.0370]
- not counted under the percentile interval: dinov2ms_L7L11 R1n seed 2, +0.0180 [+0.0063, +0.0292]
- not counted under the percentile interval: dinov2ms_L7L11 R1 seed 0, +0.0459 [+0.0311, +0.0596]
- not counted under the percentile interval: dinov2ms_L7L11 R1 seed 1, +0.0338 [+0.0220, +0.0440]
- not counted under the percentile interval: dinov2ms_L7L11 R3 seed 0, -0.0072 [-0.0187, +0.0053]

Profiles (size average, seeds averaged): L_uni, U, A, D at 10 / 20 / 30 degrees; slope of E per unit support ratio (two-stage percentile); two-level seed bootstrap interval of E (percentile).

| cell | L_uni 10 / 20 / 30 | U 10 / 20 / 30 | A 10 / 20 / 30 | D 10 / 20 / 30 | slope of E [pct] | E two-level [pct] |
|---|---|---|---|---|---|---|
| dinov2reg_L6 R1n | +0.0110* / +0.0316* / +0.0449* | +0.836 / +0.530 / +0.332 | +1.144 / +0.928 / +0.655 | -0.880u / +0.153 / +0.516 | -0.0856 [-0.1736, -0.0141] | [+0.0483, +0.0881] (above 0) |
| dinov2reg_L6 R1 | +0.0016 / +0.0152* / +0.0257* | +0.959 / +0.603 / +0.331 | +1.411 / +1.129 / +0.804 | nc / -0.325 / +0.293 | -0.0859 [-0.1711, -0.0074] | [+0.0224, +0.0546] (above 0) |
| dinov2reg_L6 R2 | +0.0116* / +0.0415* / +0.0701* | +0.891 / +0.610 / +0.342 | +1.003 / +0.826 / +0.602 | -0.027 / +0.445 / +0.605 | +0.1039 [+0.0078, +0.1905] | [+0.0892, +0.1232] (above 0) |
| dinov2reg_L6 R3 | +0.0106* / +0.0385* / +0.0603* | +0.891 / +0.606 / +0.384 | +0.985 / +0.894 / +0.744 | +0.142u / +0.270 / +0.415 | +0.0108 [-0.1693, +0.1780] | [+0.0686, +0.1270] (above 0) |
| dinov2reg_L6 R4 | +0.0196* / +0.0498* / +0.0802* | +0.824 / +0.553 / +0.280 | +0.935 / +0.776 / +0.567 | +0.371 / +0.502 / +0.602 | +0.1042 [-0.0021, +0.2047] | [+0.0918, +0.1301] (above 0) |
| dinov2reg_L12 R1n | -0.0014 / -0.0109 / -0.0056 | nr / nr / nr | nr / nr / nr | nc / nc / nc | -0.2933 [-0.4901, -0.0964] | [-0.0446, +0.0511] (includes 0) |
| dinov2reg_L12 R1 | -0.0120* / -0.0300* / -0.0320* | +0.658u / +0.146u / +0.087u | +0.793u / +0.490u / +0.296u | +0.604u / +0.597 / +0.771u | -0.2275 [-0.3550, -0.1038] | [-0.0928, +0.0159] (includes 0, changed) |
| dinov2reg_L12 R2 | -0.0113* / -0.0075 / +0.0118* | nr / nr / nr | nr / nr / nr | +0.723u / nc / +0.626u | -0.0612 [-0.2704, +0.1449] | [-0.0121, +0.0596] (includes 0) |
| dinov2reg_L12 R3 | -0.0006 / -0.0045 / +0.0041 | nr / nr / nr | nr / nr / nr | nc / nc / nc | -0.3094 [-0.4338, -0.1787] | [-0.0393, +0.0476] (includes 0) |
| dinov2reg_L12 R4 | -0.0070* / +0.0021 / +0.0191* | +1.147 / +0.955 / +0.597 | +1.120 / +0.947 / +0.786 | +0.816u / nc / +0.530u | -0.0005 [-0.1314, +0.1337] | [+0.0211, +0.0739] (above 0) |
| dinov2ms_L7L11 R1n | +0.0008 / +0.0096 / +0.0157 | nr / nr / nr | nr / nr / nr | nc / nc / nc | -0.0483 [-0.1163, +0.0317] | [-0.0033, +0.0283] (includes 0) |
| dinov2ms_L7L11 R1 | -0.0102* / -0.0042 / +0.0137* | +1.515u / +1.212u / +0.305u | +1.296u / +1.041u / +0.646u | +0.574u / nc / +0.509u | -0.0679 [-0.1431, +0.0072] | [-0.0176, +0.0500] (includes 0, changed) |
| dinov2ms_L7L11 R2 | -0.0011 / -0.0074 / +0.0002 | +0.939u / +0.607u / +1.011u | +1.294u / +1.323u / +1.255u | nc / nc / nc | -0.0466 [-0.0911, +0.0016] | [-0.0301, -0.0073] (below 0) |
| dinov2ms_L7L11 R3 | -0.0055 / -0.0034 / -0.0030 | +0.877 / +0.925 / +0.932 | +1.081 / +1.069 / +0.856 | nc / nc / nc | -0.1588 [-0.2038, -0.1116] | [-0.0807, -0.0093] (below 0) |
| dinov2ms_L7L11 R4 | -0.0057 / -0.0089* / -0.0031 | +0.747 / +0.604 / +0.860 | +0.911 / +0.947 / +0.914 | nc / +0.133u / nc | -0.0687 [-0.1292, -0.0098] | [-0.0356, -0.0114] (below 0) |
| dinov2ms_L11 R1n | -0.0282* / -0.0394* / -0.0396* | +0.469 / +0.259 / +0.255 | +0.746 / +0.755 / +0.700 | +0.479 / +0.330 / +0.403 | -0.1036 [-0.1762, -0.0297] | [-0.0971, -0.0167] (below 0) |
| dinov2ms_L11 R1 | -0.0259* / -0.0343* / -0.0304* | +0.178 / -0.088 / +0.037 | +0.535 / +0.549 / +0.638 | +0.565 / +0.414 / +0.375 | -0.0902 [-0.1556, -0.0313] | [-0.0464, -0.0175] (below 0) |
| dinov2ms_L11 R2 | -0.0122* / -0.0286* / -0.0295* | +0.849 / +0.645 / +0.633 | +1.105 / +1.095 / +1.016 | -0.692u / -0.268 / -0.044 | -0.1585 [-0.2273, -0.0920] | [-0.0992, -0.0620] (below 0) |
| dinov2ms_L11 R3 | -0.0203* / -0.0351* / -0.0352* | +0.779 / +0.618 / +0.618 | +1.072 / +1.022 / +0.902 | -0.327 / -0.057 / +0.257 | -0.3026 [-0.3658, -0.2411] | [-0.1357, -0.0523] (below 0) |
| dinov2ms_L11 R4 | -0.0143* / -0.0316* / -0.0387* | +0.849 / +0.666 / +0.592 | +1.003 / +0.943 / +0.807 | -0.018u / +0.172 / +0.472 | -0.2119 [-0.3294, -0.0992] | [-0.1218, -0.0696] (below 0) |

`*` WCU interval excludes 0; `u` ratio unstable; `nr` not reportable (denominator interval includes 0; rule 4.3); `nc` D not computed (uniform-loss interval includes 0). Intervals of U, A and D are in the JSON.

Per-seed values (seeds size-averaged): E with its WCU interval and WCR p, E with the published percentile interval, L_uni(20) with its WCU interval, U20 and A20 as points (`nr` where that seed's E WCU interval includes 0; rule 4.3; the points are in the JSON).

| cell | seed | E [WCU], p | E [published percentile] | L_uni 20 [WCU] | U20 | A20 |
|---|---|---|---|---|---|---|
| dinov2reg_L6 R1n | 0 | +0.0637 [+0.0502, +0.0776], p 0.0002 | [+0.0519, +0.0772] | +0.0412 [+0.0228, +0.0592] | +0.353 | +0.831 |
| dinov2reg_L6 R1n | 1 | +0.0537 [+0.0386, +0.0691], p 0.0002 | [+0.0403, +0.0685] | +0.0275 [+0.0126, +0.0417] | +0.488 | +0.880 |
| dinov2reg_L6 R1n | 2 | +0.0842 [+0.0686, +0.0998], p 0.0004 | [+0.0704, +0.1019] | +0.0260 [+0.0137, +0.0379] | +0.691 | +1.032 |
| dinov2reg_L6 R1 | 0 | +0.0334 [+0.0156, +0.0516], p 0.003 | [+0.0175, +0.0490] | +0.0194 [+0.0117, +0.0269] | +0.419 | +1.032 |
| dinov2reg_L6 R1 | 1 | +0.0371 [+0.0176, +0.0567], p 0.0006 | [+0.0207, +0.0539] | +0.0107 [+0.0070, +0.0146] | +0.712 | +1.297 |
| dinov2reg_L6 R1 | 2 | +0.0446 [+0.0274, +0.0619], p 0.0004 | [+0.0295, +0.0596] | +0.0156 [+0.0074, +0.0233] | +0.651 | +1.061 |
| dinov2reg_L6 R2 | 0 | +0.1096 [+0.0903, +0.1285], p 0.0002 | [+0.0932, +0.1260] | +0.0505 [+0.0352, +0.0653] | +0.540 | +0.771 |
| dinov2reg_L6 R2 | 1 | +0.1032 [+0.0818, +0.1242], p 0.0002 | [+0.0842, +0.1217] | +0.0383 [+0.0303, +0.0462] | +0.629 | +0.872 |
| dinov2reg_L6 R2 | 2 | +0.1067 [+0.0900, +0.1233], p 0.0002 | [+0.0917, +0.1217] | +0.0359 [+0.0265, +0.0447] | +0.664 | +0.838 |
| dinov2reg_L6 R3 | 0 | +0.1109 [+0.0832, +0.1388], p 0.0002 | [+0.0863, +0.1354] | +0.0522 [+0.0387, +0.0653] | +0.529 | +0.825 |
| dinov2reg_L6 R3 | 1 | +0.0989 [+0.0625, +0.1356], p 0.0002 | [+0.0680, +0.1295] | +0.0360 [+0.0284, +0.0436] | +0.636 | +0.961 |
| dinov2reg_L6 R3 | 2 | +0.0838 [+0.0563, +0.1113], p 0.0002 | [+0.0594, +0.1081] | +0.0274 [+0.0192, +0.0351] | +0.673 | +0.906 |
| dinov2reg_L6 R4 | 0 | +0.1134 [+0.0940, +0.1323], p 0.0002 | [+0.0953, +0.1298] | +0.0477 [+0.0335, +0.0618] | +0.579 | +0.776 |
| dinov2reg_L6 R4 | 1 | +0.1025 [+0.0831, +0.1224], p 0.0002 | [+0.0847, +0.1200] | +0.0651 [+0.0465, +0.0841] | +0.365 | +0.694 |
| dinov2reg_L6 R4 | 2 | +0.1183 [+0.0989, +0.1378], p 0.0002 | [+0.1002, +0.1360] | +0.0364 [+0.0251, +0.0475] | +0.692 | +0.845 |
| dinov2reg_L12 R1n | 0 | -0.0224 [-0.0668, +0.0236], p 0.306 | [-0.0605, +0.0145] | -0.0202 [-0.0419, +0.0012] | nr | nr |
| dinov2reg_L12 R1n | 1 | +0.0316 [-0.0187, +0.0833], p 0.1892 | [-0.0130, +0.0714] | -0.0120 [-0.0327, +0.0086] | nr | nr |
| dinov2reg_L12 R1n | 2 | +0.0006 [-0.0532, +0.0557], p 0.9538 | [-0.0473, +0.0426] | -0.0006 [-0.0143, +0.0133] | nr | nr |
| dinov2reg_L12 R1 | 0 | -0.0329 [-0.0682, +0.0038], p 0.0696 | [-0.0656, -0.0049] | -0.0259 [-0.0393, -0.0123] | nr | nr |
| dinov2reg_L12 R1 | 1 | -0.0874 [-0.1284, -0.0462], p 0.001 | [-0.1219, -0.0527] | -0.0520 [-0.0738, -0.0303] | +0.405 | +0.671 |
| dinov2reg_L12 R1 | 2 | +0.0151 [-0.0126, +0.0440], p 0.2606 | [-0.0104, +0.0367] | -0.0120 [-0.0332, +0.0087] | nr | nr |
| dinov2reg_L12 R2 | 0 | +0.0185 [-0.0337, +0.0717], p 0.4382 | [-0.0254, +0.0610] | -0.0129 [-0.0248, -0.0008] | nr | nr |
| dinov2reg_L12 R2 | 1 | +0.0190 [-0.0225, +0.0622], p 0.3352 | [-0.0150, +0.0557] | -0.0015 [-0.0140, +0.0115] | nr | nr |
| dinov2reg_L12 R2 | 2 | +0.0388 [+0.0111, +0.0669], p 0.0096 | [+0.0154, +0.0627] | -0.0082 [-0.0169, +0.0006] | +1.212 | +0.920 |
| dinov2reg_L12 R3 | 0 | +0.0300 [-0.0189, +0.0807], p 0.1952 | [-0.0108, +0.0722] | +0.0202 [+0.0002, +0.0402] | nr | nr |
| dinov2reg_L12 R3 | 1 | -0.0163 [-0.0541, +0.0233], p 0.3704 | [-0.0512, +0.0156] | -0.0262 [-0.0543, +0.0016] | nr | nr |
| dinov2reg_L12 R3 | 2 | -0.0035 [-0.0536, +0.0479], p 0.8948 | [-0.0471, +0.0377] | -0.0075 [-0.0211, +0.0060] | nr | nr |
| dinov2reg_L12 R4 | 0 | +0.0652 [+0.0422, +0.0885], p 0.0004 | [+0.0463, +0.0860] | +0.0149 [+0.0051, +0.0248] | +0.772 | +0.828 |
| dinov2reg_L12 R4 | 1 | +0.0366 [+0.0014, +0.0732], p 0.047 | [+0.0078, +0.0672] | -0.0015 [-0.0125, +0.0095] | +1.040 | +0.867 |
| dinov2reg_L12 R4 | 2 | +0.0400 [+0.0207, +0.0593], p 0.001 | [+0.0232, +0.0574] | -0.0071 [-0.0159, +0.0019] | +1.177 | +1.214 |
| dinov2ms_L7L11 R1n | 0 | +0.0016 [-0.0107, +0.0144], p 0.7738 | [-0.0102, +0.0131] | +0.0124 [-0.0047, +0.0290] | nr | nr |
| dinov2ms_L7L11 R1n | 1 | +0.0168 [-0.0059, +0.0399], p 0.133 | [-0.0025, +0.0370] | +0.0083 [-0.0162, +0.0321] | nr | nr |
| dinov2ms_L7L11 R1n | 2 | +0.0180 [+0.0061, +0.0298], p 0.0078 | [+0.0063, +0.0292] | +0.0082 [-0.0019, +0.0182] | +0.542 | +1.047 |
| dinov2ms_L7L11 R1 | 0 | +0.0459 [+0.0295, +0.0630], p 0.0008 | [+0.0311, +0.0596] | -0.0004 [-0.0099, +0.0092] | +1.008 | +1.117 |
| dinov2ms_L7L11 R1 | 1 | +0.0338 [+0.0220, +0.0461], p 0.0012 | [+0.0220, +0.0440] | -0.0027 [-0.0118, +0.0064] | +1.079 | +0.979 |
| dinov2ms_L7L11 R1 | 2 | -0.0204 [-0.0350, -0.0051], p 0.0068 | [-0.0349, -0.0077] | -0.0096 [-0.0225, +0.0039] | +0.531 | +1.109 |
| dinov2ms_L7L11 R2 | 0 | -0.0114 [-0.0234, +0.0006], p 0.0562 | [-0.0220, -0.0012] | -0.0072 [-0.0202, +0.0055] | nr | nr |
| dinov2ms_L7L11 R2 | 1 | -0.0214 [-0.0339, -0.0089], p 0.0042 | [-0.0324, -0.0104] | -0.0050 [-0.0135, +0.0034] | +0.765 | +1.293 |
| dinov2ms_L7L11 R2 | 2 | -0.0236 [-0.0342, -0.0130], p 0.0006 | [-0.0330, -0.0145] | -0.0100 [-0.0189, -0.0012] | +0.578 | +1.175 |
| dinov2ms_L7L11 R3 | 0 | -0.0072 [-0.0209, +0.0066], p 0.2716 | [-0.0187, +0.0053] | +0.0088 [-0.0021, +0.0196] | nr | nr |
| dinov2ms_L7L11 R3 | 1 | -0.0593 [-0.0811, -0.0369], p 0.0002 | [-0.0781, -0.0399] | -0.0170 [-0.0315, -0.0030] | +0.713 | +1.169 |
| dinov2ms_L7L11 R3 | 2 | -0.0684 [-0.0993, -0.0367], p 0.0008 | [-0.0952, -0.0420] | -0.0019 [-0.0114, +0.0073] | +0.972 | +0.942 |
| dinov2ms_L7L11 R4 | 0 | -0.0185 [-0.0289, -0.0076], p 0.0022 | [-0.0281, -0.0094] | -0.0038 [-0.0100, +0.0022] | +0.793 | +0.981 |
| dinov2ms_L7L11 R4 | 1 | -0.0284 [-0.0442, -0.0125], p 0.0016 | [-0.0435, -0.0158] | -0.0060 [-0.0119, -2.12e-06] | +0.790 | +1.006 |
| dinov2ms_L7L11 R4 | 2 | -0.0205 [-0.0344, -0.0067], p 0.0074 | [-0.0326, -0.0090] | -0.0169 [-0.0235, -0.0102] | +0.176 | +0.836 |
| dinov2ms_L11 R1n | 0 | -0.0174 [-0.0308, -0.0033], p 0.009 | [-0.0308, -0.0061] | -0.0374 [-0.0462, -0.0286] | -1.153 | +0.254 |
| dinov2ms_L11 R1n | 1 | -0.0423 [-0.0575, -0.0266], p 0.0006 | [-0.0557, -0.0296] | -0.0386 [-0.0469, -0.0301] | +0.089 | +0.684 |
| dinov2ms_L11 R1n | 2 | -0.0999 [-0.1208, -0.0780], p 0.0004 | [-0.1184, -0.0811] | -0.0422 [-0.0552, -0.0297] | +0.577 | +0.873 |
| dinov2ms_L11 R1 | 0 | -0.0223 [-0.0340, -0.0100], p 0.0016 | [-0.0328, -0.0120] | -0.0356 [-0.0440, -0.0271] | -0.600 | +0.199 |
| dinov2ms_L11 R1 | 1 | -0.0335 [-0.0504, -0.0167], p 0.001 | [-0.0488, -0.0192] | -0.0391 [-0.0495, -0.0286] | -0.166 | +0.677 |
| dinov2ms_L11 R1 | 2 | -0.0388 [-0.0518, -0.0255], p 0.0004 | [-0.0511, -0.0275] | -0.0282 [-0.0323, -0.0242] | +0.272 | +0.640 |
| dinov2ms_L11 R2 | 0 | -0.0845 [-0.1113, -0.0577], p 0.0004 | [-0.1071, -0.0622] | -0.0394 [-0.0552, -0.0235] | +0.534 | +1.097 |
| dinov2ms_L11 R2 | 1 | -0.0752 [-0.0952, -0.0557], p 0.0002 | [-0.0918, -0.0584] | -0.0134 [-0.0280, +0.0008] | +0.822 | +1.200 |
| dinov2ms_L11 R2 | 2 | -0.0813 [-0.1000, -0.0631], p 0.0004 | [-0.0973, -0.0655] | -0.0329 [-0.0431, -0.0230] | +0.595 | +0.997 |
| dinov2ms_L11 R3 | 0 | -0.0609 [-0.0927, -0.0290], p 0.0014 | [-0.0871, -0.0337] | -0.0364 [-0.0508, -0.0222] | +0.402 | +1.038 |
| dinov2ms_L11 R3 | 1 | -0.0935 [-0.1262, -0.0606], p 0.0004 | [-0.1217, -0.0657] | -0.0378 [-0.0568, -0.0189] | +0.595 | +0.999 |
| dinov2ms_L11 R3 | 2 | -0.1214 [-0.1656, -0.0773], p 0.0002 | [-0.1573, -0.0841] | -0.0311 [-0.0449, -0.0177] | +0.744 | +1.031 |
| dinov2ms_L11 R4 | 0 | -0.0974 [-0.1269, -0.0675], p 0.0004 | [-0.1242, -0.0729] | -0.0241 [-0.0343, -0.0140] | +0.752 | +1.008 |
| dinov2ms_L11 R4 | 1 | -0.0998 [-0.1348, -0.0644], p 0.0006 | [-0.1295, -0.0701] | -0.0281 [-0.0390, -0.0171] | +0.718 | +0.969 |
| dinov2ms_L11 R4 | 2 | -0.0871 [-0.1148, -0.0597], p 0.0006 | [-0.1112, -0.0647] | -0.0427 [-0.0571, -0.0284] | +0.510 | +0.839 |

Two-level seed bootstrap: the interval of E changes position relative to 0 in 2 of 20 units.

## 4. Rule Section 6 item 5 (A4.5 core share f_core30, illusory minus control180), exploratory

| cell | f_core30 effect [WCU] | WCR p | skewness of the cluster means | published percentile interval |
|---|---|---|---|---|
| dinov2reg_L6 R1n | +0.0314 [+0.0207, +0.0420] | 0.0002 | +0.083 | [+0.0227, +0.0405] |
| dinov2reg_L6 R1 | +0.0163 [+0.0072, +0.0257] | 0.0016 | +0.154 | [+0.0085, +0.0242] |
| dinov2reg_L6 R2 | +0.0511 [+0.0452, +0.0572] | 0.0002 | +0.367 | [+0.0456, +0.0567] |
| dinov2reg_L6 R3 | +0.0363 [+0.0269, +0.0460] | 0.0002 | +0.197 | [+0.0276, +0.0447] |
| dinov2reg_L6 R4 | +0.0546 [+0.0472, +0.0625] | 0.0002 | -0.485 | [+0.0475, +0.0612] |
| dinov2reg_L12 R1n | -0.0119 [-0.0395, +0.0170] | 0.3828 | -0.364 | [-0.0359, +0.0107] |
| dinov2reg_L12 R1 | -0.0216 [-0.0367, -0.0062] | 0.0048 | -0.457 | [-0.0349, -0.0095] |
| dinov2reg_L12 R2 | -0.0016 [-0.0225, +0.0200] | 0.8772 | +0.208 | [-0.0193, +0.0161] |
| dinov2reg_L12 R3 | -0.0091 [-0.0381, +0.0201] | 0.5098 | +0.005 | [-0.0335, +0.0152] |
| dinov2reg_L12 R4 | +0.0138 [+0.0013, +0.0263] | 0.0324 | -0.079 | [+0.0029, +0.0246] |
| dinov2ms_L7L11 R1n | +0.0206 [+0.0119, +0.0296] | 0.0004 | +0.262 | [+0.0127, +0.0287] |
| dinov2ms_L7L11 R1 | +0.0167 [+0.0082, +0.0258] | 0.002 | -0.316 | [+0.0088, +0.0240] |
| dinov2ms_L7L11 R2 | -0.0062 [-0.0101, -0.0021] | 0.0112 | +0.599 | [-0.0100, -0.0021] |
| dinov2ms_L7L11 R3 | -0.0181 [-0.0240, -0.0119] | 0.0002 | +0.509 | [-0.0233, -0.0125] |
| dinov2ms_L7L11 R4 | -0.0100 [-0.0152, -0.0046] | 0.0026 | +0.267 | [-0.0145, -0.0052] |
| dinov2ms_L11 R1n | -0.0287 [-0.0433, -0.0141] | 0.0002 | -0.414 | [-0.0408, -0.0170] |
| dinov2ms_L11 R1 | -0.0157 [-0.0254, -0.0058] | 0.0024 | +0.006 | [-0.0238, -0.0078] |
| dinov2ms_L11 R2 | -0.0412 [-0.0493, -0.0331] | 0.0002 | -0.059 | [-0.0486, -0.0340] |
| dinov2ms_L11 R3 | -0.0485 [-0.0611, -0.0354] | 0.0002 | +0.355 | [-0.0591, -0.0372] |
| dinov2ms_L11 R4 | -0.0515 [-0.0651, -0.0371] | 0.0004 | -0.190 | [-0.0634, -0.0398] |

## 5. Not computable from saved outputs

- Units: none. Every score and MIT1003 file of the 20 units exists; every rule 5.6 item and every part of items 1, 3 and 5 was computed for every unit.
- Rule 6 item 4 (A4.4 log mass): not computable from saved outputs; needs per-display maps (masses in the notch sectors and the inducer ink); the G3 and A14 runs saved no maps (V3_MAPS=none in gpu/run_gpu_re.sh; 0 files under gpu/g3/maps and gpu/a14/maps); computable only after re-inferring maps on the CPU from the saved checkpoints (a new scoring step with its own pipeline check).
- Rule 6 item 6 (A4.6 calibration robustness): not computable from saved outputs; needs logits re-inferred heads-only from a float16 stim_v3 feature cache or a backbone pass: no feature cache for the new feature sets (v2/readout/cache files found: 0) and the G3 and A14 runs saved no maps (V3_MAPS=none in gpu/run_gpu_re.sh; 0 files under gpu/g3/maps and gpu/a14/maps); computable only after re-inferring maps on the CPU from the saved checkpoints (a new scoring step with its own pipeline check).
- Rule 6 item 7 (A6 second part: absolute CC, NSS and information gain with the centre bias off): not computable from saved outputs; heads-only from cached MIT1003 features: no <fset>_mit200.npy cache for the new feature sets (found: 0).
- Rule 6 item 10 (A12 Kavsek-style AOIs): not computable from saved outputs; needs per-display maps (masses in the whole contour area P); the G3 and A14 runs saved no maps (V3_MAPS=none in gpu/run_gpu_re.sh; 0 files under gpu/g3/maps and gpu/a14/maps); computable only after re-inferring maps on the CPU from the saved checkpoints (a new scoring step with its own pipeline check).
- Rule 6 item 11 (A13 region variant): not computable from saved outputs; needs per-display maps (masses in P and in D minus P); the G3 and A14 runs saved no maps (V3_MAPS=none in gpu/run_gpu_re.sh; 0 files under gpu/g3/maps and gpu/a14/maps); computable only after re-inferring maps on the CPU from the saved checkpoints (a new scoring step with its own pipeline check).
- Rule 6 item 15 (rigid rotation outside P5: E_rig, L_rig, A* and s*): not computable from saved outputs; the G3 and A14 heads were not scored on stim_rigid (scores_rigid/heads/scores files for them: 0); rule 5.5 includes them only 'if they are scored on these displays later'.

## 6. Not computed by this script

- Rule 6 item 2 (A4.2 ratio E90 / E180): not computed (not listed for the new cells); rule 6.2 lists the 20 swap cells, IIE, MSDB and the 15 nulls; E against control90 of the new cells is the rule 5.6 item and is reported in Section 1.
- Rule 6 item 7 (A6 first part: validity margin curve): not computed (described for the swap cells); rule 6.7 and rule 1.1 item 4 describe the curve for the 20 swap cells; the stored MIT1003 CCs of the new cells would allow it, but no margin curve is defined for them.

