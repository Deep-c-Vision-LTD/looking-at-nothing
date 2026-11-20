# A3 token analysis of the published ViTs under both nulls (RUN_PLAN S5)

**EXPLORATORY** (REANALYSIS_RULE.md Section 6 item 8 (i), (ii), (iii), (v); never decision-bearing; no headline dissociation is drawn from either null, rule 7.6). The pre-specified rule of 15 October 2026 (background and control90 nulls) is the analysis of record; the near-exterior null is post hoc. Registers checkpoint (iv): `A3iv_registers_tokens.json` (earlier step).

## 0. Reproduction of the published token analyses (copies in explore/tokens/, outputs in A3_tokens/published/)

- results.json: 27577 leaves (20276 numeric), max |difference| 0.0, mismatches 0 (identical: True).
- encoding_nulls.json: 1488 leaves (1248 numeric), max |difference| 0.0, mismatches 0 (identical: True).
- per_config_indices.npz: same keys True, 45 float arrays, max |difference| 0.0.
- Published verdicts reproduced from the bootstrap with the published rounding: 180 of 180 (all blocks, three variants, three ViTs). Verdict changes when unrounded bounds are used: {'dinov2': [], 'clip': [], 'siglip': []}.

## 1. Blocks that pass, by null (size average)

| network | (i) pre-specified, unrounded | (i) published rounding | (ii) near-exterior | both |
|---|---|---|---|---|
| DINOv2-B | [4, 5, 6, 7, 9] | [4, 5, 6, 7, 9] | [6, 7, 10, 11, 12] | [6, 7] |
| CLIP-L | [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] | [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] | [] | [] |
| SigLIP-L | [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] | [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] | [7] | [] |

- Rule 7.6: CLIP-L pre-specified 7 to 24: True; CLIP-L near-exterior none: True; blocks read by a readout passing both: {'dinov2': [6], 'clip': [], 'siglip': []} (only DINOv2-B 6: True); DINOv2-B block 7 passes both: True.
- Sweep agreement (pre-specified verdict with the saliency sign): percentile intervals 12 of 18, disagreeing ['dinov2_L8', 'clip_L20', 'clip_L24', 'siglip_L4', 'siglip_L8', 'siglip_L16'] (rule 1.1 item 9 six reproduced: True); WCU intervals 12 of 18, disagreeing ['dinov2_L8', 'clip_L20', 'clip_L24', 'siglip_L4', 'siglip_L8', 'siglip_L16'].

## 2. (iii) Paired interior-minus-near-exterior excess (size average; WCU) and its retention under mis20 (BCa)

| network | block | excess [WCU], p | retention mis20 [BCa] |
|---|---|---|---|
| DINOv2-B | 1 | -0.2907 [-0.5166, -0.0596], p 0.0066 | 0.8414 [+0.6282, +1.1123] (unstable) |
| DINOv2-B | 2 | -0.0292 [-0.1409, +0.0818], p 0.611 | 1.5768 [+0.9883, +281.8840] (not reportable (denominator interval includes 0)) |
| DINOv2-B | 3 | +0.0129 [-0.0491, +0.0745], p 0.7258 | -1.3314 [-9858.9505, -0.0670] (not reportable (denominator interval includes 0)) |
| DINOv2-B | 4 | +0.0106 [-0.0174, +0.0378], p 0.4112 | -1.4663 [-47.9312, +3.0850] (not reportable (denominator interval includes 0)) |
| DINOv2-B | 5 | +0.0460 [+0.0124, +0.0785], p 0.0122 | -0.4411 [-1.9729, -0.1150] (unstable) |
| DINOv2-B | 6 | +0.1093 [+0.0814, +0.1367], p 0.0004 | -0.0873 [-0.3949, +0.1007] |
| DINOv2-B | 7 | +0.1078 [+0.0871, +0.1280], p 0.0002 | 0.1189 [-0.1951, +0.3575] |
| DINOv2-B | 8 | +0.0744 [+0.0465, +0.1017], p 0.0002 | -0.4540 [-0.9458, -0.0388] |
| DINOv2-B | 9 | +0.0725 [+0.0218, +0.1237], p 0.0094 | -0.3904 [-1.4610, +0.0935] (unstable) |
| DINOv2-B | 10 | +0.1000 [+0.0237, +0.1751], p 0.0126 | 0.3454 [-0.1422, +0.6729] (unstable) |
| DINOv2-B | 11 | +0.1580 [+0.0572, +0.2567], p 0.0028 | 0.5647 [+0.4005, +0.7209] (unstable) |
| DINOv2-B | 12 | +0.1861 [+0.0699, +0.3003], p 0.0018 | 0.8047 [+0.7324, +0.9507] (unstable) |
| CLIP-L | 1 | -0.0503 [-0.1009, +0.0025], p 0.0676 | 0.8784 [+0.5754, +1.0549] (not reportable (denominator interval includes 0)) |
| CLIP-L | 2 | -0.1106 [-0.2027, -0.0159], p 0.0034 | 0.4847 [+0.1891, +0.6531] (unstable) |
| CLIP-L | 3 | -0.0922 [-0.1626, -0.0205], p 0.0046 | 0.9596 [+0.2818, +1.3894] (unstable) |
| CLIP-L | 4 | +0.0459 [+0.0105, +0.0809], p 0.0086 | 0.1150 [-82.7868, +0.6719] (unstable) |
| CLIP-L | 5 | +0.0475 [+0.0104, +0.0840], p 0.011 | -0.0373 [-1.5500, +1.3508] (unstable) |
| CLIP-L | 6 | +0.0608 [+0.0135, +0.1075], p 0.0086 | 0.3772 [-0.2139, +1.1847] (unstable) |
| CLIP-L | 7 | +0.0882 [+0.0371, +0.1386], p 0.0006 | 0.4403 [+0.2340, +0.5857] (unstable) |
| CLIP-L | 8 | +0.1101 [+0.0472, +0.1711], p 0.0006 | 0.5462 [+0.4465, +0.7045] (unstable) |
| CLIP-L | 9 | +0.1190 [+0.0532, +0.1838], p 0.0006 | 0.3730 [+0.1934, +0.5038] |
| CLIP-L | 10 | +0.1321 [+0.0551, +0.2068], p 0.0006 | 0.5040 [+0.3995, +0.6101] (unstable) |
| CLIP-L | 11 | +0.0959 [+0.0343, +0.1567], p 0.0024 | 0.5536 [+0.4443, +0.6289] (unstable) |
| CLIP-L | 12 | +0.1192 [+0.0383, +0.1985], p 0.0018 | 0.5351 [+0.4082, +0.6492] (unstable) |
| CLIP-L | 13 | +0.0569 [-0.0152, +0.1302], p 0.1142 | -0.2903 [-21.0026, +2.2586] (not reportable (denominator interval includes 0)) |
| CLIP-L | 14 | +0.0986 [+0.0336, +0.1648], p 0.007 | 0.1863 [-1.3333, +0.4099] (unstable) |
| CLIP-L | 15 | +0.1075 [+0.0442, +0.1694], p 0.0046 | 0.3669 [-0.0280, +0.5642] (unstable) |
| CLIP-L | 16 | +0.1278 [+0.0678, +0.1874], p 0.0008 | 0.3682 [+0.1402, +0.6059] |
| CLIP-L | 17 | +0.1371 [+0.0701, +0.2044], p 0.0012 | 0.3744 [+0.1619, +0.6723] |
| CLIP-L | 18 | +0.1175 [+0.0615, +0.1733], p 0.001 | 0.3963 [+0.1674, +0.7405] |
| CLIP-L | 19 | +0.1169 [+0.0662, +0.1673], p 0.0004 | 0.3773 [+0.1289, +0.7277] |
| CLIP-L | 20 | +0.1049 [+0.0592, +0.1516], p 0.0004 | 0.3390 [+0.0776, +0.7422] |
| CLIP-L | 21 | +0.0715 [+0.0230, +0.1199], p 0.0104 | 0.1176 [-0.3817, +0.8873] (unstable) |
| CLIP-L | 22 | +0.0859 [+0.0374, +0.1350], p 0.0044 | 0.2440 [-0.1681, +0.9385] (unstable) |
| CLIP-L | 23 | +0.0886 [+0.0389, +0.1389], p 0.0036 | 0.2629 [-0.0755, +0.9101] (unstable) |
| CLIP-L | 24 | +0.0785 [+0.0313, +0.1264], p 0.0038 | 0.1943 [-0.1232, +0.8132] (unstable) |
| SigLIP-L | 1 | -0.1070 [-0.1789, -0.0300], p 0.0006 | 0.9784 [+0.8052, +1.1751] (unstable) |
| SigLIP-L | 2 | -0.0539 [-0.0849, -0.0222], p 0.0012 | -0.2171 [-1.5957, +0.1490] (unstable) |
| SigLIP-L | 3 | -0.0241 [-0.0693, +0.0188], p 0.242 | 1.8306 [-2.7086, +109.6752] (not reportable (denominator interval includes 0)) |
| SigLIP-L | 4 | +0.0085 [-0.0333, +0.0497], p 0.7678 | -3.1684 [-4611.4175, -0.4108] (not reportable (denominator interval includes 0)) |
| SigLIP-L | 5 | +0.0273 [-0.0165, +0.0705], p 0.2726 | -0.2619 [-8.5941, +8.8699] (not reportable (denominator interval includes 0)) |
| SigLIP-L | 6 | +0.0567 [-0.0006, +0.1142], p 0.0446 | 0.1131 [-1.4279, +0.4804] (not reportable (denominator interval includes 0)) |
| SigLIP-L | 7 | +0.1133 [+0.0498, +0.1758], p 0.0016 | 0.3623 [+0.1739, +0.4949] |
| SigLIP-L | 8 | +0.0854 [+0.0296, +0.1405], p 0.0032 | 0.1264 [-0.3721, +0.3821] (unstable) |
| SigLIP-L | 9 | +0.0689 [+0.0237, +0.1142], p 0.0042 | 0.2431 [-0.1750, +0.4775] (unstable) |
| SigLIP-L | 10 | +0.0681 [+0.0266, +0.1094], p 0.0036 | 0.3678 [+0.0520, +0.5721] (unstable) |
| SigLIP-L | 11 | +0.0493 [+0.0134, +0.0848], p 0.0062 | -0.3271 [-1.5365, +0.2031] (unstable) |
| SigLIP-L | 12 | +0.0616 [+0.0328, +0.0902], p 0.0006 | 0.0727 [-0.4162, +0.3167] |
| SigLIP-L | 13 | +0.0674 [+0.0350, +0.1009], p 0.001 | -0.2043 [-0.9100, +0.0500] |
| SigLIP-L | 14 | +0.0792 [+0.0427, +0.1174], p 0.0002 | -0.2026 [-0.6416, +0.0322] |
| SigLIP-L | 15 | +0.1413 [+0.0999, +0.1823], p 0.0002 | 0.3300 [+0.1480, +0.5184] |
| SigLIP-L | 16 | +0.1181 [+0.0803, +0.1561], p 0.0002 | 0.1085 [-0.0854, +0.3503] |
| SigLIP-L | 17 | +0.1210 [+0.0839, +0.1575], p 0.0002 | 0.0858 [-0.1026, +0.2979] |
| SigLIP-L | 18 | +0.1192 [+0.0879, +0.1503], p 0.0002 | 0.0265 [-0.1581, +0.2492] |
| SigLIP-L | 19 | +0.1047 [+0.0763, +0.1331], p 0.0002 | -0.0523 [-0.2455, +0.1806] |
| SigLIP-L | 20 | +0.1025 [+0.0708, +0.1346], p 0.0002 | -0.1602 [-0.3408, +0.1033] |
| SigLIP-L | 21 | +0.0851 [+0.0568, +0.1139], p 0.0004 | -0.3170 [-0.6258, +0.0863] |
| SigLIP-L | 22 | +0.0827 [+0.0613, +0.1047], p 0.0004 | -0.2361 [-0.6740, +0.1311] |
| SigLIP-L | 23 | +0.0892 [+0.0519, +0.1267], p 0.0002 | -0.5524 [-0.8252, -0.2611] |
| SigLIP-L | 24 | +0.0938 [+0.0550, +0.1318], p 0.0002 | -0.7207 [-1.0336, -0.4247] |

## 3. (v) Comparison table: token verdict under each null, saliency E, S20

| row | token (i) | token (ii) | E [WCU], p | E [percentile] | S20 [BCa] | agree (i) WCU / pct | agree (ii) WCU / pct |
|---|---|---|---|---|---|---|---|
| dinov2_L12 (swap R4) | fail | pass | -0.0975 [-0.1301, -0.0642], p 0.0004 | [-0.1260, -0.0703] | 0.4405 [+0.3005, +0.5706] | True / True | False / False |
| dinov2_L6 (swap R4) | pass | pass | +0.0824 [+0.0592, +0.1053], p 0.0002 | [+0.0624, +0.1023] | 0.3496 [+0.2520, +0.4936] | True / True | True / True |
| siglip_L24 (swap R4) | pass | fail | +0.1447 [+0.1186, +0.1703], p 0.0004 | [+0.1221, +0.1702] | 0.1626 [+0.0310, +0.2924] | True / True | False / False |
| dinov2_L2 (sweep) | fail | fail | -2.88e-05 [-0.0195, +0.0201], p 0.98 | [-0.0167, +0.0164] | -180.3071 [-2789.0567, -2739.6728] (not reportable (denominator interval includes 0)) | True / True | True / True |
| dinov2_L4 (sweep) | pass | fail | +0.0507 [+0.0425, +0.0592], p 0.0006 | [+0.0426, +0.0591] | 0.2612 [+0.1656, +0.3540] | True / True | False / False |
| dinov2_L6 (sweep) | pass | pass | +0.0824 [+0.0592, +0.1053], p 0.0002 | [+0.0624, +0.1023] | 0.3496 [+0.2520, +0.4936] | True / True | True / True |
| dinov2_L8 (sweep) | fail | fail | +0.0347 [+0.0208, +0.0480], p 0.0002 | [+0.0234, +0.0465] | -0.5661 [-0.9478, -0.3670] | False / False | False / False |
| dinov2_L10 (sweep) | fail | pass | -0.0285 [-0.0468, -0.0098], p 0.0044 | [-0.0455, -0.0134] | 0.7593 [+0.1802, +1.9935] (unstable) | True / True | False / False |
| dinov2_L12 (sweep) | fail | pass | -0.0975 [-0.1301, -0.0642], p 0.0004 | [-0.1260, -0.0703] | 0.4405 [+0.3005, +0.5706] | True / True | False / False |
| clip_L4 (sweep) | fail | fail | -0.0003 [-0.0206, +0.0204], p 0.9804 | [-0.0173, +0.0167] | -16.7576 [-2414.0341, -27.4462] (not reportable (denominator interval includes 0)) | True / True | True / True |
| clip_L8 (sweep) | pass | fail | +0.0409 [+0.0157, +0.0656], p 0.0004 | [+0.0212, +0.0611] | 0.7754 [+0.4285, +1.6880] (unstable) | True / True | False / False |
| clip_L12 (sweep) | pass | fail | +0.0605 [+0.0461, +0.0753], p 0.0006 | [+0.0480, +0.0732] | 0.1038 [-0.0385, +0.2780] | True / True | False / False |
| clip_L16 (sweep) | pass | fail | +0.0176 [+0.0036, +0.0317], p 0.0158 | [+0.0052, +0.0299] | -0.1555 [-1.3598, +0.2131] (unstable) | True / True | False / False |
| clip_L20 (sweep) | pass | fail | -0.0101 [-0.0253, +0.0048], p 0.173 | [-0.0243, +0.0026] | 1.1876 [-3.8036, +4.5402] (not reportable (denominator interval includes 0)) | False / False | True / True |
| clip_L24 (sweep) | pass | fail | +0.0137 [-0.0176, +0.0444], p 0.3792 | [-0.0120, +0.0423] | 0.7605 [-6.6130, +4.0394] (not reportable (denominator interval includes 0)) | False / False | True / True |
| siglip_L4 (sweep) | fail | fail | +0.0198 [+0.0093, +0.0306], p 0.0054 | [+0.0093, +0.0289] | 0.5354 [+0.1116, +1.1164] (unstable) | False / False | False / False |
| siglip_L8 (sweep) | fail | fail | +0.0551 [+0.0404, +0.0699], p 0.0006 | [+0.0417, +0.0685] | 0.4496 [+0.2269, +0.6562] | False / False | False / False |
| siglip_L12 (sweep) | fail | fail | -0.0206 [-0.0348, -0.0066], p 0.0078 | [-0.0331, -0.0083] | 0.5822 [+0.1534, +1.6698] (unstable) | True / True | True / True |
| siglip_L16 (sweep) | pass | fail | +0.0019 [-0.0131, +0.0169], p 0.7772 | [-0.0122, +0.0156] | -0.9920 [-167.7423, +1.3538] (not reportable (denominator interval includes 0)) | False / False | True / True |
| siglip_L20 (sweep) | pass | fail | +0.0473 [+0.0157, +0.0784], p 0.0054 | [+0.0200, +0.0739] | 0.5895 [+0.1919, +1.5234] (unstable) | True / True | False / False |
| siglip_L24 (sweep) | pass | fail | +0.1447 [+0.1186, +0.1703], p 0.0004 | [+0.1221, +0.1702] | 0.1626 [+0.0310, +0.2924] | True / True | False / False |
| iie_shapenetc (swap R4, CNN) | per layer: passes at no layer (published CNN analysis) | not built | +0.0820 [+0.0428, +0.1216], p 0.0006 | [+0.0504, +0.1171] | 0.9031 [+0.7409, +1.1237] | n/a | n/a |

Token verdicts: (i) unrounded bounds; the published-rounding verdict differs only where listed in Section 0. Agreement: figure-specific agrees with a saliency E interval above 0, not figure-specific with one that is not.
