# Rule 4.5 continuity for the primary estimates of P1, P2, P4 and P5 (`report_continuity.py`)

**REPORTING ONLY.** Nothing here is decision-bearing. Each row is a primary estimate of `out/primary_P1_P2_P3_P4.json` or `out/primary_P5.json` (reproduced, gates G1 and G2), with the rule's interval and p as stored there and the published two-stage percentile interval (10,000 resamples, RandomState(2026); rule 4.5, continuity). The earlier pipeline never computed these estimates; the interval is its method applied to them (Section 1), and for the 18 P4 ratio differences a composition of its method, not a published computation (Section 7). All values are unrounded in `continuity_intervals.json`; 4 decimals here.

- Code SHA-256: `explore/report_continuity.py` 2c3eb599e546daa6bf43b1556a9b41828f778f28272a6c7319618edbe7c2c03e; `explore/excommon.py` d168232289be46a9e883385aeee279165e1c7924879c07c5f03204b3b38a397a; `explore/explore_saved.py` 95236f1daacac26606f90294226f2394e1e73c9c6bacd4bb19d31148f01dfb7a; `explore/report_p6_cells.py` 5db1fc3bc45f4e4820f28d16eb070ef319979760167c70384b7dbaabaaec7665.
- Rule SHA-256 (`REANALYSIS_RULE.md`): b03f397b6b1808e7055f0d6db8f5685c0e6a2850261eeb3d8525dddbeca73a38. Public commit (`PUBLIC_PUSH.txt`): f67c7df98c27136be8ff3db4eea57d1973579602.
- Run 2026-11-20T09:36:58Z to 2026-11-20T09:37:00Z (2.0 s compute). require_public: 33 hashed files and every stimulus verified.

## 0. Reproduction gates (all passed; any failure would have stopped the run before writing)

- G0 reused helpers unchanged (report_p6_cells.py, explore_saved.py and excommon.py equal the hashes in P6_reporting.json): passed.
- G1 every point estimate equals the stored primary value: 164 checks, 0 failed, max |difference| 0.0, tolerance 1e-12: passed.
- G2 families reproduced in full: `out/primary_P1_P2_P3_P4.json` (2026-11-18T14:02:44Z; P1, P2, P4) 1502 leaves, 1322 numeric, max |difference| 0.0; `out/primary_P5.json` (2026-11-18T18:37:03Z; every key) 1354 leaves, 1219 numeric, max |difference| 0.0; tolerance 1e-12: passed.
- G3 published values at 4 decimals (v3/scores/results.json; v3/gpu/full/results_full.json, the 13 swap cells of P4): P1_L20 68 of 68 (largest |difference| 4.98e-05); P2_square_Delta20 68 of 68 (largest |difference| 4.93e-05); E_swap 52 of 52 (largest |difference| 4.71e-05); P4_ratio_U20 39 of 39 (largest |difference| 4.99e-05); P4_ratio_A20 39 of 39 (largest |difference| 4.96e-05): passed.

## 1. Method

- Resample (the published one): clusters with replacement, then positions with replacement within each drawn cluster slot, 10,000 resamples, RandomState(2026). P1, P4 and P5: the pooled design (12 clusters of 5 configurations); P2: the square design (6 clusters of 5).
- Linear quantities: percentile interval of the resampled means and the two-sided bootstrap p (`analyze_re.pct_summary`, as `v3/scores/analyze_alt.summ` and `v2/readout/analyze.boot`). A P1 or P2 model-minus-null difference is a paired difference on the same configurations, summarised as the earlier pipeline summarised its paired readout contrasts (`v3/gpu/analyze_full.py`); it equals the percentile interval of the resample-wise difference of the two bootstrap distributions that the earlier pipeline drew for the model and the null with the same indices. The P5 displays were never scored by the earlier pipeline; their intervals are its method applied to the new scores.
- Ratio: percentile interval of the ratio of resampled means, no p (`analyze_re.pct_ratio`). P4 ratio difference (a composition, Section 7): percentile interval of the resample-wise difference of the two ratios under the same indices (rule 5.4: "paired by configuration (the same resample for both readouts)"), which is the bootstrap distribution behind the rule's BCa interval without the BCa adjustment (exactly so when no resample is infinite; count in Section 5); no p.
- Ratios follow rule 4.3: `not reportable` (denominator WCU interval includes 0) is printed without a value; `unstable` = denominator |t| < 4. A p of 0 is printed `below resolution (0 outside every resample)`. `agree` = the two intervals agree on whether they exclude 0 and on which side.

## 2. Summary

124 primary estimates now have a continuity interval: 104 linear ones with the published method and its bootstrap p, 2 rigid share(s) s* with the published ratio method and no p, and the 18 P4 ratio differences by a composition of the published method, with no p (Section 7).

| family | estimates | continuity interval agrees with the rule interval on excluding 0 |
|---|---|---|
| P1 model minus null, L20 (WCU) | 30 | 30 of 30 |
| P2 model minus null, square Delta20 (WCU) | 30 | 30 of 30 |
| P4 readout minus R4 ratio differences (BCa) | 18 | 18 of 18 (same resamples, with and without the BCa adjustment, so not an independent comparison) |
| P5 at 20 degrees: linear estimands (WCU) and the rigid share s* (BCa) | 46 (44 linear, 2 s*) | 42 of 44 linear (s* is not tested against 0) |

Disagreements on excluding 0:

- P5/DeepGaze IIE/A_star_minus_null/corner_s25: +0.0164, published percentile [+0.0014, +0.0328], rule WCU [-0.0020, +0.0350].
- P5/DeepGaze MSDB/A_star_minus_null/edge_s25: +0.0136, published percentile [+0.0007, +0.0259], rule WCU [-0.0009, +0.0282].

## 3. P1: model minus null, L20 = r(illusory) - r(mis20), 60 configurations, 12 clusters

| model | null | estimate | 95% WCU (rule) | WCR p | published percentile | bootstrap p | agree |
|---|---|---|---|---|---|---|---|
| DeepGaze IIE | corner sigma 8 | +0.0658 | [+0.0417, +0.0897] | 0.0004 | [+0.0459, +0.0866] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | corner sigma 16 | +0.0609 | [+0.0355, +0.0857] | 0.0004 | [+0.0400, +0.0825] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | corner sigma 25 | +0.0572 | [+0.0305, +0.0829] | 0.0006 | [+0.0355, +0.0797] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | corner sigma 40 | +0.0536 | [+0.0271, +0.0793] | 0.0008 | [+0.0321, +0.0761] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | corner sigma 60 | +0.0509 | [+0.0262, +0.0749] | 0.0008 | [+0.0308, +0.0719] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | edge sigma 8 | +0.0759 | [+0.0509, +0.1004] | 0.0004 | [+0.0549, +0.0968] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | edge sigma 16 | +0.0679 | [+0.0425, +0.0930] | 0.0004 | [+0.0467, +0.0891] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | edge sigma 25 | +0.0624 | [+0.0370, +0.0875] | 0.0004 | [+0.0413, +0.0836] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | edge sigma 40 | +0.0572 | [+0.0324, +0.0816] | 0.0004 | [+0.0367, +0.0781] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | edge sigma 60 | +0.0537 | [+0.0298, +0.0771] | 0.0004 | [+0.0339, +0.0738] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | ink sigma 8 | +0.0566 | [+0.0346, +0.0782] | 0.0004 | [+0.0385, +0.0752] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | ink sigma 16 | +0.0585 | [+0.0359, +0.0805] | 0.0004 | [+0.0400, +0.0775] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | ink sigma 25 | +0.0585 | [+0.0353, +0.0812] | 0.0004 | [+0.0393, +0.0781] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | ink sigma 40 | +0.0570 | [+0.0332, +0.0804] | 0.0004 | [+0.0372, +0.0771] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | ink sigma 60 | +0.0546 | [+0.0309, +0.0778] | 0.0004 | [+0.0348, +0.0746] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | corner sigma 8 | +0.0456 | [+0.0251, +0.0657] | 0.0006 | [+0.0293, +0.0631] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | corner sigma 16 | +0.0406 | [+0.0196, +0.0611] | 0.0006 | [+0.0239, +0.0585] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | corner sigma 25 | +0.0369 | [+0.0150, +0.0585] | 0.0008 | [+0.0195, +0.0555] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | corner sigma 40 | +0.0334 | [+0.0113, +0.0552] | 0.0022 | [+0.0159, +0.0520] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | corner sigma 60 | +0.0306 | [+0.0103, +0.0508] | 0.003 | [+0.0143, +0.0480] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | edge sigma 8 | +0.0556 | [+0.0366, +0.0741] | 0.0006 | [+0.0401, +0.0718] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | edge sigma 16 | +0.0476 | [+0.0284, +0.0664] | 0.0006 | [+0.0320, +0.0640] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | edge sigma 25 | +0.0421 | [+0.0227, +0.0612] | 0.0006 | [+0.0264, +0.0586] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | edge sigma 40 | +0.0369 | [+0.0178, +0.0557] | 0.0006 | [+0.0213, +0.0532] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | edge sigma 60 | +0.0334 | [+0.0147, +0.0517] | 0.0008 | [+0.0183, +0.0493] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | ink sigma 8 | +0.0364 | [+0.0187, +0.0538] | 0.0006 | [+0.0221, +0.0514] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | ink sigma 16 | +0.0383 | [+0.0204, +0.0559] | 0.0006 | [+0.0237, +0.0535] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | ink sigma 25 | +0.0383 | [+0.0199, +0.0561] | 0.0006 | [+0.0234, +0.0538] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | ink sigma 40 | +0.0368 | [+0.0182, +0.0548] | 0.0006 | [+0.0217, +0.0527] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | ink sigma 60 | +0.0343 | [+0.0159, +0.0523] | 0.0006 | [+0.0194, +0.0500] | below resolution (0 outside every resample) | yes |

## 4. P2: model minus null, square Delta20 = r(alt20) - r(mis20), 30 configurations, 6 clusters

| model | null | estimate | 95% WCU (rule) | WCR p | published percentile | bootstrap p | agree |
|---|---|---|---|---|---|---|---|
| DeepGaze IIE | corner sigma 8 | +0.0239 | [+0.0051, +0.0427] | 0.0114 | [+0.0124, +0.0364] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | corner sigma 16 | +0.0239 | [+0.0053, +0.0426] | 0.01145 | [+0.0124, +0.0364] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | corner sigma 25 | +0.0240 | [+0.0054, +0.0425] | 0.0114 | [+0.0125, +0.0364] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | corner sigma 40 | +0.0240 | [+0.0055, +0.0425] | 0.0114 | [+0.0125, +0.0365] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | corner sigma 60 | +0.0241 | [+0.0056, +0.0425] | 0.01136 | [+0.0125, +0.0366] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | edge sigma 8 | +0.0242 | [+0.0058, +0.0425] | 0.01132 | [+0.0126, +0.0366] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | edge sigma 16 | +0.0242 | [+0.0058, +0.0425] | 0.01132 | [+0.0126, +0.0366] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | edge sigma 25 | +0.0242 | [+0.0058, +0.0425] | 0.01132 | [+0.0126, +0.0366] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | edge sigma 40 | +0.0242 | [+0.0058, +0.0425] | 0.01132 | [+0.0126, +0.0366] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | edge sigma 60 | +0.0242 | [+0.0058, +0.0425] | 0.01132 | [+0.0126, +0.0366] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | ink sigma 8 | +0.0242 | [+0.0058, +0.0425] | 0.01132 | [+0.0126, +0.0366] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | ink sigma 16 | +0.0242 | [+0.0058, +0.0425] | 0.01132 | [+0.0126, +0.0366] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | ink sigma 25 | +0.0242 | [+0.0058, +0.0425] | 0.01132 | [+0.0126, +0.0366] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | ink sigma 40 | +0.0242 | [+0.0058, +0.0425] | 0.01132 | [+0.0126, +0.0366] | below resolution (0 outside every resample) | yes |
| DeepGaze IIE | ink sigma 60 | +0.0242 | [+0.0058, +0.0425] | 0.01132 | [+0.0126, +0.0366] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | corner sigma 8 | +0.0205 | [+0.0104, +0.0306] | 0.004973 | [+0.0131, +0.0276] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | corner sigma 16 | +0.0205 | [+0.0104, +0.0307] | 0.005058 | [+0.0131, +0.0276] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | corner sigma 25 | +0.0206 | [+0.0104, +0.0307] | 0.00493 | [+0.0131, +0.0277] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | corner sigma 40 | +0.0206 | [+0.0104, +0.0308] | 0.00493 | [+0.0131, +0.0278] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | corner sigma 60 | +0.0207 | [+0.0104, +0.0310] | 0.004844 | [+0.0131, +0.0279] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | edge sigma 8 | +0.0208 | [+0.0104, +0.0311] | 0.004887 | [+0.0132, +0.0280] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | edge sigma 16 | +0.0208 | [+0.0104, +0.0311] | 0.004887 | [+0.0132, +0.0280] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | edge sigma 25 | +0.0208 | [+0.0104, +0.0311] | 0.004887 | [+0.0132, +0.0280] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | edge sigma 40 | +0.0208 | [+0.0104, +0.0311] | 0.004887 | [+0.0132, +0.0280] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | edge sigma 60 | +0.0208 | [+0.0104, +0.0311] | 0.004887 | [+0.0132, +0.0280] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | ink sigma 8 | +0.0208 | [+0.0104, +0.0311] | 0.004887 | [+0.0132, +0.0280] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | ink sigma 16 | +0.0208 | [+0.0104, +0.0311] | 0.004887 | [+0.0132, +0.0280] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | ink sigma 25 | +0.0208 | [+0.0104, +0.0311] | 0.004887 | [+0.0132, +0.0280] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | ink sigma 40 | +0.0208 | [+0.0104, +0.0311] | 0.004887 | [+0.0132, +0.0280] | below resolution (0 outside every resample) | yes |
| DeepGaze MSDB | ink sigma 60 | +0.0208 | [+0.0104, +0.0311] | 0.004887 | [+0.0132, +0.0280] | below resolution (0 outside every resample) | yes |

## 5. P4: readout minus R4 ratio differences, 60 configurations, 12 clusters

Difference: the rule's BCa interval, BCa p and Holm p (stored), and the composed published-method interval (percentile interval of the resample-wise difference; a composition, not a published computation; no published-method p, Section 7). Both ratios: the published percentile interval (values exist in `v3/gpu/full/results_full.json`; reproduced, G3), printed next to the difference they summarise (rule 4.3).

| # | feature set | readout | ratio | difference | 95% BCa (rule) | BCa p | Holm p | counted | published percentile (composed) | agree | readout ratio [published percentile] | R4 ratio [published percentile] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | DINOv2-B block 12 | R1n | U20 | -0.0604 | [-0.1363, +0.0508] | 0.2552 | 1 | no | [-0.1374, +0.0473] | yes | +0.6587 [+0.6024, +0.7112] | +0.7191 [+0.6165, +0.8066] |
| 2 | DINOv2-B block 12 | R1n | A20 | +0.0215 | [-0.0211, +0.0725] | 0.338 | 1 | no | [-0.0219, +0.0714] | yes | +0.8788 [+0.8224, +0.9453] | +0.8574 [+0.7931, +0.9263] |
| 3 | DINOv2-B block 12 | R1 | U20 | -0.1499 | [-0.2364, -0.0172] | 0.03226 | 0.2258 | no | [-0.2417, -0.0259] | yes | +0.5691 [+0.4706, +0.6710] | +0.7191 [+0.6165, +0.8066] |
| 4 | DINOv2-B block 12 | R1 | A20 | -0.0725 | [-0.1257, +0.0246] | 0.1194 | 0.7162 | no | [-0.1299, +0.0144] | yes | +0.7849 [+0.6959, +0.9048] | +0.8574 [+0.7931, +0.9263] |
| 5 | DINOv2-B block 12 | R2 | U20 | -0.0252 | [-0.0585, +0.0206] | 0.2482 | 1 | no | [-0.0586, +0.0202] | yes | +0.6939 [+0.6192, +0.7593] | +0.7191 [+0.6165, +0.8066] |
| 6 | DINOv2-B block 12 | R2 | A20 | -0.0151 | [-0.0521, +0.0134] | 0.339 | 1 | no | [-0.0546, +0.0121] | yes | +0.8423 [+0.7863, +0.8897] | +0.8574 [+0.7931, +0.9263] |
| 7 | DINOv2-B block 12 | R3 | U20 | -0.1353 | [-0.2436, -0.0458] | 0.0007534 | 0.01204 | yes | [-0.2493, -0.0482] | yes | +0.5838 [+0.4382, +0.6714] | +0.7191 [+0.6165, +0.8066] |
| 8 | DINOv2-B block 12 | R3 | A20 | -0.1717 | [-0.2909, -0.0954] | below resolution (0 outside every resample; D4) | 0.001051 | yes | [-0.2836, -0.0917] | yes | +0.6857 [+0.5779, +0.7824] | +0.8574 [+0.7931, +0.9263] |
| 9 | DINOv2-B block 6 | R2 | U20 | +0.0824 | [+0.0286, +0.1577] | 0.0007523 | 0.01204 | yes | [+0.0257, +0.1540] | yes | +0.6724 [+0.5363, +0.7885] | +0.5900 [+0.4254, +0.7259] |
| 10 | DINOv2-B block 6 | R2 | A20 | +0.1641 | [+0.1059, +0.2370] | below resolution (0 outside every resample; D4) | 0.0006792 | yes | [+0.1039, +0.2346] | yes | +1.1136 [+1.0348, +1.2002] | +0.9495 [+0.8611, +1.0242] |
| 11 | DINOv2-B block 6 | R3 | U20 | +0.0643 | [+0.0144, +0.1080] | 0.01503 | 0.1352 | no | [+0.0107, +0.1052] | yes | +0.6543 [+0.4593, +0.8042] | +0.5900 [+0.4254, +0.7259] |
| 12 | DINOv2-B block 6 | R3 | A20 | +0.0670 | [+0.0223, +0.1311] | 0.002157 | 0.02373 | yes | [+0.0164, +0.1208] | yes | +1.0166 [+0.8972, +1.1267] | +0.9495 [+0.8611, +1.0242] |
| 13 | SigLIP-L block 24 | R2 | U20 | +0.0094 | [-0.0309, +0.0454] | 0.6026 | 1 | no | [-0.0310, +0.0452] | yes | +0.8219 [+0.6693, +0.9633] | +0.8125 [+0.6715, +0.9485] |
| 14 | SigLIP-L block 24 | R2 | A20 | +0.0492 | [+0.0135, +0.0775] | 0.01015 | 0.1015 | no | [+0.0112, +0.0761] | yes | +0.6793 [+0.5355, +0.8266] | +0.6301 [+0.4799, +0.8012] |
| 15 | ShapeNet-C ResNet-50 | R2 | U20 | -0.1278 | [-0.2381, -0.0604] | below resolution (0 outside every resample; D4) | 0.0128 | yes | [-0.2544, -0.0648] | yes | -0.0309 [-0.3113, +0.1565] | +0.0969 [-0.1117, +0.2668] |
| 16 | ShapeNet-C ResNet-50 | R2 | A20 | -0.0543 | [-0.1065, -0.0174] | 0.001643 | 0.01971 | yes | [-0.1152, -0.0209] | yes | +0.4360 [+0.2420, +0.5989] | +0.4903 [+0.3192, +0.6580] |
| 17 | ShapeNet-C ResNet-50 | R3 | U20 | +0.1526 | [+0.0256, +0.2711] | 0.02365 | 0.1892 | no | [+0.0225, +0.2696] | yes | +0.2495 [+0.0767, +0.3816] | +0.0969 [-0.1117, +0.2668] |
| 18 | ShapeNet-C ResNet-50 | R3 | A20 | +0.1761 | [+0.0870, +0.2645] | 0.001072 | 0.01394 | yes | [+0.0725, +0.2527] | yes | +0.6664 [+0.4904, +0.7902] | +0.4903 [+0.3192, +0.6580] |

A BCa p printed `below resolution` is the stored p set by the clip of `analyze_re.bca_p`: no resample of the difference lies on the other side of 0 (DEVIATIONS.md D4). Holm p and the counted flag are those of the stored P4.

Non-finite resamples over the 18 differences: 0 infinite (a resampled denominator mean exactly 0; kept by `np.nanpercentile`, as in `pct_ratio`, and dropped by `analyze_re.bca`) and 0 nan. With none infinite, each composed interval is exactly the 2.5 and 97.5 percentiles of the resamples behind the rule's BCa interval.

## 6. P5 at 20 degrees, per model, 60 configurations, 12 clusters

### DeepGaze IIE

| quantity | estimate | 95% WCU (rule) | WCR p | published percentile | bootstrap p | agree |
|---|---|---|---|---|---|---|
| L_uni(20) = E(0) - E'_mis(20), loss under uniform rotation | +0.0449 | [+0.0233, +0.0664] | 0.0014 | [+0.0268, +0.0632] | below resolution (0 outside every resample) | yes |
| L_rig(20) = E(0) - E_rig(20), loss under rigid rotation | +0.0284 | [+0.0118, +0.0452] | 0.001 | [+0.0151, +0.0440] | below resolution (0 outside every resample) | yes |
| A*(20) = E_rig(20) - E'_mis(20), alignment effect at matched orientation | +0.0166 | [+0.0004, +0.0325] | 0.0316 | [+0.0039, +0.0313] | 0.0052 | yes |
| C*(20) = L_uni(20) - 2 L_rig(20), half criterion | -0.0118 | [-0.0368, +0.0139] | 0.3208 | [-0.0325, +0.0112] | 0.2978 | yes |
| secondary: drafted alignment loss (RI - M) - (RC - C) | +0.0218 | [+0.0052, +0.0380] | 0.0094 | [+0.0085, +0.0367] | below resolution (0 outside every resample) | yes |
| secondary: control-free RI - M | -0.0119 | [-0.0309, +0.0076] | 0.1926 | [-0.0272, +0.0056] | 0.1698 | yes |
| secondary: control's change under rigid rotation RC - C | -0.0337 | [-0.0461, -0.0211] | 0.0006 | [-0.0441, -0.0235] | below resolution (0 outside every resample) | yes |

Rigid share s*(20) = L_rig(20) / L_uni(20): +0.6313, 95% BCa (rule) [+0.3272, +0.8500], published percentile [+0.3812, +0.8848], no published-method p; next to L_uni(20) +0.0449 [+0.0233, +0.0664] and L_rig(20) +0.0284 [+0.0118, +0.0452] (WCU).

| null | A* model minus null | 95% WCU (rule) | WCR p | published percentile | bootstrap p | agree |
|---|---|---|---|---|---|---|
| corner sigma 8 | +0.0321 | [+0.0170, +0.0471] | 0.0006 | [+0.0199, +0.0458] | below resolution (0 outside every resample) | yes |
| corner sigma 16 | +0.0229 | [+0.0062, +0.0395] | 0.0086 | [+0.0094, +0.0379] | 0.0002 | yes |
| corner sigma 25 | +0.0164 | [-0.0020, +0.0350] | 0.0744 | [+0.0014, +0.0328] | 0.0314 | no |
| corner sigma 40 | +0.0108 | [-0.0079, +0.0294] | 0.23 | [-0.0041, +0.0274] | 0.1758 | yes |
| corner sigma 60 | +0.0090 | [-0.0078, +0.0254] | 0.2764 | [-0.0042, +0.0242] | 0.2164 | yes |
| edge sigma 8 | +0.0421 | [+0.0238, +0.0602] | 0.0004 | [+0.0276, +0.0586] | below resolution (0 outside every resample) | yes |
| edge sigma 16 | +0.0344 | [+0.0155, +0.0530] | 0.0006 | [+0.0194, +0.0514] | below resolution (0 outside every resample) | yes |
| edge sigma 25 | +0.0288 | [+0.0099, +0.0472] | 0.0038 | [+0.0138, +0.0457] | below resolution (0 outside every resample) | yes |
| edge sigma 40 | +0.0235 | [+0.0056, +0.0411] | 0.0088 | [+0.0093, +0.0395] | below resolution (0 outside every resample) | yes |
| edge sigma 60 | +0.0200 | [+0.0032, +0.0365] | 0.014 | [+0.0067, +0.0351] | 0.0004 | yes |
| ink sigma 8 | +0.0226 | [+0.0071, +0.0379] | 0.0036 | [+0.0105, +0.0368] | below resolution (0 outside every resample) | yes |
| ink sigma 16 | +0.0245 | [+0.0087, +0.0400] | 0.0026 | [+0.0122, +0.0389] | below resolution (0 outside every resample) | yes |
| ink sigma 25 | +0.0249 | [+0.0087, +0.0408] | 0.0026 | [+0.0123, +0.0397] | below resolution (0 outside every resample) | yes |
| ink sigma 40 | +0.0241 | [+0.0074, +0.0404] | 0.0052 | [+0.0110, +0.0390] | below resolution (0 outside every resample) | yes |
| ink sigma 60 | +0.0219 | [+0.0053, +0.0385] | 0.009 | [+0.0088, +0.0370] | below resolution (0 outside every resample) | yes |

### DeepGaze MSDB

| quantity | estimate | 95% WCU (rule) | WCR p | published percentile | bootstrap p | agree |
|---|---|---|---|---|---|---|
| L_uni(20) = E(0) - E'_mis(20), loss under uniform rotation | +0.0063 | [-0.0043, +0.0167] | 0.2236 | [-0.0030, +0.0151] | 0.178 | yes |
| L_rig(20) = E(0) - E_rig(20), loss under rigid rotation | +0.0049 | [-0.0130, +0.0227] | 0.5608 | [-0.0108, +0.0194] | 0.5398 | yes |
| A*(20) = E_rig(20) - E'_mis(20), alignment effect at matched orientation | +0.0014 | [-0.0137, +0.0167] | 0.8382 | [-0.0125, +0.0146] | 0.8116 | yes |
| C*(20) = L_uni(20) - 2 L_rig(20), half criterion | -0.0034 | [-0.0350, +0.0286] | 0.8286 | [-0.0301, +0.0245] | 0.811 | yes |
| secondary: drafted alignment loss (RI - M) - (RC - C) | +0.0250 | [+0.0041, +0.0461] | 0.025 | [+0.0070, +0.0435] | 0.0072 | yes |
| secondary: control-free RI - M | -0.0024 | [-0.0106, +0.0059] | 0.5582 | [-0.0103, +0.0050] | 0.5544 | yes |
| secondary: control's change under rigid rotation RC - C | -0.0275 | [-0.0423, -0.0124] | 0.0008 | [-0.0407, -0.0151] | below resolution (0 outside every resample) | yes |

Rigid share s*(20) = L_rig(20) / L_uni(20): not reportable (the WCU interval of L_uni includes 0; rule 4.3); no value or interval is printed.

| null | A* model minus null | 95% WCU (rule) | WCR p | published percentile | bootstrap p | agree |
|---|---|---|---|---|---|---|
| corner sigma 8 | +0.0170 | [+0.0010, +0.0330] | 0.0464 | [+0.0020, +0.0307] | 0.0222 | yes |
| corner sigma 16 | +0.0078 | [-0.0078, +0.0235] | 0.308 | [-0.0066, +0.0212] | 0.2784 | yes |
| corner sigma 25 | +0.0013 | [-0.0150, +0.0175] | 0.8642 | [-0.0132, +0.0149] | 0.8356 | yes |
| corner sigma 40 | -0.0043 | [-0.0204, +0.0118] | 0.5918 | [-0.0184, +0.0092] | 0.5592 | yes |
| corner sigma 60 | -0.0061 | [-0.0215, +0.0092] | 0.444 | [-0.0199, +0.0070] | 0.3762 | yes |
| edge sigma 8 | +0.0269 | [+0.0120, +0.0419] | 0.0042 | [+0.0137, +0.0397] | 0.0006 | yes |
| edge sigma 16 | +0.0192 | [+0.0046, +0.0340] | 0.0194 | [+0.0062, +0.0316] | 0.005 | yes |
| edge sigma 25 | +0.0136 | [-0.0009, +0.0282] | 0.065 | [+0.0007, +0.0259] | 0.0376 | no |
| edge sigma 40 | +0.0083 | [-0.0064, +0.0231] | 0.2426 | [-0.0049, +0.0209] | 0.2182 | yes |
| edge sigma 60 | +0.0049 | [-0.0102, +0.0202] | 0.5076 | [-0.0089, +0.0179] | 0.4636 | yes |
| ink sigma 8 | +0.0075 | [-0.0079, +0.0231] | 0.3172 | [-0.0067, +0.0209] | 0.2886 | yes |
| ink sigma 16 | +0.0094 | [-0.0057, +0.0247] | 0.211 | [-0.0046, +0.0226] | 0.1848 | yes |
| ink sigma 25 | +0.0098 | [-0.0051, +0.0249] | 0.188 | [-0.0040, +0.0227] | 0.165 | yes |
| ink sigma 40 | +0.0089 | [-0.0060, +0.0241] | 0.2246 | [-0.0049, +0.0219] | 0.2 | yes |
| ink sigma 60 | +0.0068 | [-0.0085, +0.0222] | 0.3556 | [-0.0071, +0.0199] | 0.3238 | yes |

## 7. No published-method analogue, and compositions (stated, not invented)

- No analogue: a p-value for every ratio-type primary estimate: the P4 ratio differences (18), both ratios of each P4 contrast (36, 26 distinct ratios) and the rigid share s* (2): interval reported; p not computed. Why: the published ratio summaries (v3/scores/analyze_alt.ratio, v2/scores/analyze_v3.ratio, v3/gpu/analyze_full.py, analyze_re.pct_ratio) return a value and a percentile interval only; the earlier pipeline gave a bootstrap p only for means (summ). These rows carry the interval and no p; the rule's BCa p (P4) stays the only p of these quantities.
- Composition, not a published computation: the 18 P4 readout-minus-R4 ratio differences (U20 or A20): interval reported and labelled a composition; no p. How: percentile interval of the resample-wise difference of the two ratios of resampled means under the published two-stage indices (pct_ratio_diff); with no infinite resample (n_inf = 0, recorded per difference) equal to the 2.5 and 97.5 percentiles of the bootstrap distribution behind the rule's BCa interval, without the BCa adjustment. Why: the published pipeline's only ratio difference, diff(t) = ratio_alt(t) - ratio_uni(t), has one denominator and was resampled as a single ratio; a difference of two ratios with different denominators was never resampled, so no published value or published computation exists for these estimates.

## 8. Primary estimates whose continuity interval exists elsewhere

- P1 L_M(20) and P2 square Delta_M(20) of DeepGaze IIE and MSDB: published values in v3/scores/results.json (supplement table supp_tab_cont_p12); reproduced here at 4 decimals (gate G3), not new.
- both ratios of every P4 contrast (U20 or A20 of the readout and of R4): published values in v3/gpu/full/results_full.json (supplement table supp_tab_cont_p4, which already gives both published ratios of all 18 contrasts, 26 distinct ratios of 13 cells, so none is missing); reproduced here at 4 decimals (gate G3) and printed next to each difference, not new.
- P3: E of the 10 valid non-reference swap cells and of the 4 R4 references: published values in v3/gpu/full/results_full.json (supplement table tab_readout_report); not touched here.
- P6: E of the 18 new cells and of the 2 registers R4 references: results/explore/P6_reporting.json (report_p6_cells.py, D13); not touched here.
- the 15 nulls' own P5 quantities at 20 degrees (rule 5.5 secondary list: 'reported for every model and null'): results/explore/explore_rigid.json (R8, supplement table supp_tab_p5_own); null quantities, not estimates stored in the primary files, so outside this script (no stored value for gate G1); not computed here.

## 9. Computed but not printed (rule 4.3)

- DeepGaze MSDB rigid share s*(20) = L_rig(20) / L_uni(20) (`P5/DeepGaze MSDB/rigid_share`): not reportable (denominator interval includes 0); values in the JSON only.

