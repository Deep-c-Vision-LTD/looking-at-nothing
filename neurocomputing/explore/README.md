# Exploratory analysis scripts (REANALYSIS_RULE.md Section 6)

Written 18 November 2026, after the rule was fixed (`REANALYSIS_RULE.sha256`, 11:29:31Z) and made public (`PUBLIC_PUSH.txt`: Deep-c-Vision-LTD/looking-at-nothing f67c7df, 11:36:31Z). None of these scripts has been run on real data. Only two tests have been run, and neither read a score, map or weight file:

- `test_explore.py`, on synthetic inputs: 89 checks, 0 failed, about 6 s, 2 threads, no torch import. Output in `tests_explore_output.txt`.
- `test_torch_imports.py`, an import smoke test of the CPU torch stack: the stub, `gpu/recommon.py`, `sweep.py` reloaded per backbone, and untrained heads run on random features. 14 checks, 0 failed, about 6 s. Its pid was watched with `ioreg` every second until it exited, and it held no Metal client. Output in `tests_torch_imports_output.txt`.

Every script:

- quotes its rule section verbatim in its header and says how it implements it (the choices the rule leaves open are stated there);
- calls `require_public()` (which is `analyze_re.require_public`, rule Section 8.1) before it reads anything (checked by the tests);
- writes one JSON to `neuro/reanalysis/results/explore/`, labelled exploratory. Each JSON records the SHA-256 of the script and helper code, the rule hash and the public commit;
- reports values unrounded. Non-finite values are written as `null`, with a note.

The scripts are CPU only. Run one at a time, after the heavy job that is running now, as `nice -n 10 env OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY -B explore/<script>.py`, where `$PY` is the saliency project's Python 3.12. Torch is imported lazily and only by the scripts marked "torch" below. Those scripts:

- put the `_stub/ab_objective.py` stub first on the path, with `V3_DEVICE=cpu`;
- call `excommon.metal_guard()` after every model load; it queries `ioreg` and makes the process abort if its own pid holds an `AGXDeviceUserClient`.

Still check by hand within 30 s of starting one: `ioreg -l -w0 -r -c AGXDeviceUserClient | grep -o '"IOUserClientCreator" = "pid [0-9]*, [^"]*"' | grep "pid <PID>,"`.

## Scripts

CPU times are estimates for a machine with no other load, from `understand/data_inventory.md` Section 6. Under the GPU job's load, multiply by 1.5 to 2.

| script | rule item | inputs (read only) | output | torch | needs rigid-rotation scores | est. CPU |
|---|---|---|---|---|---|---|
| `a44_logmass.py` | 6.4 A4.4 log mass (inside, notch sector minus ink, ink); stim_v3 at phi 0, plus stim_rigid from the scoring-pass shares | stim_v3 PNGs and AOIs; saved DeepGaze maps (`v2/scores/maps`); MSI-Net PNGs; swap checkpoints with the float16 feature cache `v2/readout/cache/stim`; sweep `readout_L<L>.pt` (backbone pass); registers checkpoints (backbone pass); 15 nulls regenerated; published per-image scores (checks); `scores_rigid/` if present | `A4_4_logmass.json` | yes (sweep, swap, registers) | optional: the stim_rigid section says "not scored yet" until Step 3 has written `scores_rigid/`; rerun afterwards to fill it | 15 to 25 min |
| `a46_calibration.py` | 6.6 A4.6 calibration robustness (rank equalisation, T = 0.5 to 2; robust / sensitive label) | as A4.4, plus stim_alt PNGs and AOIs and `v3/scores/maps` (DeepGaze stim_alt); logits re-inferred (cache for swap on stim_v3, otherwise backbone passes) | `A4_6_calibration.json` | yes | no (stim_v3 and stim_alt only) | 2.5 to 3 h (dominated by 60,000 rank equalisations of 1024 x 768 maps) |
| `a11_scale.py` | 6.9 A11 effective spatial scale (NNLS corner + ink mixture, 11 sigmas, pooled R2) | saved DeepGaze IIE and MSDB maps on 540 stim_v3 Kanizsa and 360 stim_alt displays; display PNGs; published r_inside (check) | `A11_scale.json` | no | no | 10 to 20 min |
| `a12_kavsek.py` | 6.10 A12 Kavsek-style AOIs (E_K, uniform-map gain, E_K minus gain) | as A4.4 (map sources of `mapsrc.py`); `stim_rigid/aoi` (rotated P and D) | `A12_kavsek.json` | yes | optional (stim_rigid section, from the scoring-pass r_K) | 12 to 20 min |
| `a13_region.py` | 6.11 A13 region variant (E_N and its support-ratio slope) | as A4.4 | `A13_region.json` | yes | optional (stim_rigid section, from the scoring-pass r_N) | 12 to 20 min |
| `a3_reg_tokens_extract.py` | 6.8 (iv), extraction (RUN_PLAN T1): token components of `facebook/dinov2-with-registers-base`, all 13 hidden states, CLS and 4 registers dropped, 336 and 448 px, 100 configurations | stim_v3 PNGs and AOIs; HF cache | `neuro/reanalysis/A3_tokens/registers/dinov2reg_s{336,448}/<config>.npz` (as RUN_PLAN T1, about 50 MB), logs in `A3_tokens/registers/logs/` | yes (transformers only, no ab_objective) | no | about 0.2 h |
| `a3_reg_tokens_analyze.py` | 6.8 (iv) under both nulls: (i) the pre-specified rule of 15 October 2026, (ii) the near-exterior null, (iii) the paired excess with WCU and its retention under mis20; plain DINOv2-B verdicts shown alongside | the token components above; `v2/encoding/results.json`, `v2/verify/encoding_nulls.json` | `A3iv_registers_tokens.json` | no | no | about 1 min |
| `a15a_token_norms.py` | 6.12 A15(a) CLIP-L and SigLIP-L token norms (blocks 4 to 24 of each), with the Critic fix 18 condition | the receptive-field rule's 24 Kanizsa displays; the first 24 held-out MIT1003 images; HF cache | `A15a_token_norms.json` | yes | no | 5 to 30 min |
| `s13_block12_nulls.py` | 6.14 S13: U20(block 12 R4) minus U20(null k), BCa with the stability flag, and the sign-adjusted loss difference (WCU), for the 10 ink and edge nulls | `v3/gpu/full/scores` (dinov2_L12 R4), `v3/scores/raw_nulls.json`, `nn/registers/registers_results.json` (registers U20, shown alongside) | `S13_block12_vs_nulls.json` | no | no | under 1 min |
| `item7_corrected.py` | 6.13 secondary item 7 with the corrected sign (copy of the receptive-field rule's code). Checks first that the stored value is reproduced exactly | `v3/scores/raw_v3maps`, `v3/scores/raw`, `nn/results/raw_v3maps`, `nn/results/raw_alt`, `nn/results/erf/<bb>.json`, `nn/results/summary.json` | `item7_corrected.json` | no | no | 1 to 2 min |
| `a6_margin_curve.py` | 6.7 A6, first part: valid-set composition and sign agreement for margins 0 to 0.06 in steps of 0.005 | `v3/gpu/full/scores/mit_<tag>.json`, `stim_v3_<tag>.json` | `A6_margin_curve.json` | no | no | seconds |

Helpers:

- `excommon.py` has the shared statistics, bookkeeping, regions and provenance. It does not import torch. It wraps `analyze_re`'s wild cluster bootstrap, BCa and Fieller, and adds `slope_pct`, a copy of `analyze_alt.slope` without its rounding.
- `mapsrc.py` produces the per-display maps for every model group, with a pipeline check (see below).
- `test_explore.py` holds the unit tests.

## Pipeline check in the map-based scripts

The map-based scripts are A4.4, A4.6, A11, A12 and A13. They compute r_inside on the map of every display that has a published per-image value and compare the two. Tolerances:

- saved maps: 1e-6;
- local nulls: 1e-9;
- readouts re-inferred on the CPU: 1e-4, as in rule Section 3.2.

A unit that fails is reported as "not computable (pipeline check failed ...)" and its values are withheld. The tolerance, the number of comparisons and the largest difference are written to each JSON under `pipeline_check`.

## What is not here

- A6, second part: absolute CC, NSS and information gain with the centre bias off (RUN_PLAN S4).
- `explore_saved.py`: A4.1, A4.2, A4.3, A4.5, A10 and the P2 companions (RUN_PLAN S3).
- A3 (i) to (iii) and (v) for the three published ViTs (RUN_PLAN S5).
- `explore_rigid.py` (RUN_PLAN R8).
- The test-time registers code that A15(a) may require.

None of these were part of this task.

## Notes for the reader of the results

- A4.6 applies the rule's readout transform as written, exp(logit / T). For readouts, T = 1 is the odds map, not the published sigmoid map, so "T1" does not reproduce the published E for readouts. For DeepGaze, T = 1 is the published density. The robustness criterion compares every transform with the published E.
- Readouts are averaged over seeds and sizes as rule Section 2.2 prescribes. For log masses, the logs are averaged.
- Output locations: these scripts write to `results/explore/` (task instruction). RUN_PLAN.md lists `out/<name>.json` and `A15/norm_diag_clip_siglip.json` for some of them. The token components follow RUN_PLAN T1 (`A3_tokens/registers/`).
- `DEVIATIONS.md` (this folder) records the departure from the code that rule 6.13 asks to be logged for secondary item 7. That departure is the sign of the bootstrap rows as well as the point L20.
- `mapsrc.py` reloads `sweep.py` for each backbone. In `scoring/score_rigid.py` (fixed), a second `import sweep` in the same process returns the first backbone's module, so RUN_PLAN R3 stops after `sweep:dinov2`. The battery script in `results/scripts/` already expects this and refers to `DEVIATIONS.md` for it.

## Added later (18 November 2026, evening)

- `explore_rigid.py` (RUN_PLAN R8, rule Section 6 item 15) and its tests `test_explore_rigid.py` (synthetic inputs only; output `tests_explore_rigid_output.txt`, 66 checks, 0 failed). Written after R7 had been run and read; its choices are listed in its header and in `../DEVIATIONS.md` D7. Output: `../results/explore/explore_rigid.json` and `explore_rigid.md`. Run: `nice -n 10 env OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 $PY -B -I explore/explore_rigid.py` (numpy and scipy only; about 10 s).
- `a44_logmass.py`, `a12_kavsek.py` and `a13_region.py` were rerun unchanged after DeepGaze MSDB had been scored on `stim_rigid`, which filled MSDB's stim_rigid rows (`../DEVIATIONS.md` D6; first-run outputs in `../results/explore/_previous/2026-11-18_before_msdb_rigid/`; comparison in `../results/explore/msdb_rerun_comparison.json`).
- `ttr/` (18 November 2026, late evening): test-time registers (Jiang et al. 2025) on the A15(a)-flagged blocks, rule Section 6 item 12, as designed in `ttr/DESIGN.md` (choices and departures in `../DEVIATIONS.md` D8). Exploratory; it cannot change P3, P4, P5 or the title. Code: `ttr/ttr_common.py` (hooks: one zero register appended after the positional embeddings, per-image per-neuron max of Algorithm 2, CLIP's CLS kept), `ttr/ttr_select.py <clip|siglip>` (register neurons from 100 MIT1003 training images; no display read), `ttr/ttr_score.py plain|norms|arms` (plain arm with gates F and P; gate K on the A15(a) images; ttr and reg0 arms), `ttr/ttr_analyze.py` (statistics, numpy and scipy only). Tests: `ttr/test_ttr.py` on tiny random-weight models and synthetic tables (58 checks, 0 failed; `ttr/tests_ttr_output.txt`). Run order and wall times: `TTR_T`, `TTR_P_siglip`, `TTR_P_clip`, `TTR_S_siglip`, `TTR_S_clip`, `TTR_K`, `TTR_C_siglip`, `TTR_C_clip`, `TTR_A`, `TTR_A_r2` in `../results/run_logs/` (9,810 s in all). Outputs: per-display scores in `../scores_ttr/{plain,reg0,ttr,norms}/`, selection and gate records in `../results/explore/ttr/`, the report in `../results/explore/ttr_results.json` and `ttr_results.md`. With this, the "test-time registers code" listed under "What is not here" exists.
- 18 to 19 November 2026 (23:40 to 00:01 UTC), the remaining saved-output steps of RUN_PLAN Step 1 (choices and departures in `../DEVIATIONS.md` D9 to D11; script hashes in `../results/explore_scripts.sha256`, recorded before each first run):
  - S3 `explore_saved.py` (rule Section 6 items 1, 2, 3, 5, 17, 18; items 7 first part, 14 and 20 recomputed with their own code and compared leaf by leaf with the stored outputs, identical). Tests `test_explore_saved.py` (synthetic, 44 checks, 0 failed). Output `../results/explore/explore_saved.json` and `explore_saved.md`. numpy and scipy only; about 10 s.
  - S4 `a6_absolute.py` (rule 6.7 second part: CC, NSS and information gain on held-out MIT1003 with the centre bias off; heads only on the cached float16 features; centre-bias-on reproduction gate first). Tests `test_a6_absolute.py` (26 checks, 0 failed). Output `../results/explore/A6_absolute.json` and `A6_absolute.md`. Torch on the CPU with the stub; about 5.5 min.
  - S5 `tokens/`: copies of `v2/encoding/analyze.py`, `plots.py`, `report.py` and `looking-at-nothing/analysis/encoding_nulls.py` with outputs redirected to `../A3_tokens/published/` (they reproduce the published `results.json`, `per_config_indices.npz` and `encoding_nulls.json` exactly), and `tokens/a3_tokens_published.py` (rule 6.8 (i), (ii), (iii), (v) for DINOv2-B, CLIP-L, SigLIP-L). Tests `tokens/test_a3_tokens.py` (25 checks, 0 failed). Output `../results/explore/A3_tokens_published.json` and `.md`, `../A3_tokens/published/tab_tokens.tex`. No torch; about 1 min in all.
  - With these, every item of "What is not here" above exists.
