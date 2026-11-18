# Run plan for the re-analysis (after the rule is public)

- Written 18 November 2026 with `REANALYSIS_RULE.md` and revised the same day after the two rule reviews (`REVIEW_RESPONSES.md`), before the rule was fixed. Nothing listed here has been run on real data, except the renderer (stimuli only), the unit tests (synthetic data only) and the feature-path check on one synthetic noise image.
- Folders: `RE` = this folder (`neuro/reanalysis` in the project tree; `neurocomputing/` in the public repository); `ROOT` = the project root (two folders above `RE`, or `$LAN_PROJECT_ROOT`). Interpreter: `$PY`, the Python 3.12 environment of the saliency project (torch 2.12, numpy 2.5, scipy 1.18). Model code: `$LAN_SAL_SRC` (default: the `saliency_research` folder next to the project root). No path in this package is absolute.
- Every CPU step while the GPU job runs: one Python process at a time, `nice -n 10`, `OMP_NUM_THREADS=4`, `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1` (and `python -B`). Torch-importing CPU steps load the stub `RE/_stub/ab_objective.py` (identical to the project-root file except `DEV = "cpu"` and locations read from environment variables, so no Metal backend query at import). After starting any torch process, check that its PID holds no Metal client:
  `ioreg -l -w0 -r -c AGXDeviceUserClient | grep -o '"IOUserClientCreator" = "pid [0-9]*, [^"]*"' | grep "pid <PID>,"` (no output = good; otherwise kill it and fix the code). Checked on 18 November 2026: `test_features_cpu.py` (stub, `recommon.py` with `V3_DEVICE=cpu`, three backbones loaded and run on the CPU) held no Metal client.
- Every script that scores a display or computes a statistic for this re-analysis calls `require_public()` first (rule Section 8.1); new scripts written for Steps 1 to 4 must do the same (`analyze_re.require_public()` or `scoring/common_rigid.require_public()`).
- `tests_output.txt` and `tests_features_output.txt` are hashed: after fixing, rerun the tests only into other files.
- CPU estimates use the per-display times of `neuro/understand/data_inventory.md` Section 6, measured with no GPU job running, and a factor of 1.5 to 2 under the GPU job's load (as the DeepGaze IIE logs suggest).

## Step 0. Fix and publish

| # | action | output |
|---|---|---|
| 0.1 | Done on 18 November 2026: the rule, its code and this plan were revised after the two reviews; `fix_rule.py` was run once all files were final | `REANALYSIS_RULE.sha256` (SHA-256 of the rule and of every file of its Section 10, UTC time of fixing) |
| 0.2 | Main session, with the author's OK: copy `neuro/push_staging/` into the repository as listed in `neuro/push_staging/MANIFEST.md` (files byte for byte, because their hashes are part of the rule), commit, push | public commit |
| 0.3 | `echo "<full commit hash> https://github.com/Deep-c-Vision-LTD/looking-at-nothing/commit/<full commit hash> pushed <UTC time>" > RE/PUBLIC_PUSH.txt` | enables every guarded script |
| 0.4 | Check: `cd RE && grep -E '^[0-9a-f]{64}  ' REANALYSIS_RULE.sha256 \| shasum -a 256 -c --quiet && echo fixed files unchanged` | |

## Step 1. CPU, from saved outputs (any time after Step 0)

| # | command (in `RE`) | reads | writes | CPU |
|---|---|---|---|---|
| S1 | `nice -n 10 $PY -B -I analyze_re.py sanity` | published score files (read only) | `out/sanity.json`; must print PASS (rule Section 3.1), else stop | < 2 min |
| S2 | `nice -n 10 $PY -B -I analyze_re.py primary --which P1,P2,P3,P4` | `v3/scores/raw_v3maps`, `v3/scores/raw`, `v3/scores/raw_nulls.json`, `v3/gpu/full/scores` | `out/primary_P1_P2_P3_P4.json` (with the P1, P2 and P3 wording filled in) | about 5 min (9,999 Webb draws x 30 P1 comparisons; 46,656 enumerated vectors x 30 P2 comparisons; 10,000 BCa resamples x 18 P4 contrasts, plus the ratio intervals) |
| S3 | exploratory A4.1, A4.2, A4.3, A4.5, A6 margin curve, S13, A10 per-support-ratio table, P2 companions (new script `explore_saved.py`, written from rule Section 6 items 1, 2, 3, 5, 7, 14, 17, 18, 20, using the engine of `analyze_re.py` and calling `require_public()`) | the same saved scores; `v3/gpu/full/scores/mit_*.json` | `out/explore_saved.json` | < 0.3 h |
| S4 | A6 absolute metrics with the centre bias off: copy of `v3/gpu/eval_full.py` `run_mit()` with `use_cb = False`, heads-only on `v2/readout/cache/<fset>_mit200.npy` (`V3_DEVICE=cpu`, `gpu/recommon.py` import path) | 60 checkpoints, MIT1003 maps of the saliency project, `ab/maps.npy` (information-gain baseline) | `out/A6_absolute.json` | 0.2 h |
| S5 | A3 token analysis under both nulls from saved components (copies of `v2/encoding/analyze.py`, `report.py`, `plots.py` and `looking-at-nothing/analysis/encoding_nulls.py` into `RE/explore/tokens/`, outputs redirected) | `v2/encoding/tokens`, `results.json`, `per_config_indices.npz`, `v2/verify/encoding_nulls.json` | `A3_tokens/{tokens.json, tab_tokens.tex, fig_tokens.pdf}` | 0.1 h |
| S6 | Secondary item 7 with the corrected sign: copy of `nn/analyze_nn.py` (lines 265 to 267 use the unrounded E) into `RE/explore/item7/`, rule's CI and 100,000-permutation p | `nn/results/*`, `v3/scores/raw*` | `out/item7_corrected.json` | minutes |

## Step 2. CPU, new token and norm computations (torch on the CPU; one process)

| # | command | writes | CPU |
|---|---|---|---|
| T1 | A3(iv) registers token components: copy of `looking-at-nothing/analysis/encoding/run_vit.py` into `RE/explore/tokens/`, backbone `facebook/dinov2-with-registers-base`, CLS and 4 register tokens dropped (5 tokens), all 12 blocks, 336 and 448 px, the 100 configurations of `v2/encoding`. The copy takes its output folders from `common_enc` (`TOK`, `LOGS`, inside `looking-at-nothing`), so both must be redirected to `RE/A3_tokens/registers/` | `A3_tokens/registers/` (about 50 MB) | 0.2 h |
| T2 | A15(a) token norms: copy of `nn/registers/norm_diag.py` with CLIP-L/14 and SigLIP-L/16, all blocks read (SigLIP-L 24 and the 12 sweep blocks), 24 Kanizsa displays of the receptive-field rule and the 24 MIT1003 images of `norm_diag.json`, 336 and 448 px. The original imports `regcommon`, which loads the original `ab_objective` (Metal query at import), so the copy must import `gpu/recommon.py` with `V3_DEVICE=cpu` instead | `A15/norm_diag_clip_siglip.json`; applies the Critic fix 18 condition (rule Section 6, item 12) | 0.5 h |

## Step 3. CPU scoring of the whole-display rotation set and its controls (P5; mock review A9)

Stimuli: `RE/stim_rigid` (660 displays: 480 rigid rotations and 180 orientation-matched controls mis<theta>_control180; `items.json`, `aoi/`, `manifest.json`). Outputs: `RE/scores_rigid/` (`<tag>.json` in the `{name: {measure: value}}` layout of `v3/scores/raw`, with the extra measures r_N, r_K, f_polygon, f_notch, f_ink of rule Section 6 items 4, 10 and 11; heads in `scores_rigid/heads/scores/stim_rigid_<tag>.json`). No maps are saved: the map-based exploratory analyses of the rigid set use the extra measures, and A4.6 and A11 are stim_v3 and stim_alt only. Each script first runs `require_public()`, then its pipeline check on stim_v3 displays (rule Section 3.2), and stops if either fails. Run in this order, one at a time:

| # | command (in `RE`) | models | CPU (660 displays; no load / under the GPU job's load) |
|---|---|---|---|
| R1 | `nice -n 10 env OMP_NUM_THREADS=4 $PY -B -I scoring/local_nulls_rigid.py` | 15 local nulls | 1.5 min / 2 to 3 min |
| R2 | `nice -n 10 env OMP_NUM_THREADS=4 $PY -B scoring/score_rigid.py iie` | DeepGaze IIE | 19 min / 35 to 60 min |
| R3 | `nice -n 10 env OMP_NUM_THREADS=4 $PY -B scoring/score_rigid.py sweep:dinov2 sweep:clip sweep:siglip` | all 18 sweep readouts, 336 and 448 px | 21 min / 35 to 45 min |
| R4a | CPU check of the heads (rule Section 3.2), in `RE/gpu`: `for spec in "dinov2 dinov2_L12_R4_s0,dinov2_L6_R1n_s0" "siglip siglip_L24_R4_s0,siglip_L24_R1n_s0" "iie iie_shapenetc_R4_s0,iie_shapenetc_R1n_s0"; do set -- $spec; nice -n 10 env V3_DEVICE=cpu V3_MAPS=none OMP_NUM_THREADS=4 $PY -B eval_re.py --bb $1 --out ../check_cpu_$1 --ckpt-dir $ROOT/v3/gpu/full/ckpt --tags $2 --stimsets stim_v3 --stim-limit 18 --skip-mit && $PY -B -I check_eval_re.py ../check_cpu_$1 --tol 1e-4 \|\| break; done`; registers: `eval_re.py --bb dinov2reg --out ../check_cpu_dinov2reg --ckpt-dir $ROOT/nn/registers/full/ckpt --tags dinov2reg_L12_R4_s0 --stimsets stim_v3 --stim-limit 18 --skip-mit`, then `check_eval_re.py ../check_cpu_dinov2reg --tol 1e-4 --pub $ROOT/nn/registers/full/scores`. Stop if any check fails | two heads per plain feature set, one registers head, 18 displays | about 4 min / 6 min |
| R4 | in `RE/gpu`: `nice -n 10 env V3_DEVICE=cpu V3_MAPS=none OMP_NUM_THREADS=4 $PY -B eval_re.py --bb dinov2 --out ../scores_rigid/heads --ckpt-dir $ROOT/v3/gpu/full/ckpt --stimsets stim_rigid --skip-mit`, then the same with `--bb siglip` and `--bb iie` | the 60 swap heads (backbone on the CPU; the float16 feature cache is valid only for the stim_v3 displays, so it is not used here) | 26 min / 40 to 55 min |
| R5 | in `RE/gpu`: `nice -n 10 env V3_DEVICE=cpu V3_MAPS=none OMP_NUM_THREADS=4 $PY -B eval_re.py --bb dinov2reg --out ../scores_rigid/heads --ckpt-dir $ROOT/nn/registers/full/ckpt --stimsets stim_rigid --skip-mit` | the 6 registers R4 heads | 6 min / 8 to 11 min |
| R6 | `nice -n 10 env OMP_NUM_THREADS=4 $PY -B scoring/score_rigid.py msdb`, one process (overnight while the GPU job runs, so that the CPU battery is done before `GO_AHEAD`; never in shards while G3 or A14 run, because `gpu_idle.sh` treats busy Python processes as heavy and shards would slow the GPU data loader). After the GPU work, three shards (`SCORE_SHARD=0/3`, `1/3`, `2/3`, 3 threads each, then `scoring/score_rigid.py merge DeepGaze_MSDB`) are allowed | DeepGaze MSDB | 4.8 h / 6 to 8 h in one process (26 s per display); about 1.6 h wall with 3 shards and no load |
| R7 | `nice -n 10 $PY -B -I analyze_re.py primary --which P5` (computes P1 and P2 in the same call for the abstract rule of rule Section 5.5) | P5 (IIE and MSDB against the 15 nulls) | < 10 min |
| R8 | rigid-rotation profiles of every readout (rule Section 6, item 15; new script `explore_rigid.py` with the engine of `analyze_re.py`, calling `require_public()`) | exploratory | < 0.2 h |

Total CPU for the battery: about 1.3 h without MSDB and 4.8 h for MSDB with no load; about 2 to 3 h without MSDB and 6 to 8 h for MSDB under the GPU job's load.

Note for R4 and R5: `eval_re.py` writes into `scores_rigid/heads/{scores,logs,done,...}` only; the checkpoints are read from the published runs.

## Step 4. CPU, map-based exploratory analyses (one process)

| # | what (rule Section 6) | inputs | writes | CPU |
|---|---|---|---|---|
| M1 | A4.4 log mass (inside, notch sectors = analytic sector minus ink, ink = at least 128 grey levels from the background, using `render_rigid.notch_geom` and `render_rigid.ink_mask` at phi = 0 for stim_v3; on `stim_rigid` the shares stored by Step 3) | saved maps `v2/scores/maps`, `v3/scores/maps`, `v3/gpu/full/maps/stim_alt`; heads-only re-inference for the swap stim_v3 maps from `v2/readout/cache/stim` (in memory); the 9 sweep blocks without maps through a copy of `nn/score_nn.py` (maps in memory) | `out/A4_logmass.json` | 1.5 h |
| M2 | A4.6 calibration robustness, stim_v3 and stim_alt only (rank equalisation; T = 0.5, 0.75, 1, 1.5, 2; logits re-inferred, never inverted; MSI-Net rank equalisation only) | as M1 | `out/A4_calibration.json` | 2 to 2.5 h |
| M3 | A12 Kavsek-style AOIs and A13 region variant (`render_rigid.polygon_exact` at phi = 0 for stim_v3 from the saved maps; for `stim_rigid` the r_K and r_N values stored by Step 3) | as M1; `scores_rigid/` | `out/A12_A13.json` | 0.8 h |
| M4 | A11 mixture fit, stim_v3 and stim_alt only (no point-spread probe) | saved IIE and MSDB maps | `out/A11.json` | 0.7 h |

## Step 5. GPU runs G3 and A14 (after the GPU job ends; decision D6)

Scripts in `RE/gpu/` (copies, imports redirected to `recommon.py`; nothing else changed except as listed in each header):

| file | copied from | change |
|---|---|---|
| `recommon.py` | `nn/registers/regcommon.py` | adds `dinov2reg` (as regcommon) and `dinov2ms` (`hidden_states[7]+[11]`, 1,536 channels; `hidden_states[11]`, 768 channels); stub import when `V3_DEVICE=cpu`; `require_public()` and `measures_extra()` (lazy import of `scoring/common_rigid.py`) |
| `stats_re.py` | `v3/gpu/stats_full.py` | import; `require_public()` (R1n input statistics; needed because no R1n was trained on these features before) |
| `train_re.py` | `nn/registers/train_reg.py` (= `v3/gpu/train_full.py`) | import; `require_public()` |
| `eval_re.py` | `nn/registers/eval_reg.py` (= `v3/gpu/eval_full.py`) | import; `require_public()`; `--ckpt-dir`, `--tags`, `--skip-mit`; stimulus set `stim_rigid` with the extra measures; `V3_MAPS` default `none` |
| `analyze_gpu.py` | `nn/registers/analyze_reg.py` (= `v3/gpu/analyze_full.py`) | import; `require_public()` (published percentile statistics, for continuity) |
| `check_eval_re.py` | `nn/registers/check_eval.py` | compares a rescored published checkpoint with the published scores; exit 1 above the tolerance |
| `gpu_idle.sh` | `nn/registers/gpu_idle.sh` | also treats as heavy: any Python process holding a Metal client (whatever its CPU share), `train_re.py`, `eval_re.py`, `stats_re.py`, `run_gpu_re.sh`, and the ablation runner `ablation/run_all.sh` with its steps (`train_arm.py`, `eval_arm.py`, `cache_clip.py`, `score_arm.py`) |
| `run_gpu_re.sh` | `nn/registers/run_reg.sh` | queue below, with the gates |

Command, once the ablation runner that held the GPU on 18 November 2026 (`ablation/run_all.sh`) has exited, its log ends with its analyze step, no other GPU job runs, and the author has created `RE/gpu/GO_AHEAD`:

```
cd "$RE/gpu" && nohup bash run_gpu_re.sh both > logs/nohup.log 2>&1 &
```

Gates inside the script, in order: rule fixed and public, and every hashed file and stimulus unchanged (`require_public()`); `GO_AHEAD`; no ablation runner alive (refuse, do not wait); `gpu_idle.sh` idle twice, 60 s apart, before every step (a Python process holding a Metal client counts as heavy); at least 10 GB free before every step; MPS available; sanity gate 3 (the copied evaluation reproduces the published scores of `dinov2_L12_R4_s0` and of `dinov2reg_L12_R4_s0` on MIT1003, stim_v3 and stim_alt to 1e-6). Each cell is trained once per seed (rule Section 5.6); a crashed step is resumed or restarted with identical settings and logged.

Queue and time (MPS; published epoch times: DINOv2 group of 6 heads 517 to 542 s, group of 4 heads 452 to 516 s, registers R4 pair 443 to 497 s; the backbone pass and data loading dominate; add at least 60 s per step for the idle check):

| step | heads per forward pass | estimate |
|---|---|---|
| sanity gate 3 (eval of two published heads, MIT1003 + stim_v3 + stim_alt) | 1 each | 10 min |
| G3 `stats_dinov2reg` | n/a | 7 min (published DINOv2 statistics pass: 404 s) |
| G3 `train_dinov2reg_s{0,1,2}`, archs R1n, R1, R2, R3 | 8 (4 readouts x blocks 12 and 6) | 49 to 60 min per seed (6 epochs of about 490 to 600 s) |
| G3 `eval_dinov2reg_s{0,1,2}` | 8 | about 4 to 6 min per seed |
| A14 `stats_dinov2ms` | n/a | 7 min |
| A14 `train_dinov2ms_s{0,1,2}`, archs R1n, R1, R2, R3, R4 | 10 (5 readouts x 2 feature sets; the 1,536-channel heads are larger) | 56 to 75 min per seed (6 epochs of about 560 to 700 s) |
| A14 `eval_dinov2ms_s{0,1,2}` | 10 | about 7 min per seed |
| continuity statistics (CPU) | n/a | minutes |
| total | | about 2.9 to 3.4 h (G3) + 3.2 to 4.0 h (A14) of MPS time |

Do all four readouts on the registers checkpoint fit in one shared forward pass, as in the swap? Yes. The published swap trained up to 6 heads per pass (R1n, R1, R4 on blocks 12 and 6) and had a prepared "folded" mode with all 10; G3 trains 8 heads per seed in one pass (memory is not limiting: 768-channel features on a 24 x 24 grid, batch 32). The grouping does not change any head's initial weights or batch order: each head is built immediately after `torch.manual_seed(seed)`, the batch order comes from `numpy.RandomState(seed)`, and the R1n calibration runs before the dropout stream is seeded. Only R2, R3 and R4 contain dropout (`Dropout2d(0.1)`; `deepgaze_pytorch/deepgaze2e.py`, the pointwise readout, has none), so in the G3 job the R2 and R3 heads draw the same dropout sequence (block 12 R2, block 12 R3, block 6 R2, block 6 R3 per batch) as the published R2 and R3 group did, provided R1n and R1 draw no random numbers in their forward and backward passes. In A14 all five readouts share one job, so the R2, R3 and R4 dropout masks differ from those of a split job: a different random draw under the same recipe, which is harmless but not identical to a split run.

Disk: no maps are saved (`V3_MAPS=none`); checkpoints and resumable states need about 2 to 5 GB in all (states are kept for resumption; they can be moved off the disk after the runs, by the author). After the runs: `analyze_re.py primary --which P6` (P3 is recomputed in the same call for the title decision), which writes the title decision (rule Section 5.6) into `out/primary_P6.json`.

## Step 6. Order and dependencies

1. Step 0 (fix, push, `PUBLIC_PUSH.txt`) before anything else.
2. S1 sanity gate; then S2 (P1 to P4).
3. S3 to S6, T1, T2, R1 to R5, R6 (one process, overnight), R7, R8, M1 to M4: one at a time while the GPU job runs.
4. Step 5 (G3, A14) after the GPU job, independent of Steps 1 to 4; P6 and the title decision last.
5. Any departure: `REANALYSIS_DEVIATIONS.md` (rule Section 9).

## Files of this package

- Rule and notes: `REANALYSIS_RULE.md`, `REVIEW_RESPONSES.md`, `reviews/` (the two review notes; `reviews/support/` holds the reviewers' scripts and outputs, not hashed), `RF_RULE_CORRECTIONS.md`, `RUN_PLAN.md`, `known_values.json` (from `extract_known.py`).
- Engine and tests: `analyze_re.py`, `test_analyze_re.py`, `tests_output.txt`, `test_features_cpu.py`, `tests_features_output.txt`.
- Stimuli: `render_rigid.py`, `stimcode/` (verbatim copies of `stimuli_v3.py`, `stimuli_v2.py`, `stimuli_ehr.py` plus a `paths.py` shim), `stim_rigid/` (660 PNG, 420 AOI npz, `items.json`, `manifest.json`, `contact_sheet.png`), `stim_rigid_render.log`.
- Scoring copies: `scoring/common_rigid.py`, `scoring/score_rigid.py`, `scoring/local_nulls_rigid.py`; stub `_stub/ab_objective.py`.
- GPU copies: `gpu/` (table above).
- Fixing: `fix_rule.py`, `REANALYSIS_RULE.sha256`.
