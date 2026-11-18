# Design review: rigid-rotation stimuli, the A9 analysis (P5), the scoring plan and the G3/A14 run plan

- Independent reviewer, 18 November 2026 (10:15 to 10:42 UTC). I did not write the files under review.
- Read in full: `neuro/REWRITE_PLAN.md` (including the Critic section K.1 to K.8), `neuro/reanalysis/REANALYSIS_RULE.md`, `RUN_PLAN.md`, `render_rigid.py`, `stim_rigid/manifest.json` (all non-file keys), `stim_rigid/items.json` (structure), `scoring/{common_rigid,score_rigid,local_nulls_rigid}.py`, `gpu/{recommon,train_re,eval_re,stats_re,check_eval_re,run_gpu_re.sh,gpu_idle.sh}`, the P5, P6 and title code of `analyze_re.py`, its P5 unit test, `_stub/ab_objective.py`; diffs against `nn/registers/{regcommon,train_reg,eval_reg,gpu_idle.sh}`, `v3/gpu/{fullcommon,train_full,stats_full}.py`, `v3/scores/{common_alt,score_alt,local_nulls_alt,measure_v3maps}.py`, `gate_kanizsa.py` (project root) and `looking-at-nothing/models/gate_kanizsa.py`, `v2/readout/common.py`, `deepgaze_pytorch/{deepgazemsdb.py,features/dino.py}`; timing logs in `v3/gpu/full/logs`, `nn/registers/logs`, `nn/registers/full/logs`, `nn/RUN_LOG.md`, `neuro/understand/data_inventory.md` Section 6.
- What I ran (CPU only, `nice -n 10`, `OMP_NUM_THREADS=4`, one process at a time, no scoring of any model or null): independent stimulus checks `neuro/review_rule/checks/design_checks{,2,3}.py` (numpy, PIL, scipy; no torch) with outputs `design_checks{,2,3}.json` and a crop sheet `crops_v3ill_mis20_rig20ill_rig20ctrl_v3ctrl.png`; one Metal probe `checks/metal_probe.py` (stub, `AutoModel.from_pretrained` offline, `import deepgaze_pytorch`, no model run on any image; its PID held no Metal client); `gpu/gpu_idle.sh` three times (read-only `ps`). Two over-tight crop images of mine were moved to `neuro/_trash/review_rule_crops/`. Nothing outside `neuro/` was written; no downloads; no git.

## Verdict

The 480 rigid-rotation displays are what the rule says they are (findings 10 to 12). Two problems must be fixed before the rule is hashed and pushed: the P5 estimand does not hold edge orientation fixed on the control side (finding 1), and the GPU-idle gate does not see the running 31-hour job (finding 2). Six SHOULD-FIX items follow (3 to 8). The scoring path matches the published pipeline. The CPU and GPU time estimates are plausible but slightly optimistic under load. Disk use fits.

| # | level | topic |
|---|---|---|
| 1 | BLOCKER | P5: the "alignment-specific loss" carries the control display's orientation response |
| 2 | BLOCKER | `gpu_idle.sh` reports idle while the 31-hour GPU job runs |
| 3 | SHOULD-FIX | P5 half criterion is underpowered; category 2 wording would assert a shrinkage that was not shown |
| 4 | SHOULD-FIX | Scoring scripts have no public-push guard; existing guards check too few files |
| 5 | SHOULD-FIX | Title rule: vacuous pass when no non-R4 readout is valid; P3 not required for T1 |
| 6 | SHOULD-FIX | Sanity gate 3 does not cover the two new feature paths |
| 7 | SHOULD-FIX | Absolute user paths and the project-root `gate_kanizsa.py` in files to be hashed and pushed |
| 8 | SHOULD-FIX | No maps are kept for `stim_rigid`, but RUN_PLAN M3 uses them |
| 9 | NOTE | CPU time under load, and ordering of MSDB shards against G3/A14 |
| 10 | NOTE (pass) | Stimuli are rigid rotations about the figure centre; inventory complete |
| 11 | NOTE (pass) | Pixel statistics match stim_v3 |
| 12 | NOTE (pass) | AOIs recomputed from geometry |
| 13 | NOTE (pass) | Scoring path equals the published pipeline |
| 14 | NOTE | G3/A14 recipe parity, MSDB tensors, dropout grouping |
| 15 | NOTE | GPU time estimates against the logs |
| 16 | NOTE | Disk |
| 17 | NOTE | MPS, downloads, writes outside `neuro/` |
| 18 | NOTE | Small consistency items |

---

## 1. BLOCKER: P5 does not hold edge orientation fixed on the control side

**Evidence.** Rule Section 5.5 and `analyze_re.p5_parts` define, per configuration, with I = r(illusory), C = r(control180), M = r(mis20), RI = r(rig20 illusory), RC = r(rig20 control180):

- L_uni = I - M (control-free), L_rig = (I - C) - (RI - RC) (a difference of contrasts),
- A = L_uni - L_rig = (RI - M) - (RC - C).

Write each display's score as a baseline for inward or outward notches plus three effects: o, the change produced by turning every notch edge by theta (o_in for inward notches, o_out for outward notches); a, the alignment effect lost when inward inducers are misaligned; g, the change produced by rotating the configuration on the canvas (rotated AOI raster, oblique virtual figure; g_in, g_out). Then I = b_in, M = b_in + o_in - a, RI = b_in + o_in + g_in, C = b_out, RC = b_out + o_out + g_out, and

- A (rule) = a + (g_in - g_out) - o_out.

The inward orientation term o_in cancels, but the orientation response of the control display, o_out, does not: the rule's control for M is the unrotated control180, whose edges are cardinal, while the control for RI is the rotated control180, whose edges are oblique. A pure orientation account (a = 0, g = 0) with any o_out different from 0 yields a nonzero "alignment-specific loss". The rule's sentence "the property-1 estimate with the orientation change partialled out" (Section 5.5) and the Category 1 wording "so it is not an orientation effect" (Section 7.2) are true only if o_out = 0. The unit test does not probe this: `test_analyze_re.synth_scores` sets r(rig20 control180) = r(control180) plus noise (line 266), that is, o_out = g_out = 0 by construction. The same mixing affects C(theta) = L_uni - 2 L_rig and s(theta) = L_rig / L_uni, which divide a contrast difference by a control-free loss.

I checked that the orientation premise itself holds: the edge-orientation histograms (straight notch edges only) of rig theta illusory correlate with stim_v3 mis theta at r >= 0.907 and of rig theta control180 at r >= 0.921, against r <= 0.666 with the unrotated illusory display (180 pairs, theta = 10, 20, 30; `checks/design_checks3.json`). So RC has oblique edges like M; C does not. No existing display set has oblique edges with outward notches at unrotated positions (stim_v3 conditions: illusory, real, real_filled, realC, control180, control90, mis10/20/30; stim_alt: alt10/20/30 a/b).

**Fix (render before fixing the rule; rendering is allowed now).**

1. Render the orientation-matched control of the uniform rotation, "mis theta control180": every control180 inducer turned by a further +theta about its own centre, positions unrotated, AOIs = the stim_v3 AOIs of the configuration. In `render_rigid.py`: `T.down(T.kanizsa_hi(shape, cx, cy, side, sr, 180 + theta, False, False, fg, bg))`; name `{cfg}_mis{theta}_control180`, `config = f"{cfg}_mis{theta}c"`, AOI file `aoi/{cfg}_mis{theta}c_aoi.npz` with keys copied from `v2/stim_v3/img/{cfg}_illusory_aoi.npz` and `v3/stim_alt/aoi/{cfg}_aoi.npz` (core30, core45), plus `ink_*`, `notch_*`, `notchgeom_*` as for the rigid set. Required: theta = 20 (60 displays). Recommended: theta = 10, 20, 30 (180 displays), so the exploratory angles are orientation-matched too. Add the check "rotation 180 + theta of each inducer equals stim_v3 mis theta rotated by 180 about each inducer centre" and the md5s to the manifest; update `common_rigid.py`'s count assertion (480 and 240 become 540 and 300, or 660 and 420).
2. Redefine P5 in one currency, every term a contrast against an orientation-matched control: E(0) = I - C, E_rig(theta) = RI - RC, E'_mis(theta) = M - MC (MC = r(mis theta control180)). Then
   - L_uni^E(theta) = E(0) - E'_mis(theta) (loss under uniform rotation),
   - L_rig(theta) = E(0) - E_rig(theta) (unchanged; loss from turning edges and figure together, alignment kept),
   - A*(theta) = L_uni^E - L_rig = E_rig(theta) - E'_mis(theta) = a + (g_in - g_out) (alignment effect at fixed edge orientation theta),
   - C*(theta) = L_uni^E - 2 L_rig, s*(theta) = L_rig / L_uni^E (BCa, stability flag on L_uni^E).
   Tests (i) and (ii) unchanged in form, on A* and C*. In `analyze_re.p5_parts`: `E_mis = per_config(D, val, f"mis{theta}", f"mis{theta}_control180")`, `L_uni = E_ill - E_mis`, `L_rig = E_ill - E_rig`, `A = E_rig - E_mis`.
3. Keep, as reported secondary quantities (never decision-bearing): the current A (rule as drafted), the control-free A_cf = RI - M, and the control's change under rigid rotation RC - C, for every model and null. P1 keeps its control-free L20.
4. Text: Section 5.5 "partialled out" becomes "A* compares an aligned and a misaligned display whose notch edges have the same orientation, each against a control with the same edge orientation"; Section 7.2 Category 1 "so it is not an orientation effect" becomes "so it is not explained by the change in edge orientation"; Section 1.1 item 14 lists the added displays as rendered before fixing.
5. Add a unit test with a = 0 and o_out = 0.02: the drafted A must come out near -0.02, A* near 0; and one with a = 0.03 and o_out = 0.02: A* near 0.03.

Cost: +60 displays (+12.5% of the battery; MSDB +26 min) or +180 (+37.5%; MSDB +79 min). Note that A* is a difference of two contrasts, so its interval will be wider than that of L20 (see finding 3).

## 2. BLOCKER: `gpu_idle.sh` reports idle while the 31-hour job runs

**Evidence.** The running job is `xsoyelnet_ml_letter/ablation/train_arm.py --arm A1 --seed 0` (pid 61966, parent `zsh .../ablation/run_all.sh`, pid 57509), and it holds an AGX Metal client (`ioreg ... | grep "pid 61966,"` gives `"pid 61966, Python"`). `gpu/gpu_idle.sh` calls a process heavy only if it is a Python process above 20% CPU or its command line names a job of this project. I ran it three times about 3 s apart: exit 1 once (pid 61966 at 50.5% CPU), then exit 0 twice (1.8% and 2.3% CPU). An MPS-bound training process often sits below 20% CPU, so "idle twice, 60 s apart" can pass while the job trains. `run_all.sh` also runs about 20 GPU steps (`train_arm.py`, `eval_arm.py`, `cache_clip.py`) with CPU steps (`score_arm.py`) in between; none of these names is listed. Only the manual `GO_AHEAD` file stops a premature start. In the other direction, `run_all.sh` waits for "no Python AGX client" before each GPU step, so it could start a step in the 60 s gaps between `run_gpu_re.sh` steps, after which the broken check would not see it.

**Fix.** In `gpu/gpu_idle.sh`, before `exit 0`, add a Metal-client test and the job names:

```bash
# any other Python process holding a Metal (AGX) client is a GPU job
for p in $(ioreg -l -w0 -r -c AGXDeviceUserClient 2>/dev/null | grep -o '"IOUserClientCreator" = "pid [0-9]*, [^"]*"' \
           | grep -i ', python' | sed -E 's/.*pid ([0-9]+),.*/\1/' | sort -u); do
  case "$self_pids" in *" $p "*) continue;; esac
  busy="$busy\n  pid $p holds a Metal client"
done
# the ablation runner and its steps, whatever their CPU share
pgrep -f 'xsoyelnet_ml_letter/ablation/run_all\.sh' >/dev/null && busy="$busy\n  ablation run_all.sh alive"
```

and extend the name pattern with `train_arm\.py|eval_arm\.py|cache_clip\.py|score_arm\.py`. In `run_gpu_re.sh`, also refuse to start (not just wait) while `pgrep -f 'xsoyelnet_ml_letter/ablation/run_all\.sh'` finds a process. In rule Section 8.2 and RUN_PLAN Step 5, name the job ("the ablation runner `xsoyelnet_ml_letter/ablation/run_all.sh` has exited and its log `ablation/logs/run_all.log` ends with the analyze step") instead of "about 19 November 15:30 UTC".

## 3. SHOULD-FIX: the P5 half criterion is underpowered, and Category 2 wording would then assert something unshown

**Evidence.** Category 1 needs (ii) C > 0 with p < .05. A rough power check from the published percentile half-widths (se = width / 3.92): IIE L20 se 0.0097, E se 0.0133; MSDB L20 se 0.0075, E se 0.0100. Suppose rigid rotation removes nothing (expected L_rig = 0) and E and E_rig correlate at rho across configurations. Then sd(L_rig) is about se(E) x sqrt(2(1 - rho)), and sd(C) is about sqrt(se(L20)^2 + 4 sd(L_rig)^2). For MSDB: rho = 0.5 gives sd(C) 0.021 and expected t 1.4 (power about 0.3); rho = 0.8 gives 0.015 and t 2.0 (about 0.5). For IIE: t 1.8 (about 0.4) and t 2.6 (about 0.7). With A* of finding 1, L_uni^E is itself a contrast difference, so power is lower still. A model can therefore land in Category 2 even when rigid rotation removes none of the loss. The Category 2 sentence ("Rotating the whole display ... reproduced [s] of the uniform-rotation loss ... uniform rotation does not isolate alignment") would then state a shrinkage that the data did not show. The abstract consequence (lead with alternation) is conservative and fits D7. The wording does not fit.

**Fix.** Split Category 2 by the point estimate, with the abstract consequence unchanged:

- 2a "shrinks": (i) met and s* >= 0.5. Wording as drafted.
- 2b "half criterion undecided": (i) met, s* < 0.5, (ii) not met. Wording: "The alignment-specific loss ([A*] [CI]) separated [model] from all 15 local nulls. Rotating the whole display reproduced [s*] [CI] of the uniform-rotation loss, an amount that could not be shown to be below half, so the uniform-rotation result is reported with that caveat and the abstract leads with the alternation test."

State in Section 4 or 5.5, before any result, that (ii) is expected to have modest power (give the numbers above as approximate).

## 4. SHOULD-FIX: nothing in the scoring code stops scoring before the push; existing guards check too little

**Evidence.** `analyze_re.require_public()` refuses to run without `REANALYSIS_RULE.sha256` and `PUBLIC_PUSH.txt`, but it verifies only `REANALYSIS_RULE.md` and `analyze_re.py`. `scoring/score_rigid.py`, `scoring/local_nulls_rigid.py` and `gpu/eval_re.py` (RUN_PLAN R1 to R6, run directly) have no guard at all ("after REANALYSIS_RULE.md is public" appears only in a docstring). `run_gpu_re.sh` checks that the two files exist, not that the hashed files are unchanged. None of the scoring scripts checks the PNG and AOI files against `stim_rigid/manifest.json`.

**Fix.** One shared function, for example in `scoring/common_rigid.py` and imported by `gpu/recommon.py`:

```python
def require_public():
    rec, pub = os.path.join(RE, "REANALYSIS_RULE.sha256"), os.path.join(RE, "PUBLIC_PUSH.txt")
    if not (os.path.exists(rec) and os.path.exists(pub)) or len(open(pub).read().split()[0]) < 7:
        sys.exit("refusing: rule not fixed and public")
    for line in open(rec):
        h, *f = line.split()
        if len(h) == 64 and hashlib.sha256(open(os.path.join(RE, f[0]), "rb").read()).hexdigest() != h:
            sys.exit(f"refusing: {f[0]} changed after fixing")
    man = json.load(open(os.path.join(STIM_RIGID, "manifest.json")))
    bad = [f for f, m in man["files"].items() if hashlib.md5(open(os.path.join(STIM_RIGID, f), "rb").read()).hexdigest() != m]
    if bad:
        sys.exit(f"refusing: {len(bad)} stimulus files differ from the manifest")
```

Call it at the top of `score_rigid.py`, `local_nulls_rigid.py`, `eval_re.py`, `stats_re.py` and `train_re.py`. Make `analyze_re.require_public()` verify every hashed line, not two. In `run_gpu_re.sh`, use `(cd "$RE" && grep -E '^[0-9a-f]{64}  ' REANALYSIS_RULE.sha256 | shasum -a 256 -c --quiet) || { say "STOP: fixed files changed"; exit 2; }`.

## 5. SHOULD-FIX: title rule, vacuous pass and P3 consistency (author decision needed)

**Evidence.** Rule 5.6 (c) and `analyze_re.title_decision` require that every valid non-R4 cell on `dinov2reg_L6`, `dinov2reg_L12` and `dinov2ms_L7L11` keeps R4's sign. If no non-R4 cell on a title-bearing set meets the accuracy criterion, (c) passes vacuously and T1 is chosen with no readout comparison on that set. This is not remote. In the swap, R1n and R1 failed the criterion on 3 of 4 feature sets (gaps 0.042 to 0.045). On `dinov2ms_L7L11`, R1's image-wide LayerNorm sees two layers with different token norms, which makes a low CC plausible. Separately, T1 requires "interval excluding zero" for the new cells but only "no opposite-signed interval" for the 10 old P3 cells, so T1 can stand with, say, 9 of 10 P3 cells after Holm (the SigLIP-L block 24 R3 cell has the widest known interval, +0.113 [0.039, 0.188]).

**Fix (proposed; D6 is the author's decision).** Add to (c): "and at least two non-R4 cells on each title-bearing feature set meet the accuracy criterion; otherwise the scoped title". Add (e): "all 10 P3 cells keep R4's sign after Holm adjustment". If the author prefers the literal D6 wording, state explicitly in 5.6 that a feature set with no valid non-R4 cell passes (c), and that the cover letter then says so. Also state that "keeps the sign" is decided by the Holm-adjusted WCR p-value, the printed interval is WCU, and if the two disagree the decision stands and both are reported.

## 6. SHOULD-FIX: sanity gate 3 covers only the plain DINOv2 path

**Evidence.** Gate 3 rescores `dinov2_L12_R4_s0` (`FC.feats`, plain path). G3 uses the `dinov2reg` branch of `recommon.feats`, and A14 uses the new `dinov2ms` branch with `_grid` and channel concatenation. Neither is exercised by the gate. The registers R4 scores are the reference of two P6 families.

**Fix.** (a) Add to gate 3: `eval_re.py --bb dinov2reg --out $CK --ckpt-dir $ROOT/nn/registers/full/ckpt --tags dinov2reg_L12_R4_s0 --stimsets stim_v3,stim_alt` then `check_eval_re.py $CK --pub $ROOT/nn/registers/full/scores` at 1e-6 (the registers check reached max |difference| 0.0 on MPS). Use separate check folders so the two `scores/` directories do not mix. (b) Now, on the CPU with a synthetic image (allowed in this run): check that `feats(dinov2ms)["dinov2ms_L7L11"][:, 768:]` equals `["dinov2ms_L11"]`, and that `["dinov2ms_L11"]` equals the plain extractor's `hidden_states[11]` path (`C2.Extractor("dinov2")` with `hidden_index=[11]`, key `hs11`), bit for bit.

## 7. SHOULD-FIX: user paths and the project-root `gate_kanizsa.py` in the pushed package

**Evidence.** Files hashed by `fix_rule.py` and to be pushed contain absolute user paths: `RUN_PLAN.md` lines 4, 63, 85 and `_stub/ab_objective.py` lines 21, 26, 31, 32. The plan's pre-push check (REWRITE_PLAN Section 6 item 9) forbids `/Users/` paths. Scrubbing after fixing would break the recorded hashes. `scoring/score_rigid.py` imports the project-root `gate_kanizsa.py`, which is not in the repository; its docstring names a company that the pre-push check bans. The repository file `looking-at-nothing/models/gate_kanizsa.py` has identical `load_models` code for `iie` and `msdb` (diffed), and its `paths` import has no side effects.

**Fix.** Before `fix_rule.py` runs: replace the paths with `LAN_PROJECT_ROOT` / `LAN_SAL_SRC` lookups (as `recommon.py` already does) and `RE`-relative text in RUN_PLAN. Point `score_rigid.py` at `looking-at-nothing/models` (for example `sys.path.insert(1, os.path.join(ROOT, "looking-at-nothing", "models"))`, after the stub), and keep the pipeline check, which proves equality on 4 stim_v3 displays. Grep the package for `/Users/` and for the banned names, with word boundaries, as the last step before hashing. (Today the word-boundary grep finds no banned names in `neuro/reanalysis`.)

## 8. SHOULD-FIX: no maps are kept for `stim_rigid`, but RUN_PLAN M3 relies on them

**Evidence.** `score_rigid.py` and `eval_re.py` (with `V3_MAPS=none`) save no maps. RUN_PLAN M3 computes Kavsek-style and region-variant AOIs from "saved maps as M1" and "the stored `polygon` for stim_rigid", but no rigid maps will exist. Any later map-based analysis on the rigid set (A11, A12, A13, log mass, calibration) would require rescoring MSDB (3.5 h or more).

**Fix.** Either (a) save q16 float16 maps of IIE and MSDB on the rigid set (and on the new controls of finding 1): about 1.5 MB per display uncompressed, 0.5 to 1 GB in all, which fits the disk (finding 16). Or (b) add `r_K` and `r_N` (rule 6.10, 6.11, from the stored `polygon` and `ind`) to `common_rigid.measures` so they are computed in the single scoring pass. Or (c) state in RUN_PLAN M3 and rule 6.10 to 6.11 that A12 and A13 are stim_v3-only. Option (b) costs nothing.

## 9. NOTE: CPU time under load, and the order of MSDB shards against G3/A14

Per-display bases (`data_inventory.md` Section 6; checked against `nn/RUN_LOG.md`, measured on 17 November with no GPU job running): nulls 0.06 to 0.1 s, IIE 1.7 s (3.0 to 5.5 s logged under load), 18 sweep readouts 1.8 to 2.0 s, 60 swap heads with backbones 2.4 s, 6 registers heads 0.5 s, MSDB 26.2 s. For 480 displays:

| step | no load | under the ablation's load (x1.5 to 2, as the IIE logs suggest) |
|---|---|---|
| R1 nulls | 1 min | 1 to 2 min |
| R2 IIE | 14 min | 25 to 45 min |
| R3 sweep | 15 to 16 min | 25 to 32 min |
| R4a + R4 swap | 3 + 19 min | 5 + 30 to 40 min |
| R5 registers | 4 min | 6 to 8 min |
| R6 MSDB, one process | 3.5 h | 4.5 to 6 h |
| total | about 1 h + 3.5 h | about 1.5 to 2.2 h + 4.5 to 6 h |

RUN_PLAN's "about 1 h without MSDB, about 4.5 h with it" is the no-load figure; give the range. Multiply by 1.125 (theta = 20 controls) or 1.375 (all three angles) after finding 1. Interaction: `gpu_idle.sh` treats any Python process above 20% CPU as heavy. MSDB shards started after the ablation ends would therefore hold back G3/A14 until they finish, and vice versa, and MSDB shards running beside G3/A14 would slow its data loader. Run MSDB as one process overnight while the ablation runs (allowed under the hard rules), so the CPU battery is done before `GO_AHEAD`.

## 10. NOTE (pass): the displays are rigid rotations about the figure centre; none dropped

- Inventory (`checks/design_checks.json`): 480 PNG, 240 AOI files, `items.json`, `contact_sheet.png` = 722 files, exactly the manifest's list, all 722 md5 sums match. The expected names for 60 configurations x 4 angles x 2 conditions: none missing, none extra. `refused` = 0, `failures` = 0; render log "wrote 722 files"; generator md5 in the manifest equals the current `render_rigid.py`. The three `stimcode/` copies are byte-identical to `looking-at-nothing/stimuli/`.
- Geometry: inducer centres in `items.json` equal the design (square side 190, triangle side 230, positions, clockwise rotation about (cx, cy)) to 4.8e-5 px (4-decimal rounding). Notch directions equal atan2 towards the centroid plus 0 or 180 exactly. The vertex centroid equals the configuration centre for both shapes.
- Rotation test, independent of `render_rigid.py` (`checks/design_checks2.json`, 120 displays): I re-rendered each stim_v3 display at 4x with the stim_v3 renderer, rotated it bicubically by the stated angle about (cx, cy), and downsampled. Mean absolute difference to the rigid PNG was at most 0.091 grey levels. Rotating in the opposite direction gave at least 4.4 (at least 121 times larger) wherever the shape's symmetry distinguishes the two directions (all but square 45 degrees). The largest single-pixel differences (up to 112; at most 215 pixels above 64 per display) sit at the corners where a notch edge meets the arc (114 of 120 cases) or at the apex (4): PIL rasterisation of pie corners, not geometry. Illusory and control180 pass alike, so inducer centres and notch directions rotate together in both.
- Notch from the pixels (ring at 0.6 r, 1,680 inducers): direction error at most 2.6 degrees; the same method on stim_v3 gives 3.1 (for support ratio 0.7: 0.9 against 1.5).
- Edge orientation: see finding 1 (rig theta illusory against stim_v3 mis theta r >= 0.907; same rotation direction).
- Visual: the contact sheet and `crops_v3ill_mis20_rig20ill_rig20ctrl_v3ctrl.png` show aligned notches on rotated figures, outward notches on the rotated controls, and stim_v3 mis20 with the same notch-edge slant but misaligned. `square_sr70_dark_p1_rig45_control180.png` viewed at full size: clean, inside the canvas.

## 11. NOTE (pass): pixel statistics match stim_v3

All 480 files: RGB, 1024 x 768, R = G = B, background mode 255 (dark) or 0 (light), every border pixel equals the background. Ink mass (sum of |pixel - background| / 255) is within -0.24% to +0.06% of the stim_v3 display. Pixels at least 128 from background: -0.56% to +0.35%. Fully inked pixels: -0.65% to +0.51%. Antialiased pixels (21 to 234 grey levels from background): +18% at most against the axis-aligned stim_v3 displays, as expected for oblique edges, but -1.7% to +1.0% against stim_v3 mis20 at the same edge slant. Distinct grey levels: 155 to 233 (stim_v3 140 to 223). Supersampling (4x, LANCZOS) and Pillow output are confirmed by the manifest's 120 of 120 pixel-identical phi = 0 renders.

## 12. NOTE (pass): AOIs recomputed from geometry

On 80 rigid AOI files (20 configurations x 4 angles): the disc AOI equals my analytic recomputation exactly. `inside` equals (PIL polygon raster of the rotated vertices) minus discs exactly. `core30`, `core45` and `inside_eroded` are exact. An independent half-plane raster differs from `inside` on at most 389 pixels, all within 2 px of the polygon edge: PIL's half-pixel fill convention, which the published stim_v3 AOIs share. `polygon` equals the half-plane raster exactly. Intersection over union with the phi = 0 AOI rotated by nearest neighbour is at least 0.9887 (inside) and 0.9845 (discs). Inside area changes from -0.77% to +0.38% of phi = 0; this shifts a uniform map's r by about 0.2 percentage points, which cancels within E_rig because both rigid conditions share the AOI. At most 14 ink pixels fall outside the 1-px-dilated discs (the documented 0.38 px drawing offset). `eval_re.masks_for` will accept these files: config keys are shared by the two conditions of an angle, and the stored cores equal its own construction.

## 13. NOTE (pass): the scoring path equals the published pipeline

- DeepGaze IIE and MSDB: `score_rigid.py` calls `gate_kanizsa.load_models`, which returns `torch.exp(log density)` with a uniform centre bias (IIE at full resolution; MSDB at half resolution, `pixel_per_dva = 17.5`, `dataset=None`, resized back). The project-root and repository versions are identical for both models. Measures are `common_alt.q16` and `common_alt.measures`, verbatim. The published stim_v3 values (`raw_v3maps`) were measured on the saved float16 `sal / max` maps, which is the same quantisation. Threads are reset after `load_models`.
- Sweep readouts: `sweep.feats`, the trained `readout_L<L>.pt`, sigmoid, cv2 resize, centre bias off, as `score_alt.py`. Swap and registers heads: `eval_re.py` follows the published `eval_full.py` exactly (sigmoid at s x s, float16, cv2 resize to 1024 x 768, v2 and v3 measures).
- Nulls: `maps()` verbatim, with a 1e-9 reproduction check on 8 stim_v3 displays.
- Float16 feature cache: RUN_PLAN R4 and R5 run the backbones on the CPU for `stim_rigid` and say why; the cache (`v2/readout/cache/stim`, stim_v2 and stim_v3 only, both sizes, four feature sets) is used only for stim_v3 in M1 and M2. Correct.
- Pipeline checks before scoring: 4 stim_v3 displays at 1e-4 (IIE, MSDB, sweep), 18 displays per feature set at 1e-4 against the MPS values (heads), and 8 displays at 1e-9 (nulls). These are adequate; the heads-only probe already reproduced MPS r to 6.3e-6.

## 14. NOTE: G3/A14 recipe parity, the MSDB tensors, dropout grouping

- `train_re.py` and `stats_re.py` differ from `nn/registers/train_reg.py` and `v3/gpu/stats_full.py` only in the import line, and `train_reg.py` differs from `v3/gpu/train_full.py` only in its import. The recipe is therefore identical to the swap and registers runs: loss `AB.loss_A` (KL + 2(1 - CC)), 12,803 maps (`train_items`), 336 px, 6 epochs, batch 32 (last incomplete batch dropped), AdamW 5e-4 / 1e-4, cosine, final epoch, no augmentation; `torch.manual_seed(seed)` before each head, `RandomState(seed)` batch order, all heads of a seed in one frozen forward pass. Seeds are 0, 1, 2. R1n gets its own statistics pass (`stats_re.py`), as in the swap.
- `dinov2ms` uses the same model load (`AutoModel`, `facebook/dinov2-base`), preprocessing (INTER_AREA squash, ImageNet normalisation) and `interpolate_pos_encoding=True` path as the published `dinov2`; `dinov2reg` is the registers run's path unchanged. Eval discovery cannot confuse `dinov2ms_L11_R*` with `dinov2ms_L7L11_R*`.
- DeepGaze MSDB hooks `dinov2_vitb14.blocks[10]` and `blocks[6]` (`deepgazemsdb.py` lines 325 to 330; outputs before the final norm, CLS dropped). These are HF `hidden_states[11]` and `[7]` of the same weights, so the layer identity in rule 2.5 is right. MSDB, however, reads them from the torch-hub port at aspect-preserving 512 x 384 (interpolated to multiples of 14) and adds CLIP RN50x64 features, while A14 squashes to 336 x 336. In P6 wording, write "the two DINOv2-B blocks that DeepGaze MSDB reads" and state the input geometry; "the tensors MSDB reads" overstates.
- Dropout: in G3, R2 and R3 draw dropout in the same per-batch order as the published R2-R3 job, provided R1n and R1 draw none (true by code: the pointwise readout has no dropout). In A14 all five readouts share one job, so the R2, R3 and R4 dropout masks differ from a split job. This is harmless (a different random draw, same recipe), but RUN_PLAN should say so rather than imply identity.

## 15. NOTE: GPU time estimates against the logs

From the logs: plain DINOv2 jobs ran 517 to 542 s per epoch with 6 heads (R1n, R1, R4 x 2 blocks) and 452 to 516 s with 4 heads (R2, R3 x 2). The registers R4 pair ran 443 to 497 s per epoch (2,798 to 2,830 s per seed). Registers eval took 146 to 158 s per seed; the plain statistics pass took 404 s. The backbone pass dominates. My estimates: G3, 8 heads, about 490 to 560 s per epoch, so 49 to 56 min per seed plus about 4 min of eval: 2.9 to 3.3 h with stats and gate. A14, 10 heads including three ASPP heads on 1,536 channels, about 560 to 700 s per epoch: 3.2 to 4.0 h. RUN_PLAN's 3.4 h + 4.0 h is plausible to slightly conservative. Add at least 60 s per step for `wait_idle`. Neither run can start before `REANALYSIS_RULE.sha256`, `PUBLIC_PUSH.txt` and `GO_AHEAD` exist. The idle check is broken (finding 2), so the manual `GO_AHEAD` is today the only effective gate.

## 16. NOTE: disk

Free space was 28 GiB at 10:20 UTC and 27 GiB at 10:42 UTC: the ablation is writing. Its folder is 16 GB, and its own cap is 24.6 GB (`ABL_MAX_KB`), so it may take up to about 8.6 GB more, leaving at least about 18 GiB when G3/A14 start. Projected new use, from the measured analogues (`v3/gpu/full/state`: 90 MB per 6-head DINOv2 state, 108 MB per 4-head ASPP state; checkpoints about 12 MB per ASPP head): G3 about 0.45 GB, A14 about 1.2 GB, CPU scores and checks under 0.2 GB, rigid stimuli plus finding 1's controls under 25 MB, optional rigid maps (finding 8a) 0.5 to 1 GB. Total at most about 3 GB, inside RUN_PLAN's 4 to 5 GB budget and above both stop lines (10 GB in `run_gpu_re.sh`, 8 GiB in the ablation runner).

## 17. NOTE: MPS, downloads, writes outside `neuro/`

- MPS: in `neuro/reanalysis`, MPS appears only in `run_gpu_re.sh` and in `train_re.py`'s RNG state lines, both behind the gates. The CPU paths import the stub first (`score_rigid.py`; `recommon.py` with `V3_DEVICE=cpu`, asserted). Because `ab_objective` is cached in `sys.modules`, the later `sys.path.insert(1, ROOT)` in `score_rigid.py`, which in fact puts ROOT ahead of the stub despite its comment, cannot load the original; `sweep.AB is AB` is asserted. My probe (stub, transformers `from_pretrained` offline, `import deepgaze_pytorch`) held no Metal client. Keep the ioreg check after every start.
- Downloads: every weight the CPU battery loads is cached: `~/.cache/torch/hub/checkpoints` (`deepgaze2e.pth`, `deepgazemsdb.pth`, `densenet201`, `efficientnet-b5`, `resnext50_32x4d`, ShapeNet-C, `dinov2_vitb14_pretrain.pth`), hub repos pinned to refs (`pytorch/vision:v0.6.0`, `facebookresearch/dinov2:6a62615`, both cached and trusted through their cached folders, although `trusted_list` is empty), `~/.cache/clip/RN50x64.pt`, and the HF caches for `dinov2-base`, `dinov2-with-registers-base`, `siglip-large-patch16-256` and `clip-vit-large-patch14`. HF offline flags are set in `score_rigid.py`, `recommon.py` and `run_gpu_re.sh`.
- Writes: all outputs go under `neuro/reanalysis`. Import-time `os.makedirs` calls in read-only modules (`gate_kanizsa.py`: `gate/stimuli`; `sweep.py`: `sweep/<bb>`; `v2/readout/common.py`: cache, ckpt, maps, scores, logs) target folders that all exist, so nothing is created. Byte-code writing is disabled before project modules are imported. The planned copies for T1 and T2 are not written yet. `nn/registers/norm_diag.py` imports `regcommon`, which loads the original `ab_objective` (Metal query), so the T2 copy must import `recommon`. `run_vit.py` takes its output folders from `common_enc` (`TOK`, `LOGS` inside `looking-at-nothing`), so the T1 copy must redirect both. Write this into RUN_PLAN T1 and T2.
- Scoring before the push: see finding 4.

## 18. NOTE: small consistency items

1. `fix_rule.py` hashes `_stub/ab_objective.py` and `fix_rule.py`, which rule Section 10 does not list; list them, plus the files added by finding 1.
2. "Whole-display rotation" means the inducer configuration rotated about the figure centre on an unchanged canvas. With a uniform background this differs from rotating the image only at the canvas frame. Say so once in Methods.
3. RUN_PLAN R4a leaves the SigLIP and IIE `--tags` as placeholders; name them (for example `siglip_L24_R4_s0,siglip_L24_R1n_s0` and `iie_shapenetc_R4_s0,iie_shapenetc_R1n_s0`) so the check is reproducible.
4. `PUBLIC_PUSH.txt` is written by hand. Record the remote URL of the commit with the hash, so the paper's "public before re-scoring" can be checked by a reader.
5. P2 (not in this review's scope) compares alt20, whose notch edges slant both ways, with mis20, whose edges slant one way. The edge-orientation sets therefore match in magnitude, not in sign. That is acceptable for squares (mirror pairs), but it is worth one sentence in Methods next to the finding 1 text.
