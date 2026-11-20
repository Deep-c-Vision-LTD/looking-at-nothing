# Looking at Nothing

Code, stimuli, analysis rules, per-image scores and result files of:

> Hamit Soyel. **Looking at Nothing: On Six of Eight Frozen Feature Sets, No Valid Saliency Readout Changes the Sign of
> the Kanizsa Contrast.** Submitted, 2026.

```bibtex
@unpublished{soyel2026lookingatnothing,
  author = {Hamit Soyel},
  title  = {Looking at Nothing: On Six of Eight Frozen Feature Sets, No Valid Saliency Readout Changes the Sign of the Kanizsa Contrast},
  note   = {Submitted},
  year   = {2026}
}
```

## The result in one paragraph

Saliency models that read frozen features through a learned readout can be dissected by exchanging parts. Of four
saliency models, only **DeepGaze IIE** (**+0.098** [0.072, 0.124]) and **DeepGaze MSDB** (**+0.075** [0.056, 0.095])
place more saliency inside an illusory Kanizsa figure than when its inducers are rotated by 180 degrees, and a blurred
Harris corner detector gives more (**+0.176 to +0.241**): local structure can produce the inside contrast, so the
contrast alone is not evidence of figural processing. Both DeepGaze models lose figure mass under a uniform 20 degree
inducer rotation, unlike all 15 local nulls; a whole-display rotation control with orientation-matched displays could
not confirm, for either model, that this loss depends on inducer alignment. On squares, alternating the rotation
direction keeps more figure mass than rotating every inducer the same way (**+0.024** [+0.006, +0.043] for IIE,
**+0.021** [+0.010, +0.031] for MSDB), unlike all 15 local nulls, for which the two rotations leave the same mass. Five
readout architectures, from pointwise to full atrous spatial pyramid pooling, were trained with one loss on the same
12,803 natural-image saliency maps over eight frozen feature sets, six of them from DINOv2-B. On **six of eight** sets,
no readout that met the accuracy criterion changed the sign of the reference readout (**16 of 16** cells after Holm
adjustment; on four, only non-pointwise readouts qualified); four of the six sets re-test values known when the rule was
written, and of the four sets nobody had seen, the readout changed the sign on two. On those two, block 12 of DINOv2-B
with registers and blocks 7 and 11 of public DINOv2-B, one valid pointwise readout with image-wide normalisation (R1)
gave a contrast of the opposite sign (unadjusted intervals with the trained heads fixed: **-0.035** [-0.069, -0.001] and
**+0.020** [+0.006, +0.034]): the readout architecture can change the sign on these features. Both changes held against
the 180 degree control only, and with seeds resampled (exploratory) both intervals included zero. The re-analysis rule
fixed in advance how this test would set the title; its narrowing branch applied, so the claim is restricted to the six
sets. Within public DINOv2-B, block 6 is positive (**+0.090**) and blocks 11 and 12 are negative (**-0.095** and
**-0.114**) under every valid readout. Under its fixed rule, the receptive-field test of rotation sensitivity was
inconclusive (rho = -0.37, p = .218).

E, the inside contrast, is r(illusory) - r(180 degree control), where r = saliency mass inside the figure / (mass
inside + mass on the inducer discs). Brackets are 95% intervals: published two-stage percentile intervals for the
survey, the wild cluster percentile-t intervals of the re-analysis rule for the rotation and readout-swap values. A
readout is valid when its held-out MIT1003 correlation meets the accuracy criterion fixed on 15 October 2026 (R4's mean
minus 0.03).

## Analysis rules and timing

- **Re-analysis rule** (`neurocomputing/REANALYSIS_RULE.md`): a fixed analysis rule, made public before any
  re-scoring. Its SHA-256 (b03f397b6b1808e7055f0d6db8f5685c0e6a2850261eeb3d8525dddbeca73a38), with those of 33 files in
  all, was recorded at 11:29:31 UTC on 18 November 2026 (`neurocomputing/REANALYSIS_RULE.sha256`); the rule was pushed to
  this repository in commit f67c7df at 11:36:31 UTC (`neurocomputing/PUBLIC_PUSH.txt`), before the sanity gate
  (11:41:48 UTC) and the first re-scoring (11:53:24 UTC). It fixes the interval and test methods, the multiplicity
  control, the primary contrasts P1 to P6, the wording of every outcome and the title decision. P1 to P4 re-test values
  that were known when it was written; only P5 and P6 test outcomes that nobody had seen. Every scoring and statistics
  script of the re-analysis pipeline calls `require_public()`, which refuses to run unless the rule and its hashed files
  are unchanged; the independent recomputations (`neurocomputing/verify_rescore/`,
  `neurocomputing/results/p6_verification/`) do not.
- **Title decision.** The rule fixed three outcomes: a stronger title if every valid readout on the new feature sets kept
  the reference sign, a scoped title otherwise, and a narrowed claim and title if any valid readout had an interval of
  the opposite sign. The narrowing branch applied: not every valid readout on the new sets kept the reference sign
  (condition (c)), and R1 had an interval of the opposite sign on block 12 of DINOv2-B with registers and on DINOv2-B
  blocks 7 and 11 (condition (d)).
- **Receptive-field rule** (`property_package/ANALYSIS_RULE.md`): a fixed analysis rule, not public before scoring. Its
  SHA-256 (8e59fe87433a101aba97c99bca8e37a9e36faf08b47c9a70b1c6a141de3b403a) was recorded at 07:53:35 UTC on 17
  November 2026, three readings it leaves open were fixed in the analysis code at 08:01 UTC, and scoring and every
  receptive-field computation ran from 08:08 to 09:30 UTC the same day. The repository had last been pushed on 18
  October 2026, so this timing rests on local file records. When the rule was written, the aligned effects of all 18
  sweep readouts were known (scored when trained on 8 October 2026 and pushed on 9 October 2026; they are in
  `legacy/results/layers/`); not known were any receptive field and the rotation, core and real-figure scores of nine
  readouts. The corrections of its preamble (C1 to C3), of two documents that repeat its error (C4, C5), of its code
  (C6) and of one reporting label (C7) are in `property_package/RULE_CORRECTIONS.md`. The result is case I-d:
  inconclusive.
- **Deviations.** Every departure from the re-analysis rule (D1 to D14) is logged in the deviations log next to the
  rule (`neurocomputing/DEVIATIONS.md`), with what changed, why, what had been seen and its effect; none changes a
  decision. A later entry, the D8 correction, corrects statements of D8 (bookkeeping only). D14 adds the continuity
  intervals (the earlier pipeline's two-stage percentile method; for the 18 P4 ratio differences a composition of that
  procedure, not a published computation; no ratio-type estimate has a published-method p) of the primary estimates
  that the earlier pipeline never computed; they were computed after P1 to P6 had been seen and are reporting only.

## Reproduction

Python 3.12. Steps 1 to 4 run from a clone, from the repository root; locations are set in `paths.py`
(repository-relative defaults, environment variables for external data and weights). Steps 5 to 7 are documented as
they were run and do not run from a bare clone (see before step 5).

```bash
git clone https://github.com/Deep-c-Vision-LTD/looking-at-nothing && cd looking-at-nothing
python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
```

**1. Stimuli.** The generators reproduce `stimuli/` byte for byte; the verifiers re-render every display and check the
areas of interest. The whole-display rotation set and its orientation-matched controls are rendered by
`neurocomputing/render_rigid.py` and listed with their md5 in `neurocomputing/stim_rigid/manifest.json`.

```bash
python stimuli/stimuli_v3.py && python stimuli/verify_stim_v3.py     # 740 displays + AOIs  -> stimuli/stim_v3
python stimuli/stimuli_alt.py && python stimuli/verify_stim_alt.py   # 360 displays + AOIs  -> stimuli/stim_alt
(cd stimuli && shasum -a 256 -c SHA256SUMS)
```

**2. Data, weights and the training cache:**

```bash
export SAL_DATA=/path/to/data        # mit1003/{ALLSTIMULI,ALLFIXATIONMAPS}, salicon/{images/train,train}
export CAT2000=/path/to/cat2000      # Stimuli/, FIXATIONMAPS/, FIXATIONLOCS/ (default: $SAL_DATA/cat2000)
curl -L -O https://github.com/Deep-c-Vision-LTD/looking-at-nothing/releases/download/v1.0/layer_readouts.zip
unzip layer_readouts.zip             # -> weights/layers/<dinov2|clip|siglip>/readout_L<L>.pt (18 layer-sweep readouts)
python models/ab_objective.py cache  # -> work/train_cache/{meta.json, maps.npy}: the 12,803 training maps
```

Pretrained backbones (facebook/dinov2-base, openai/clip-vit-large-patch14, google/siglip-large-patch16-256,
alexanderkroner/MSI-Net) and the DeepGaze checkpoints download on first use in steps 3 and 4; those scripts default to
offline mode once the models are cached, so run the first job with `HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0`. The
scripts of steps 5 to 7 force offline mode, `property_package/run_all.sh` stops unless DINOv2-B, CLIP-L and SigLIP-L
are in `$HOME/.cache/huggingface/hub`, and no earlier step loads facebook/dinov2-with-registers-base, so fetch the four
Hugging Face backbones once beforehand:

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "from transformers import AutoModel as M; [M.from_pretrained(m) for m in
  ('facebook/dinov2-base', 'facebook/dinov2-with-registers-base', 'openai/clip-vit-large-patch14', 'google/siglip-large-patch16-256')]"
```

**3. Survey, local nulls, rotations and tokens** (per-image scores, then statistics; CPU, 3 threads per process): steps
4 and 5 of the earlier version, unchanged.

```bash
python scoring/score_v3.py edge sr iie msdb          # stim_v3 -> results/v2/scores/raw/<tag>.json
for b in dinov2 clip siglip; do python scoring/score_v3.py sweep:$b; done     # one process per backbone
python scoring/score_v3.py aspp ml                    # xSoyelNet-ASPP and -ML: needs their weights (not distributed)
python scoring/run_msinet_v3.py                       # in the TensorFlow 2 environment (requirements.txt), then in .venv:
python scoring/score_v3.py msinet
python scoring/local_nulls.py                         # 15 local nulls on stim_v3 -> results/v2/verify
python scoring/fill_contrast_iie.py                   # DeepGaze IIE at 25 % and 75 % fill contrast
python scoring/score_alt.py iie msdb                  # stim_alt -> results/v3/scores/raw/<tag>.json
for b in dinov2 clip siglip; do python scoring/score_alt.py sweep:$b; done
python scoring/measure_v3maps.py                      # core measures from the saved stim_v3 maps -> results/v3/scores/raw_v3maps
python scoring/local_nulls_alt.py                     # local nulls on stim_v3 and stim_alt
python analysis/analyze_v3.py && python analysis/analyze_alt.py
python analysis/local_nulls_stats.py && python analysis/position_check.py && python analysis/fill_contrast_stats.py
CORE_D=30 CORE_OUT=results/v2/verify/core_aoi_D30_835295.json python analysis/core_aoi.py DeepGaze_IIE DeepGaze_MSDB \
    xSoyelNet-ASPP xSoyelNet-ML MSI-Net Edge-energy_null L:dinov2_L12 L:dinov2_L8 L:siglip_L24 L:clip_L12
for n in dinov2 clip siglip; do for s in 336 448; do python analysis/encoding/run_vit.py $n $s; done; done
python analysis/encoding_nulls.py                     # readout-free token encoding against a near-exterior null
```

The paper numbers and smoke test of the earlier version, its Shapley decomposition, the evaluation of the v2.0
checkpoints without training and the training recipes of the reused readouts are in the README of commit 6ea8ddc
(steps 1, 6, 7 and 8 there).

**4. Published readout swap** (60 readouts; release v2.0 holds the checkpoints):

```bash
V3_DEVICE=mps readout_swap/run_full.sh      # or cuda / cpu: statistics, training, evaluation, results_full.json
```

**Steps 5 to 7** ran in the project tree of the study and do not run from a bare clone. In that tree
`property_package/` is `nn/`, `registers/` is `nn/registers/` and `neurocomputing/` is `neuro/reanalysis/`; their
scripts import code and read displays and per-image scores (plain JSON) from its `v2/` and `v3/` folders, and import
`sweep.py` from its root. `stimuli/` and `results/` of this repository hold those displays and score files (the scores
as `.json.xz`), in a different layout. `property_package/` and `registers/` locate the tree from their own folder; the
scripts of `neurocomputing/` read `LAN_PROJECT_ROOT`, and `scoring/score_rigid.py` loads the DeepGaze models through
`models/gate_kanizsa.py` of this repository (`LAN_REPO`, default `$LAN_PROJECT_ROOT/looking-at-nothing`). The shell
runners take their interpreter from `PY` (default: `saliency_research/.venv/bin/python` in the folder that holds the
project tree). The commands below use the folder names of this repository.

```bash
export PY="$(command -v python)"                     # the active .venv, for the shell runners
export LAN_PROJECT_ROOT=/path/to/project_tree       # read by the scripts of neurocomputing/ (step 7)
```

**5. Receptive-field test** (`property_package/`, CPU). `run_all.sh` scores the nine readouts not yet scored on the
rotation, core and real-figure displays, computes the effective receptive field of all 18 sweep readouts and runs the
analysis fixed by the rule (`results/summary.json`, `results/report.md`); `property_results.py` re-derives its
decision-bearing numbers and writes the report (`results/property_results.json`, `PROPERTY_RESULTS.md`,
`figs/erf_vs_rotation.{pdf,png}`). Scoring skips displays already in `results/raw_*`, and the runner skips the
receptive field of every backbone whose `results/erf/<bb>.json` exists, so with the distributed files in place only
the analysis reruns; move those files aside for a full rerun. The per-display receptive-field arrays are in release
v3.0.

```bash
bash property_package/run_all.sh                     # scoring, receptive fields, analysis; interpreter from PY
python property_package/property_results.py          # checks summary.json; report, property_results.json, figure
```

**6. Registers control** (`registers/`; the diagnostic on the CPU, the rest on Apple MPS, which `run_reg.sh` selects):
R4 on blocks 6 and 12 of DINOv2-B with registers, seeds 0, 1 and 2, with the published recipe
(`registers_results.json`, `REGISTERS_RESULTS.md`), and the token-norm diagnostic (`diag/norm_diag.json`), which the
report reads. The six R4 heads are in release v3.0.

```bash
V3_DEVICE=cpu python registers/norm_diag.py          # token-norm diagnostic -> registers/diag/norm_diag.json
bash registers/run_reg.sh                            # training, evaluation, statistics, report; interpreter from PY
```

**7. Re-analysis** (`neurocomputing/`). Rule Section 8.4 fixes the order: sanity gate; P1 to P4; exploratory analyses
from saved outputs; rigid-rotation scoring and P5; GPU runs and P6; the title decision last. The commands below follow
it for the primary steps and list the exploratory scripts last, because some of them read the rigid-rotation scores or
the G3 and A14 outputs; `explore/README.md` gives the inputs of each, and the deviations log records where the run
departed from the rule's order (D3). Run one process at a time; the GPU runner needs Apple MPS and aborts without it.
The per-image score files are distributed as `<name>.json.xz` and no script in `neurocomputing/` reads them in that
form: decompress them first (`xz -dk`, as the header of `neurocomputing/XZ_ORIGINALS.sha256` says); the first command
after `cd neurocomputing` below does this (with `-f`, so that it can be rerun) and checks every restored file against
its SHA-256.

```bash
cd neurocomputing
find . -name "*.json.xz" -exec xz -dkf {} + && grep -E "^[0-9a-f]{64}  " XZ_ORIGINALS.sha256 | shasum -a 256 -c --quiet
python -I analyze_re.py sanity                       # must print PASS (rule Section 3.1)
python -I analyze_re.py primary --which P1,P2,P3,P4
python -I scoring/local_nulls_rigid.py && python scoring/score_rigid.py iie
for b in dinov2 clip siglip; do python scoring/score_rigid.py sweep:$b; done   # one process per backbone (D1)
python scoring/score_rigid.py msdb
python -I analyze_re.py primary --which P5
touch gpu/GO_AHEAD                                   # go-ahead marker the GPU runner requires (no other GPU job running)
bash gpu/run_gpu_re.sh both                          # G3 (registers) and A14 (blocks 7 and 11); interpreter from PY
python -I analyze_re.py primary --which P6           # P6 and the title decision
python explore/explore_saved.py                      # and the other scripts of explore/ (README there)
```

The statistics engine (`analyze_re.py`, numpy and scipy only) is tested on synthetic data with known truth
(`test_analyze_re.py`, output in `tests_output.txt`). Every primary contrast (P1 to P6) was recomputed from the raw
per-image scores by code that imports no module of the engine, P6 three times, and every decision was reproduced: P1 to
P5 in `neurocomputing/verify_rescore/` (README there), P6 in `neurocomputing/results/p6_verification/`. The trained
heads of G3 and A14 are in release v3.0; its notes map each head to its P6 cell.

**Pitfalls.**
- Hugging Face `hidden_states[L]` is the output of block L (1-based) before the final layer norm. The layer sweep, the
  readout swap and the re-analysis read `hidden_states`. The feature sets `dinov2ms_L7L11` and `dinov2ms_L11` read
  `hidden_states[7]` and `[11]` (0-based blocks 6 and 10), the two DINOv2-B blocks that DeepGaze MSDB reads; MSDB reads
  them from the torch-hub port at an aspect-preserving 512 x 384 input with CLIP RN50x64 features, whereas these
  readouts see the display squashed to 336 and 448 px, without CLIP features.
- The registers checkpoint is read with the class token and the four register tokens dropped.
- DeepGaze MSDB: do not chain `.eval()` on the constructor; it runs on a 512 x 384 input at `pixel_per_dva = 17.5` with
  `dataset=None`. Both DeepGaze models get a uniform centre bias; every learned centre bias is switched off on the
  illusion displays, because figure position is varied.
- Per-image score files are stored as `<name>.json.xz` (`xz -dk` to read them by hand). Scripts that load them through
  `paths.py` read a plain `<name>.json` first, so a rerun replaces the distributed file without deleting it; the scripts
  in `neurocomputing/` read plain JSON only (step 7).
- Saliency maps, feature caches and training states are not distributed; analyses that read maps need the scoring step
  that writes them.

## Not redistributed

| What | Where to get it |
|---|---|
| MIT1003 (Judd et al., 2009) | https://people.csail.mit.edu/tjudd/WherePeopleLook/ |
| SALICON (Jiang et al., 2015) | http://salicon.net/ |
| CAT2000 (Borji & Itti, 2015) | http://saliency.mit.edu/results_cat2000.html |
| DeepGaze IIE and MSDB weights | downloaded by deepgaze_pytorch (https://github.com/matthias-k/DeepGaze) |
| MSI-Net weights | https://huggingface.co/alexanderkroner/MSI-Net |
| DINOv2-B (with and without registers), CLIP-L/14, SigLIP-L/16 | Hugging Face model hub (names above) |
| xSoyelNet-ASPP and xSoyelNet-ML weights | not distributed (commercial models of the author); their per-image scores are in `results/` |
| 18 layer-sweep readouts (302 MB) | release v1.0, `layer_readouts.zip` |
| 60 readout-swap checkpoints (591 MB, 548 MB zipped) | release v2.0, `readout_swap_checkpoints.zip`, to unzip into `results/v3/gpu/full/` |
| Heads of runs G3 (24) and A14 (30) and the registers R4 heads (6) (494 MB); per-display receptive-field arrays (23 MB) | release v3.0, `g3_heads.zip`, `a14_heads.zip`, `registers_r4_heads.zip`, `property_package_erf_cdfs.zip` (480 MB zipped), to unzip at the repository root |
| Saliency maps, feature caches, token caches, training states | regenerated by the scripts (tens of GB) |

The human values cited as motivation are the pooled dwell proportions of Kavšek, M. (2024). Perception of illusory
contours in children and adults: An eye-tracking study. *Attention, Perception, & Psychophysics, 86*(7), 2490-2503,
https://doi.org/10.3758/s13414-023-02832-z. No eye-movement data were collected for this study.

Narrative write-ups and planning notes of the author are not included. `neurocomputing/CLEANED_FILES.txt` (section 4)
lists the files cited in `neurocomputing/` but not published and where the numbers of those write-ups are; the README
of `neurocomputing/verify_rescore/` does the same for the write-ups of the verification.

## Repository layout

```
paths.py                 single configurable root (environment variables)
stimuli/                 generators, verifiers and the rendered Kanizsa, Ehrenstein and alternating-rotation sets
models/                  model code shared by every step (measure r, loaders, training set, layer sweep, MSI-Net)
scoring/                 per-image scores of models and local nulls
analysis/                survey, rotation, core-AOI, position, fill-contrast and token-encoding statistics
decomposition/           Shapley decompositions of the earlier version (not used by the present manuscript)
readout_swap/            published readout swap: 4 feature sets x 5 readouts x 3 seeds
property_package/        receptive-field test: rule, hash, run log, corrections, code, results, figure
registers/               registers control: code, registers_results.json, REGISTERS_RESULTS.md, diagnostics, logs
neurocomputing/          re-analysis
  REANALYSIS_RULE.md       the rule; REANALYSIS_RULE.sha256 (its hash record), PUBLIC_PUSH.txt (its public commit)
  reviews/                 design and statistics reviews of the rule, with their support scripts
  analyze_re.py            statistics engine; test_analyze_re.py and tests_output.txt
  render_rigid.py          whole-display rotation set (stim_rigid/, with manifest.json)
  scoring/, scores_rigid/  rigid scoring and its per-image scores
  gpu/                     registers and blocks 7 and 11 runs (G3, A14): code, logs, training records, scores
  out/                     sanity gate and P1 to P6 (JSON)
  results/                 exploratory and reporting outputs, run logs, P6 recomputations, script hash record
  explore/                 exploratory and reporting scripts (README there)
  verify_rescore/          independent recomputation of P1 to P5
  DEVIATIONS.md            deviations log
  CLEANED_FILES.txt        files changed for publication; XZ_ORIGINALS.sha256 (hashes of the decompressed scores)
paper_numbers/           numbers, tables and figures of the earlier version
results/                 result files of the earlier steps, in the tree that the source comments name
tools/smoke_test.py      quick end-to-end check of the earlier steps
legacy/                  first version of the study; legacy/results/layers/ holds the aligned effects of all 18
                         sweep readouts as scored on 8 October 2026
```

## Data and release

Release v3.0 is the version of the submitted manuscript. It is tagged on the commit that added this README and holds
the G3, A14 and registers R4 heads and the receptive-field arrays; its notes give the SHA-256 of every zip and of every
file in them. Releases v1.0 (layer-sweep readouts) and v2.0 (readout-swap checkpoints) hold the weights of the earlier
steps.

## License and data

Code: MIT (see `LICENSE`). Stimuli, per-image scores and results: CC BY 4.0. Benchmark images and fixation data are not
redistributed; only derived quantities (scores, statistics) are.
