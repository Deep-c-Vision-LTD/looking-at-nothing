verify_rescore/: independent recomputation of P1 to P5 and checks of the exploratory results

Code written for the verification steps of 18 and 19 November 2026, separately from the statistics engine
(analyze_re.py), with its outputs and logs.
- recompute_stats.py (P1 to P4) and recompute_p5.py (P5, the P5 categories and the rule 5.5 and 7.1 clauses) import no
  module of the project: loaders, design, wild cluster bootstrap, pairs cluster BCa, Holm and the decision logic are
  written from the rule text. Outputs recompute_stats.json and recompute_p5.json.
- compare_stats.py and compare_p5.py compare them with out/primary_P1_P2_P3_P4.json and out/primary_P5.json key by key
  (compare_stats.json, compare_p5.json). Every decision agrees; the only values that differ are the BCa and Holm
  p-values of three P4 contrasts (DEVIATIONS.md D4) and the informational rows of compare_p5.json.
- extra_checks.py and extra_checks2.py: seed signs of the P3 cells, WCR and WCU agreement, the accuracy criterion and
  the published 4-decimal values (extra_checks.json, extra_checks2.json; DEVIATIONS.md D4 and D5 cite them).
- rescore_probe.py, launched by run_probe.sh: CPU re-scoring of 20 random whole-display rotation displays with code
  written here (rescore_probe.json, rescore_probe_rigid.json, probe_run.log). r4a_recheck.txt: recheck of the CPU head
  check R4a (per file: number of values compared and the largest absolute difference).
- cpu_remaining/: recomputation of the exploratory steps S3, S4, S5 and of the test-time registers results from the
  raw score files (v_*.py and their JSON outputs; logs results/run_logs/VCR_*.log). These scripts import analyze_re
  only for require_public() (rule Section 8.1).
The three independent recomputations of P6 are in results/p6_verification/.

Inputs are listed in each script's header. Locations follow analyze_re.py: RE is the parent folder (neurocomputing/),
ROOT is LAN_PROJECT_ROOT (the project tree that results/ mirrors); run from this folder with the project's Python
3.12 interpreter. Score files stored as <name>.json.xz must be decompressed first (xz -dk).

Changed from the files the verification wrote (public copy, 20 November 2026):
- Scripts (8): the locations RE and ROOT (and NEURO, which they were built from) derived as above instead of
  from the layout neuro/verify_rescore next to neuro/reanalysis; in cpu_remaining/v_s4_a6.py the MIT1003 folder
  also honours LAN_SAL_SRC (as gpu/recommon.py); in run_probe.sh the absolute HERE and PY replaced as in
  gpu/run_gpu_re.sh. No other line changed.
    compare_p5.py
    compare_stats.py
    cpu_remaining/v_s4_a6.py
    cpu_remaining/vcommon.py
    recompute_p5.py
    recompute_stats.py
    rescore_probe.py
    run_probe.sh
- Logs (3): absolute paths replaced by $LAN_PROJECT_ROOT, $LAN_SAL_SRC and $HOME, as in
  ../CLEANED_FILES.txt section 1.
    compare_p5.log
    probe_run.log
    recompute_p5.log

Not published: the narrative write-ups VERIFY.md, VERIFY_P5.md and VERIFY_CPU_REMAINING.md (the "note" fields of
compare_p5.json cite VERIFY.md; the rows they mark are informational), the scripts that build or check narrative
text and their outputs:
    cpu_remaining/check_new_text_numbers.json
    cpu_remaining/check_new_text_numbers.py
    cpu_remaining/exploratory_v3_sections.md
    cpu_remaining/make_exploratory_v3_sections.py
    make_verify.py
    make_verify_p5.py
    md_check.json
    md_check.py
    verify_p5_summary.json
    verify_summary.json
