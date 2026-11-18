# nn/ run log

## 2026-11-17: full run of nn/run_all.sh

- Started 08:08:00 UTC, finished 09:30:09 UTC (1 h 22 min wall time), exit 0. Command: `bash nn/run_all.sh` (internal `caffeinate -i`, single instance), run in the background with the log monitored. All steps on the CPU; no MPS job; one heavy job at a time.
- Rule hash at start and end: 8e59fe87433a101aba97c99bca8e37a9e36faf08b47c9a70b1c6a141de3b403a (unchanged; ANALYSIS_RULE.md still read-only).
- Step times: scoring dinov2 L4/6/10 342 s, clip L8/16/20 895 s, siglip L12/16/20 786 s; ERF dinov2 426 s, clip 1294 s, siglip 1036 s; analysis 129 s.
- Fixes: none. No script was edited; no step failed or was rerun.
- Post-run sanity: DINOv2-B L12 rescored again with nn/score_nn.py into nn/results/sanity_post (CPU, 3 threads, 299 s) and compared by nn/sanity_check.py: 1,100 images x 2 sizes bit-identical in all three score files (max |diff| 0.0) and all 357 summary numbers identical (max |diff| 0.0); pass. The rescored files are also byte-identical to nn/results/sanity.
- Completeness: 9 new readouts x 2 sizes x (740 + 480 + 360) records, all present, no NaN/inf/None; ERF JSON for 18 readouts (24 displays, both sizes, readout and block-token, all/inside/inducer sets), no NaN, every per-display CDF monotone and ending at 1; summary.json has no NaN/Infinity tokens.
- Leftovers: nn/results/erf/<bb>_partial.npz (resume checkpoints, 26.7 MB each) are now redundant with <bb>_cdfs.npz; kept (nothing deleted).
- Note: the Hugging Face cache gained models--facebook--dinov2-with-registers-base at 08:21 UTC during this run. This run did not do it: every process ran with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1, and no nn/ or sweep code names that model. Another session was active on the machine at the time.
