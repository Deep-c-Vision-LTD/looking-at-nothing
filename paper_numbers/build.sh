#!/bin/bash
# Every number, table and figure of the manuscript from the result files in results/ (no model is run):
# seed-bootstrap check, numbers, extra numbers (parameter counts), tables, figures.
# Outputs: paper_numbers/output/ (or $LAN_PAPER_OUT): numbers.tex, numbers_extra.tex, tab_readout.tex, tab_survey.tex,
# tab_sweep.tex, make_numbers.log (macro, value and source of every number) and figs/.
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
OUT=${LAN_PAPER_OUT:-output}
mkdir -p "$OUT"
$PY check_seed_bootstrap.py > /dev/null
$PY make_numbers.py > "$OUT/make_numbers.log"
grep '^# ' "$OUT/make_numbers.log" | head -5
$PY make_numbers_extra.py > /dev/null
$PY make_figures.py
echo "written to $OUT"
