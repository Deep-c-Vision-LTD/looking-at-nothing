"""
fix_rule.py: record the SHA-256 of REANALYSIS_RULE.md and of every file fixed with it (rule Section 10), with the UTC
time, in REANALYSIS_RULE.sha256. Run once, after the author approves the rule and before the files are pushed.
Refuses to overwrite an existing REANALYSIS_RULE.sha256 (move it to ../_trash first if the rule is re-fixed before
it was made public, and say so in the deviations file).
After the public push, write the commit hash (first token) and its URL to PUBLIC_PUSH.txt, for example:
  echo "<full commit hash> https://github.com/Deep-c-Vision-LTD/looking-at-nothing/commit/<full commit hash> pushed <UTC time>" > PUBLIC_PUSH.txt
Revised 18 November 2026 after the rule reviews: the list now includes the review notes, the responses, the feature-path
check and its output, and the render log (rule Section 10).
Usage: python fix_rule.py
"""
import glob, hashlib, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = (["REANALYSIS_RULE.md", "REVIEW_RESPONSES.md", "reviews/stats_review.md", "reviews/design_review.md",
          "analyze_re.py", "test_analyze_re.py", "tests_output.txt", "test_features_cpu.py", "tests_features_output.txt",
          "extract_known.py", "known_values.json", "render_rigid.py", "stim_rigid_render.log", "stim_rigid/manifest.json",
          "RF_RULE_CORRECTIONS.md", "RUN_PLAN.md", "_stub/ab_objective.py", "fix_rule.py"]
         + sorted(os.path.relpath(p, HERE) for pat in ("stimcode/*.py", "gpu/*.py", "gpu/*.sh", "scoring/*.py")
                  for p in glob.glob(os.path.join(HERE, pat))))


def main():
    out = os.path.join(HERE, "REANALYSIS_RULE.sha256")
    if os.path.exists(out):
        sys.exit("refusing: REANALYSIS_RULE.sha256 exists")
    missing = [f for f in FILES if not os.path.exists(os.path.join(HERE, f))]
    if missing:
        sys.exit(f"refusing: missing {missing}")
    lines = [f"{hashlib.sha256(open(os.path.join(HERE, f), 'rb').read()).hexdigest()}  {f}" for f in FILES]
    lines.append(f"recorded_utc {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
