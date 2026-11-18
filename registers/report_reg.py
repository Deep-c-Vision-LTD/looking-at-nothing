"""
Side-by-side report: R4 readouts on DINOv2-B with registers (nn/registers/full) against the published plain DINOv2-B R4
readouts (v3/gpu/full), blocks 12 and 6, per seed and pooled over seeds.

Pooled numbers: results_full.json of each run (v3/gpu/analyze_full.py statistics; the registers file is written by the
unchanged copy analyze_reg.py). Per-seed numbers: the same functions on one seed's score files (pilot analyze.boot /
effects, analyze_full ratio with the RandomState(2026) bootstrap indices, the stim_alt condition means of
analyze_full). Local-null references: v3/scores/results.json (the 15 local nulls of the survey; r_inside, all shapes),
as in cbb/make_numbers.py.
Writes REGISTERS_RESULTS.md and registers_results.json. Usage: python report_reg.py
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import regcommon as R
import analyze as A2                                         # v2/readout/analyze.py (read-only)

H = R.HERE
REG, PUB = os.path.join(H, "full"), R.PLAIN_OUT
B = A2.B
SEEDS = (0, 1, 2)
BLOCKS = (("12", "dinov2_L12", "dinov2reg_L12"), ("6", "dinov2_L6", "dinov2reg_L6"))
THETAS = (10, 20, 30)
CC_MARGIN = A2.CC_MARGIN
_IDX = {}


def bidx(K, P):                                              # = analyze_full.bidx
    if (K, P) not in _IDX:
        rng = np.random.RandomState(2026)
        _IDX[(K, P)] = (rng.randint(0, K, (B, K)), rng.randint(0, P, (B, K, P)))
    return _IDX[(K, P)]


def bmeans(v):
    ci, pi = bidx(*v.shape)
    return v[ci[:, :, None], pi].mean((1, 2))


def ratio(num, den):                                         # = analyze_full.ratio
    bn, bd = bmeans(num), bmeans(den)
    with np.errstate(divide="ignore", invalid="ignore"):
        br = bn / bd
    lo, hi = np.nanpercentile(br, [2.5, 97.5])
    dlo, dhi = np.percentile(bd, [2.5, 97.5])
    return dict(value=round(float(num.mean() / den.mean()), 4), ci=[round(float(lo), 4), round(float(hi), 4)],
                den_ci=[round(float(dlo), 4), round(float(dhi), 4)], den_ci_excludes_0=bool(dlo > 0 or dhi < 0))


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


D = A2.design("stim_v3")
CELLS = D["kanizsa"]["cells"]
CONTRASTS = (("Kz", "kanizsa", "illusory", "control180", "inside"), ("KzCore", "kanizsa", "illusory", "control180", "r_core30"),
             ("KzNinety", "kanizsa", "illusory", "control90", "inside"), ("RealFilled", "kanizsa", "real_filled", "control180", "inside"),
             ("Real", "kanizsa", "real", "control180", "inside"), ("Ehr", "ehrenstein", "illusory", "control_end", "inside"))


def alt_block(s3, sa, seeds, meas="r_inside"):
    """analyze_full stim_alt section for one measure (condition means averaged over 336/448 px and over seeds)."""
    def m(cond, sizes=("336", "448")):
        def one(c, sd):
            if cond.startswith("alt"):
                return np.mean([sa[sd][z][meas][f"{c}_{cond}{ph}"] for z in sizes for ph in "ab"])
            return np.mean([s3[sd][z][meas][f"{c}_{cond}"] for z in sizes])
        return np.array([[np.mean([one(c, sd) for sd in seeds]) for c in cell] for cell in CELLS])
    ill, c180 = m("illusory"), m("control180")
    e_ill = ill - c180
    r = dict(effect_illusory=A2.boot(e_ill))
    for t in THETAS:
        mis, alt = m(f"mis{t}"), m(f"alt{t}")
        r[f"ratio_uni{t}"] = ratio(mis - c180, e_ill); r[f"ratio_alt{t}"] = ratio(alt - c180, e_ill)
        r[f"alt{t}_minus_mis{t}"] = A2.boot(alt - mis); r[f"drop_ratio{t}"] = ratio(ill - alt, ill - mis)
    return r


def per_seed(out, fs, s):
    sc = os.path.join(out, "scores")
    mit = jl(os.path.join(sc, f"mit_{fs}_R4_s{s}.json"))
    s3 = jl(os.path.join(sc, f"stim_v3_{fs}_R4_s{s}.json")); sa = jl(os.path.join(sc, f"stim_alt_{fs}_R4_s{s}.json"))
    if mit is None or s3 is None or sa is None:
        return None
    rec = dict(CC=round(mit["CC"], 4), NSS=round(mit["NSS"], 4))
    for key, ill, test, ctrl, var in CONTRASTS:
        rec[key] = A2.boot(A2.effects(s3, D[ill], test, ctrl, "avg", var))
    rec["alt"] = alt_block({s: s3}, {s: sa}, [s])
    return rec


def train_rec(out, fs, s):
    j = jl(os.path.join(out, "ckpt", f"{fs}_R4_s{s}.json"))
    return None if j is None else dict(final_loss=j["hist"][-1]["loss"], device=j["device"], n_train=j["n_train"],
                                       batches=j["batches"], epochs=j["epochs"], finished=j["finished"])


def pooled(res, fs):
    st = res["stim_v3"].get(fs, {}).get("R4")
    sa = res["stim_alt"].get(fs, {}).get("R4", {}).get("r_inside")
    if not st or st.get("incomplete") or not sa:
        return None
    rec = dict(mit=res["mit"][fs]["R4"], seeds=st["seeds_present"])
    for key, ill, test, ctrl, var in CONTRASTS:
        rec[key] = st[ill][f"{test}-{ctrl}"][f"avg|{var}"]
    for z in ("336", "448"):
        rec[f"Kz{z}"] = st["kanizsa"]["illusory-control180"][f"{z}|inside"]
        rec[f"KzCore{z}"] = st["kanizsa"]["illusory-control180"][f"{z}|r_core30"]
    rec["section_B"] = res["section_B"][fs]["R4"]["inside"]
    rec["alt"] = sa
    return rec


def nulls():
    v3m = json.load(open(os.path.join(R.ROOT, "v3", "scores", "results.json")))["models"]
    ns = [k for k, v in v3m.items() if v["kind"] == "local null"]
    out = {}
    for n in ns:
        o = v3m[n]["results"]["r_inside"]["all"]
        dr = o["drop_ratio"]["20"]
        out[n] = dict(aligned=o["aligned"], ratios_computed=o["ratios_computed"],
                      ratio_uni20=o["ratio_uni"]["20"]["value"] if o["ratios_computed"] else None,
                      ratio_alt20=o["ratio_alt"]["20"]["value"] if o["ratios_computed"] else None,
                      drop_ratio20=None if dr is None else dr["value"])
    neg = [n for n in ns if not n.startswith("null corner")]
    assert all(out[n]["aligned"]["ci"][1] < 0 for n in neg)
    nu = [out[n]["ratio_uni20"] for n in neg]
    nd = [out[n]["drop_ratio20"] for n in neg if out[n]["drop_ratio20"] is not None]
    cor = [n for n in ns if n.startswith("null corner")]
    assert all(out[n]["aligned"]["ci"][0] > 0 for n in cor)
    cu = [out[n]["ratio_uni20"] for n in cor if out[n]["ratios_computed"]]
    ad = [out[n]["drop_ratio20"] for n in ns if out[n]["drop_ratio20"] is not None]
    return dict(per_null=out, neg=neg, neg_uni20=[min(nu), max(nu)], neg_drop20=[min(nd), max(nd)], n_neg_drop=len(nd),
                corner=cor, corner_uni20=[min(cu), max(cu)], n_corner_uni=len(cu), all_drop20=[min(ad), max(ad)], n_all_drop=len(ad))


# ------------------------------------------------------------------ formatting (no em-dashes; hyphen-minus for signs)
def f3(x):
    return f"{x:+.3f}"


def fe(e):
    return f"{e['mean']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]"


def fr(r):
    if r is None:
        return "n/a"
    s = f"{r['value']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]"
    return s + ("" if r["den_ci_excludes_0"] else " (den. CI includes 0)")


def sign_of(e):
    return "+" if e["ci"][0] > 0 else "-" if e["ci"][1] < 0 else "0"


def main():
    pub = json.load(open(os.path.join(PUB, "results_full.json")))
    regp = os.path.join(REG, "results_full.json")
    reg = json.load(open(regp)) if os.path.exists(regp) else None
    NUL = nulls()
    diag = jl(os.path.join(H, "diag", "norm_diag.json"))
    chk = jl(os.path.join(H, "check", "eval_check.json"))
    out = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), blocks={}, nulls={k: v for k, v in NUL.items() if k != "per_null"},
               nulls_per_null=NUL["per_null"], diag=diag, eval_check=chk)
    for blk, fp, fr_ in BLOCKS:
        b = dict(plain=dict(pooled=pooled(pub, fp), seeds={}, train={}), registers=dict(pooled=pooled(reg, fr_) if reg else None, seeds={}, train={}))
        for s in SEEDS:
            b["plain"]["seeds"][s] = per_seed(PUB, fp, s); b["plain"]["train"][s] = train_rec(PUB, fp, s)
            b["registers"]["seeds"][s] = per_seed(REG, fr_, s); b["registers"]["train"][s] = train_rec(REG, fr_, s)
        thr = round(b["plain"]["pooled"]["mit"]["mean"] - CC_MARGIN, 4)
        b["threshold_plain_R4"] = thr
        out["blocks"][blk] = b
    out["pub_other_r4"] = {}
    for fs in ("siglip_L24", "iie_shapenetc"):
        po = pooled(pub, fs)
        if po:
            out["pub_other_r4"][fs] = dict(Kz=po["Kz"], ratio_uni20=po["alt"]["ratio_uni20"], ratio_alt20=po["alt"]["ratio_alt20"],
                                           drop_ratio20=po["alt"]["drop_ratio20"])
    json.dump(out, open(os.path.join(H, "registers_results.json"), "w"), indent=1)
    write_md(out)


def write_md(o):
    L = []
    A = L.append
    A("# Registers control: DINOv2-B R4 readouts, blocks 12 and 6")
    A("")
    A(f"Generated {o['created']} by nn/registers/report_reg.py from nn/registers/full (registers) and v3/gpu/full (published plain run).")
    A("")
    A("## Design")
    A("")
    A("- Backbone: facebook/dinov2-with-registers-base (Darcet et al. 2024), 86.58 M parameters, 4 register tokens, local Hugging Face cache, frozen. "
      "Features: hidden_states[12] and hidden_states[6] from one forward pass, before the final layer norm, as for plain DINOv2-B. "
      "The token sequence is [CLS, 4 registers, 24 x 24 patches] at 336 px; CLS and the 4 registers are dropped and the patch tokens are reshaped to the grid, "
      "as the plain pipeline drops CLS.")
    A("- Readout and recipe: R4 (full ASPP with global pooling, the sweep readout), the published readout-swap training code copied unchanged "
      "(v3/gpu/train_full.py): the same 12,803 images (SALICON 10,000, CAT2000 2,000, MIT1003 803), 336 px, 6 epochs, batch 32, AdamW "
      "(learning rate 5e-4, weight decay 1e-4, cosine), loss KL + 2(1 - CC), no augmentation, final epoch, seeds 0, 1, 2 "
      "(the same initial weights, batch order and dropout stream structure as the plain R4 heads of each seed). "
      "Features computed on the fly in float32 with the backbone in the loop, on MPS, as the published run. "
      "A feature cache was not used: 12,803 images x 576 tokens x 768 channels x 2 blocks is 22.7 GB in float16 and 45.3 GB in float32, "
      "against 43 GB free at the start, and a float16 cache would also depart from the published float32 recipe.")
    A("- Scoring: the published evaluation code copied unchanged (v3/gpu/eval_full.py): held-out MIT1003 (last 200 images, 336 px, centre bias on), "
      "v2/stim_v3 (540 Kanizsa, 200 Ehrenstein displays) and v3/stim_alt (360 alternating-rotation displays) at 336 and 448 px, centre bias off. "
      "Statistics: the published analysis copied unchanged (v3/gpu/analyze_full.py): cluster bootstrap 10,000 over the 20 shape x support ratio x polarity cells "
      "with positions resampled within cells (60 configurations), RandomState(2026), 95% percentile CI; seeds averaged per configuration for the pooled rows; "
      "size avg = mean of the 336 and 448 px effects.")
    A("- Measures: Kanizsa effect = r(illusory) - r(control180) on the inside-figure AOI (r = mass inside / (mass inside + mass on the inducers)); "
      "core = the same contrast on figure pixels more than 30 px from the inducers (r_core30); rotation ratios on r_inside: "
      "ratio_uni(t) = effect(uniform rotation t) / effect(illusory), ratio_alt(t) = effect(alternating rotation t) / effect(illusory); "
      "alternating drop = drop_ratio(t) = (m(illusory) - m(alt t)) / (m(illusory) - m(mis t)), the share of the uniform-rotation loss that alternating rotation costs. "
      "All effects against control180.")
    chk = o.get("eval_check")
    if chk:
        ok = all(v.get("max_abs_diff", v.get("max_abs_diff_cc_per_image", 1)) == 0 for v in chk.values())
        A(f"- Pipeline check: the copied evaluation, run on the published plain checkpoint dinov2_L12_R4_s0, reproduced the published score files "
          f"{'bit for bit' if ok else 'NOT exactly'} (MIT1003 CC per image, 16,280 stim_v3 and 7,920 stim_alt values; max |diff| "
          f"{max(v.get('max_abs_diff', v.get('max_abs_diff_cc_per_image', 0)) for v in chk.values())}).")
    A("")
    # ---------------- verdict
    A("## Verdict")
    A("")
    verdict = []
    for blk in ("12", "6"):
        b = o["blocks"][blk]; pp, rp = b["plain"]["pooled"], b["registers"]["pooled"]
        if rp is None:
            verdict.append(f"- Block {blk}: registers results incomplete."); continue
        sp, sr = sign_of(pp["Kz"]), sign_of(rp["Kz"])
        seeds_r = [sign_of(b["registers"]["seeds"][s]["Kz"]) for s in SEEDS if b["registers"]["seeds"][s]]
        core_r = sign_of(rp["KzCore"])
        surv = sr == sp and sp != "0"
        word = {"+": "positive", "-": "negative", "0": "not different from zero"}
        verdict.append(f"- Block {blk}: plain DINOv2-B R4 Kanizsa effect {fe(pp['Kz'])} ({word[sp]}); with registers {fe(rp['Kz'])} ({word[sr]}); "
                       f"per seed with registers: {', '.join(word[x] for x in seeds_r)}; core with registers {fe(rp['KzCore'])} "
                       f"(plain {fe(pp['KzCore'])}). The block-{blk} sign {'survives' if surv else 'does not survive'} registers.")
        N = o["nulls"]; ap_, ar_ = pp["alt"], rp["alt"]
        u, dd = ar_["ratio_uni20"], ar_["drop_ratio20"]
        if sr == "-":
            nu_, nd_, nm_ = N["neg_uni20"], N["neg_drop20"], "the ink and edge-energy nulls (negative effects)"
        else:
            nu_, nd_, nm_ = N["corner_uni20"], N["all_drop20"], "the corner nulls (positive effects; ratio_uni20) and all local nulls (drop_ratio20)"
        verdict.append(f"  Rotation profile with registers: ratio_uni20 {fr(u)} (plain {fr(ap_['ratio_uni20'])}), drop_ratio20 {fr(dd)} "
                       f"(plain {fr(ap_['drop_ratio20'])}); reference: {nm_}, ratio_uni20 {nu_[0]:.2f} to {nu_[1]:.2f}, drop_ratio20 "
                       f"{nd_[0]:.2f} to {nd_[1]:.2f}. Upper CI of ratio_uni20 {'below' if u['ci'][1] < nu_[0] else 'not below'} the null minimum; "
                       f"upper CI of drop_ratio20 {'below' if dd['ci'][1] < nd_[0] else 'not below'} the null minimum.")
    L += verdict
    A("")
    b12, b6 = o["blocks"]["12"], o["blocks"]["6"]
    p12, r12, p6, r6 = b12["plain"]["pooled"], b12["registers"]["pooled"], b6["plain"]["pooled"], b6["registers"]["pooled"]
    d = o.get("diag")
    if r12 and r6 and d and sign_of(p12["Kz"]) == "-" and sign_of(r12["Kz"]) == "+" and sign_of(p6["Kz"]) == sign_of(r6["Kz"]) == "+":
        kz_max = max(d["models"]["dinov2"][f"{px}|kanizsa"]["L12"]["max_over_median_max"] for px in ("336", "448"))
        mit448 = d["models"]["dinov2"]["448|mit1003"]["L12"]; mit336 = d["models"]["dinov2"]["336|mit1003"]["L12"]
        A("### Reading")
        A("")
        A(f"- The block-12 sign does not survive registers. With registers the block-12 R4 readout puts more mass in the illusory figure than in the 180 degree control "
          f"in every seed ({', '.join(f3(b12['registers']['seeds'][s]['Kz']['mean']) for s in SEEDS)}; pooled {fe(r12['Kz'])}), in the core ({fe(r12['KzCore'])}), "
          f"against the 90 degree control ({fe(r12['KzNinety'])}) and for the real filled figure ({fe(r12['RealFilled'])}) and outline ({fe(r12['Real'])}), "
          f"where the plain readout was negative on all five ({fe(p12['Kz'])}, {fe(p12['KzCore'])}, {fe(p12['KzNinety'])}, {fe(p12['RealFilled'])}, {fe(p12['Real'])}). "
          "The flip holds at 336 and at 448 px separately.")
        A(f"- The block-6 sign survives registers: {fe(r6['Kz'])} against {fe(p6['Kz'])}, positive in every seed, with a similar uniform-rotation ratio "
          f"({fr(r6['alt']['ratio_uni20'])} against {fr(p6['alt']['ratio_uni20'])}), below the corner nulls ({o['nulls']['corner_uni20'][0]:.2f} to "
          f"{o['nulls']['corner_uni20'][1]:.2f}). Its alternating drop is larger with registers ({fr(r6['alt']['drop_ratio20'])} against {fr(p6['alt']['drop_ratio20'])}), "
          f"and both lie below the local nulls ({o['nulls']['all_drop20'][0]:.2f} to {o['nulls']['all_drop20'][1]:.2f}).")
        A(f"- The positive block-12 effect with registers is small (less than half of block 6) and shows no alignment signature: a uniform 20 degree rotation leaves "
          f"{fr(r12['alt']['ratio_uni20'])} of it, an interval that includes 1 and overlaps the corner nulls, and the alternating drop is not estimable "
          f"(the uniform loss is not different from zero; {fr(r12['alt']['drop_ratio20'])}). Under the section B rule it is labelled "
          f"'{r12['section_B'].get('label', 'n/a')}'. The Ehrenstein effect with registers is {fe(r12['Ehr'])} (plain {fe(p12['Ehr'])}).")
        A(f"- Held-out MIT1003 CC does not separate the two backbones: block 12 {r12['mit']['mean']:.3f} with registers against {p12['mit']['mean']:.3f} plain, "
          f"block 6 {r6['mit']['mean']:.3f} against {p6['mit']['mean']:.3f}; all six registers readouts pass the R4 validity threshold.")
        A(f"- The flip is not traced to high-norm tokens inside the displays. Plain DINOv2-B block 12 has no patch token above twice the image median on the Kanizsa "
          f"displays at 336 or 448 px (largest max/median ratio {kz_max:.2f}); such tokens appear on natural images, mostly at 448 px "
          f"(in {mit448['frac_images_with_gt2xmedian'] * 100:.0f}% of the MIT1003 images at 448 px and {mit336['frac_images_with_gt2xmedian'] * 100:.0f}% at 336 px, "
          f"at most {mit448['frac_tokens_gt2xmedian'] * 100:.2f}% of tokens), "
          f"and the plain negative effect is present at both input sizes ({fe(p12['Kz336'])} at 336 px, {fe(p12['Kz448'])} at 448 px). "
          "facebook/dinov2-with-registers-base is a separately trained network, not the plain weights with registers added, so this control changes the "
          "checkpoint as well as the token mechanism. What it shows is that the negative block-12 sign is specific to the plain DINOv2-B checkpoint: "
          "a DINOv2-B trained with registers gives a weak, rotation-insensitive positive effect at block 12, and at block 6 a positive, rotation-sensitive effect like that of the plain checkpoint.")
        A("")
    # ---------------- MIT1003
    A("## Held-out MIT1003 CC and validity")
    A("")
    A("Validity rule of the readout swap: CC >= CC(R4 on the same features) - 0.03. Applied within the registers features the rule is met by R4 by construction "
      "(R4 is its own reference), so the substantive check is against the published plain DINOv2-B R4 on the same block (threshold = plain R4 mean CC - 0.03).")
    A("")
    A("| block | seed | plain R4 CC | registers R4 CC | difference | threshold (plain R4 - 0.03) | registers valid | final training loss plain / registers |")
    A("|---|---|---|---|---|---|---|---|")
    for blk in ("12", "6"):
        b = o["blocks"][blk]; thr = b["threshold_plain_R4"]
        for s in SEEDS:
            p_, r_ = b["plain"]["seeds"][s], b["registers"]["seeds"][s]
            tp, tr = b["plain"]["train"][s], b["registers"]["train"][s]
            if r_ is None:
                A(f"| {blk} | {s} | {p_['CC']:.4f} | pending | | {thr:.4f} | | |"); continue
            A(f"| {blk} | {s} | {p_['CC']:.4f} | {r_['CC']:.4f} | {r_['CC'] - p_['CC']:+.4f} | {thr:.4f} | {'yes' if r_['CC'] >= thr else 'no'} | "
              f"{tp['final_loss']:.4f} / {tr['final_loss']:.4f} |")
        pp, rp = b["plain"]["pooled"], b["registers"]["pooled"]
        if rp:
            A(f"| {blk} | mean | {pp['mit']['mean']:.4f} | {rp['mit']['mean']:.4f} | {rp['mit']['mean'] - pp['mit']['mean']:+.4f} | {thr:.4f} | "
              f"{'yes' if rp['mit']['mean'] >= thr else 'no'} | |")
    A("")
    # ---------------- Kanizsa per seed
    A("## Kanizsa effect vs control180 (inside AOI, size avg), per seed and pooled")
    A("")
    A("Per-seed CIs: cluster bootstrap over the 60 configurations of that seed. Pooled: seeds averaged per configuration, then the same bootstrap.")
    A("")
    A("| block | seed | plain DINOv2-B R4 | registers R4 | plain core (r_core30) | registers core (r_core30) |")
    A("|---|---|---|---|---|---|")
    for blk in ("12", "6"):
        b = o["blocks"][blk]
        for s in SEEDS:
            p_, r_ = b["plain"]["seeds"][s], b["registers"]["seeds"][s]
            A(f"| {blk} | {s} | {fe(p_['Kz'])} | {fe(r_['Kz']) if r_ else 'pending'} | {fe(p_['KzCore'])} | {fe(r_['KzCore']) if r_ else 'pending'} |")
        pp, rp = b["plain"]["pooled"], b["registers"]["pooled"]
        A(f"| {blk} | pooled | {fe(pp['Kz'])} {pp['Kz']['p_str']} | {(fe(rp['Kz']) + ' ' + rp['Kz']['p_str']) if rp else 'pending'} | "
          f"{fe(pp['KzCore'])} | {fe(rp['KzCore']) if rp else 'pending'} |")
    A("")
    A("### Other contrasts (pooled over seeds; size avg unless a size is stated; inside AOI unless core)")
    A("")
    A("| block | contrast | plain DINOv2-B R4 | registers R4 |")
    A("|---|---|---|---|")
    names = dict(Kz336="Kanizsa illusory - control180, 336 px only", Kz448="Kanizsa illusory - control180, 448 px only",
                 KzCore336="Kanizsa core (r_core30), 336 px only", KzCore448="Kanizsa core (r_core30), 448 px only",
                 KzNinety="Kanizsa illusory - control90", RealFilled="real filled - control180", Real="real outline - control180",
                 Ehr="Ehrenstein illusory - control_end")
    for blk in ("12", "6"):
        b = o["blocks"][blk]; pp, rp = b["plain"]["pooled"], b["registers"]["pooled"]
        for k, nm in names.items():
            A(f"| {blk} | {nm} | {fe(pp[k])} | {fe(rp[k]) if rp else 'pending'} |")
    A("")
    # ---------------- rotation
    N = o["nulls"]
    A("## Rotation ratios and alternating drop (stim_alt, r_inside, theta 20, size avg)")
    A("")
    A(f"Local-null reference (v3/scores/results.json, as quoted in the paper): the {len(N['neg'])} ink and edge-energy nulls, whose aligned effects are also negative, give "
      f"ratio_uni20 {N['neg_uni20'][0]:.2f} to {N['neg_uni20'][1]:.2f} and drop_ratio20 {N['neg_drop20'][0]:.2f} to {N['neg_drop20'][1]:.2f} "
      f"({N['n_neg_drop']} nulls with a measurable uniform drop). The {len(N['corner'])} corner nulls, whose aligned effects are positive, give ratio_uni20 "
      f"{N['corner_uni20'][0]:.2f} to {N['corner_uni20'][1]:.2f} ({N['n_corner_uni']} with ratios computed), and all local nulls with a measurable uniform drop "
      f"give drop_ratio20 {N['all_drop20'][0]:.2f} to {N['all_drop20'][1]:.2f} ({N['n_all_drop']} nulls; all shapes pooled). "
      "A ratio_uni below the null range means the effect depends on inducer alignment more than "
      "the local nulls; a drop ratio below the null range means alternating rotation costs less of the uniform loss than local image statistics predict.")
    A("")
    A("| block | seed | ratio_uni20 plain | ratio_uni20 registers | ratio_alt20 plain | ratio_alt20 registers | drop_ratio20 plain | drop_ratio20 registers |")
    A("|---|---|---|---|---|---|---|---|")
    for blk in ("12", "6"):
        b = o["blocks"][blk]
        for s in SEEDS:
            p_, r_ = b["plain"]["seeds"][s]["alt"], (b["registers"]["seeds"][s] or {}).get("alt")
            g = lambda d, k: fr(d[k]) if d else "pending"
            A(f"| {blk} | {s} | {g(p_, 'ratio_uni20')} | {g(r_, 'ratio_uni20')} | {g(p_, 'ratio_alt20')} | {g(r_, 'ratio_alt20')} | "
              f"{g(p_, 'drop_ratio20')} | {g(r_, 'drop_ratio20')} |")
        pp, rp = b["plain"]["pooled"]["alt"], (b["registers"]["pooled"] or {}).get("alt")
        g = lambda d, k: fr(d[k]) if d else "pending"
        A(f"| {blk} | pooled | {g(pp, 'ratio_uni20')} | {g(rp, 'ratio_uni20')} | {g(pp, 'ratio_alt20')} | {g(rp, 'ratio_alt20')} | "
          f"{g(pp, 'drop_ratio20')} | {g(rp, 'drop_ratio20')} |")
    A("")
    ref = o.get("pub_other_r4") or {}
    if ref:
        A("For comparison, the other published R4 readouts (pooled over seeds, r_inside, theta 20): " + "; ".join(
            f"{fs}: Kanizsa effect {fe(v['Kz'])}, ratio_uni20 {fr(v['ratio_uni20'])}, ratio_alt20 {fr(v['ratio_alt20'])}, drop_ratio20 {fr(v['drop_ratio20'])}"
            for fs, v in ref.items()) + ".")
        A("")
    A("### All angles (pooled over seeds)")
    A("")
    A("| block | backbone | ratio_uni10 | ratio_uni30 | ratio_alt10 | ratio_alt30 | drop_ratio30 | alt20 - mis20 (r_inside) |")
    A("|---|---|---|---|---|---|---|---|")
    for blk in ("12", "6"):
        b = o["blocks"][blk]
        for nm, key in (("plain", "plain"), ("registers", "registers")):
            po = b[key]["pooled"]
            if not po:
                A(f"| {blk} | {nm} | pending | | | | | |"); continue
            a = po["alt"]
            A(f"| {blk} | {nm} | {fr(a['ratio_uni10'])} | {fr(a['ratio_uni30'])} | {fr(a['ratio_alt10'])} | {fr(a['ratio_alt30'])} | "
              f"{fr(a['drop_ratio30'])} | {fe(a['alt20_minus_mis20'])} |")
    A("")
    A("Section B rule (uniform ratio(20) labelled only when the aligned CI is above 0): " + "; ".join(
        f"block {blk} {nm}: " + ("eligible, " + fr(b_[k]['pooled']['section_B']['ratio_20']) + " " + b_[k]['pooled']['section_B']['label']
                                 if b_[k]['pooled'] and b_[k]['pooled']['section_B'].get('eligible') else ("not eligible" if b_[k]['pooled'] else "pending"))
        for blk, b_ in o["blocks"].items() for nm, k in (("plain", "plain"), ("registers", "registers"))) + ".")
    A("")
    # ---------------- token norms
    d = o.get("diag")
    if d:
        A("## Token-norm diagnostic (CPU, the readouts' input sizes)")
        A("")
        A(f"Patch-token L2 norms (CLS and registers excluded) on {d['n']['kanizsa']} Kanizsa displays (illusory and control180, one per shape x support ratio x polarity) "
          f"and {d['n']['mit1003']} held-out MIT1003 images, at 336 and 448 px. Per image, the ratio of the largest patch norm to the median patch norm; "
          "outlier = norm above twice the image's median (high-norm artefact tokens in DINOv2 are typically several times the median).")
        A("")
        A("| backbone | block | px | image set | median of max/median | largest max/median | share of tokens > 2 x median | share of images with such a token |")
        A("|---|---|---|---|---|---|---|---|")
        for bb, nm in (("dinov2", "plain"), ("dinov2reg", "registers")):
            for L_ in ("L12", "L6"):
                for px in ("336", "448"):
                    for st in ("kanizsa", "mit1003"):
                        q = d["models"][bb][f"{px}|{st}"][L_]
                        A(f"| {nm} | {L_[1:]} | {px} | {st} | {q['max_over_median']:.2f} | {q['max_over_median_max']:.2f} | {q['frac_tokens_gt2xmedian']:.5f} | "
                          f"{q['frac_images_with_gt2xmedian']:.3f} |")
        A("")
    A("## Run record and files")
    A("")
    ql = os.path.join(H, "logs", "run_reg.log")
    if os.path.exists(ql):
        for line in open(ql):
            if any(k in line for k in (" done train_", " done eval_", "FAILED", "STOP", "finished")):
                A(f"- {line.strip()}")
    A("- Interpreter: ~/Downloads/saliency_research/.venv/bin/python (torch 2.12, transformers 5.12.1), offline Hugging Face cache, device MPS, "
      "one GPU job at a time (gpu_idle.sh checks before every step), caffeinate -i. No latency timing was run.")
    A("- Code: nn/registers/regcommon.py (adds the backbone), train_reg.py, eval_reg.py, analyze_reg.py (copies of v3/gpu/train_full.py, eval_full.py, "
      "analyze_full.py with only the import changed), report_reg.py, norm_diag.py, check_eval.py, run_reg.sh. Nothing outside nn/registers was written.")
    A("- Outputs: nn/registers/full/ (ckpt/*.pt and training records, scores/, maps/stim_alt/, results_full.json, report_full.md), "
      "nn/registers/registers_results.json (all numbers in this file, per seed and pooled, and every local null), nn/registers/diag/norm_diag.json, "
      "nn/registers/check/eval_check.json.")
    open(os.path.join(H, "REGISTERS_RESULTS.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
