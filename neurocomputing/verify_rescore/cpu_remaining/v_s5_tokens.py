"""
v_s5_tokens.py: independent recomputation of S5 (A3 token analysis of the published ViTs, rule Section 6 item 8 (i),
(ii), (iii), (v); results/explore/A3_tokens_published.json). Written 19 November 2026 for the verification of the
remaining CPU steps. Imports no project module except analyze_re for require_public(), which runs first. numpy and
scipy.ndimage only.
  - (iii) per configuration, from the saved token components (v2/encoding/tokens/<net>_s<S>/<config>.npz) and the AOI
    files: E_set = sum over the token set of <d_test, d_rc> / sum of ||d_rc||^2 for the interior set ("same" across all
    images and footprint inside the inside AOI) and the near-exterior set ("same", footprint free of figure pixels,
    footprint distance to the inducer discs no larger than the largest such distance among the configuration's interior
    tokens); token footprints [floor(r P H/S), ceil((r+1) P H/S)) x [floor(c P W/S), ceil((c+1) P W/S)) on the
    1024 x 768 display (definitions read from v2/encoding/common_enc.py and encoding_nulls.py, re-implemented here);
    sizes averaged (nanmean); excess = E_int - E_ext_near for the illusory test; WCU and WCR p (12 clusters); retention
    under mis20 = mean excess(mis20) / mean excess(illusory) (point value).
  - (ii) the published near-exterior bounds (int and ext_near, v2 two-stage percentile, RandomState(2026), 10,000)
    recomputed from the same per-configuration values and compared with v2/verify/encoding_nulls.json.
  - (i) verdicts re-derived from the published 4-decimal bounds in v2/encoding/results.json (interior lower bound above
    the background-null and control90-null upper bounds); (ii) verdicts from encoding_nulls.json; rule 7.6 statements;
    sweep agreement counts with saliency E intervals recomputed here (WCU and percentile).
Output: neuro/verify_rescore/cpu_remaining/v_s5_tokens.json
"""
import os, sys, math, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcommon as V                                                       # noqa: E402

N_HASHED = V.guard()                                                      # require_public() first
import json                                                               # noqa: E402
import numpy as np                                                        # noqa: E402
from scipy.ndimage import distance_transform_edt                          # noqa: E402

ROOT, RE = V.ROOT, V.RE
OUT = os.path.join(V.HERE, "v_s5_tokens.json")
TOK = os.path.join(ROOT, "v2", "encoding", "tokens")
STIMIMG = os.path.join(ROOT, "v2", "stim_v3", "img")
H, W = 768, 1024
NL = {"dinov2": 13, "clip": 25, "siglip": 25}
KEYS, GRID = V.design()
ITEMS = {it["config"]: it for it in V.jl(os.path.join(ROOT, "v2", "stim_v3", "items.json"))
         if it["illusion"] == "kanizsa" and it["cond"] == "illusory"}
_geo = {}


def geo(cfg, G, P, S):
    key = (cfg, G, P, S)
    if key not in _geo:
        a = np.load(os.path.join(STIMIMG, ITEMS[cfg]["aoi"]))
        ins, ind = a["inside"], a["ind"]
        fig = ins | ind
        dI = distance_transform_edt(~ind)
        sy, sx = H / S, W / S
        out = np.zeros(G * G, bool); mind = np.full(G * G, np.inf)
        for rr in range(G):
            y0 = int(math.floor(max(0.0, rr * P * sy))); y1 = int(math.ceil(min(H, (rr + 1) * P * sy)))
            for cc in range(G):
                x0 = int(math.floor(max(0.0, cc * P * sx))); x1 = int(math.ceil(min(W, (cc + 1) * P * sx)))
                k = rr * G + cc
                if y1 <= y0 or x1 <= x0:
                    continue
                out[k] = not fig[y0:y1, x0:x1].any(); mind[k] = dI[y0:y1, x0:x1].min()
        _geo[key] = (out, mind)
    return _geo[key]


def per_config(net, S):
    """{set: {test: array [K, P, n_hs]}} of E for the interior and near-exterior sets."""
    nL = NL[net]
    E = {s: {t: np.full((len(GRID), len(GRID[0]), nL), np.nan) for t in ("illusory", "mis20")} for s in ("int", "ext")}
    for i, row in enumerate(GRID):
        for j, cfg in enumerate(row):
            z = np.load(os.path.join(TOK, f"{net}_s{S}", cfg + ".npz"))
            G, P, S_ = [int(v) for v in z["grid"]]
            same = z["same"].all(0)
            out, mind = geo(cfg, G, P, S_)
            intm = same & z["inside_full"]
            dmax = mind[intm].max() if intm.any() else -1
            m = {"int": intm, "ext": same & out & (mind <= dmax)}
            tests = list(z["tests"]); dot = z["dot"].astype(np.float64); nrc = z["nrc"].astype(np.float64)
            for s in m:
                if not m[s].any():
                    continue
                R = nrc[:, m[s]].sum(-1)
                for t in ("illusory", "mis20"):
                    D = dot[:, tests.index(t)][:, m[s]].sum(-1)
                    ok = R > 0
                    E[s][t][i, j, ok] = D[ok] / R[ok]
    return E


def side(ci):
    return ci[0] > 0


def main():
    t0 = time.time()
    PUB = V.jl(os.path.join(RE, "results", "explore", "A3_tokens_published.json"))["result"]["networks"]
    NUL = V.jl(os.path.join(ROOT, "v2", "verify", "encoding_nulls.json"))
    RES = V.jl(os.path.join(ROOT, "v2", "encoding", "results.json"))["networks"]
    res = dict(n_hashed_verified=N_HASHED, iii={}, ii_bounds={}, verdicts={})
    for net in ("dinov2", "clip", "siglip"):
        Es = [per_config(net, S) for S in (336, 448)]
        with np.errstate(all="ignore"):
            avg = {s: {t: np.nanmean(np.stack([Es[0][s][t], Es[1][s][t]]), 0) for t in ("illusory", "mis20")}
                   for s in ("int", "ext")}
        d_iii, d_ii, rows = 0.0, 0.0, {}
        n_nan = 0
        for L in range(1, NL[net]):
            ex = avg["int"]["illusory"][:, :, L] - avg["ext"]["illusory"][:, :, L]
            exm = avg["int"]["mis20"][:, :, L] - avg["ext"]["mis20"][:, :, L]
            if np.isnan(ex).any():
                n_nan += 1
                continue
            w = V.wild(ex)
            ret = float(exm.mean() / ex.mean())
            st = PUB[net]["iii_paired_excess"][str(L)]
            d = max(abs(w["mean"] - st["excess"]["mean"]), abs(w["ci"][0] - st["excess"]["ci"][0]),
                    abs(w["ci"][1] - st["excess"]["ci"][1]), abs(w["p"] - st["excess"]["p"]))
            sr = st["retention_mis20"]
            if sr.get("value") is not None:
                d = max(d, abs(ret - sr["value"]))
            d_iii = max(d_iii, d)
            rows[L] = dict(mean=w["mean"], ci=w["ci"], p=w["p"], above0=bool(w["ci"][0] > 0), retention=ret)
            # (ii) bounds: int and ext_near, percentile (published Boot), against encoding_nulls.json
            for s, key in (("int", "int"), ("ext", "ext_near")):
                v = avg[s]["illusory"][:, :, L]
                pc = V.pct_ci(v)
                ref = NUL[net]["layers"][str(L)][key]
                d_ii = max(d_ii, abs(float(v.mean()) - ref["mean"]), abs(pc[0] - ref["ci"][0]), abs(pc[1] - ref["ci"][1]))
        res["iii"][net] = dict(max_abs_diff_vs_stored=d_iii, blocks_with_nan=n_nan,
                               blocks_excess_above0=[L for L, x in rows.items() if x["above0"]],
                               sample={L: rows[L] for L in rows if (net, L) in (("clip", 13), ("clip", 24), ("dinov2", 6),
                                                                              ("siglip", 6), ("siglip", 24))})
        res["ii_bounds"][net] = dict(max_abs_diff_vs_encoding_nulls=d_ii)
        # verdicts
        pre = []
        for L in range(1, NL[net]):
            b = RES[net]["kanizsa"]["avg"][L]
            assert b["layer"] == L
            fs = b["interior"]["ci"][0] > b["bg_null"]["ci"][1] and b["interior"]["ci"][0] > b["c90_null"]["ci"][1]
            assert fs == bool(b["figure_specific"])
            if fs:
                pre.append(L)
        near = [L for L in range(1, NL[net]) if NUL[net]["layers"][str(L)]["int"]["ci"][0] >
                NUL[net]["layers"][str(L)]["ext_near"]["ci"][1]]
        res["verdicts"][net] = dict(prespecified=pre, near_exterior=near, both=sorted(set(pre) & set(near)))

    # saliency E of the sweep and the swap R4 cells (WCU and percentile), agreement counts
    def sweep(bb, L):
        out = {}
        for z in ("336", "448"):
            for sub in (("v3", "scores", "raw_v3maps"), ("nn", "results", "raw_v3maps")):
                p = os.path.join(ROOT, *sub, f"{bb}_L{L}_s{z}.json")
                if os.path.exists(p):
                    for n, v in V.jl(p).items():
                        out.setdefault(n, []).append(v["r_inside"])
                    break
        return {n: float(np.mean(v)) for n, v in out.items()}

    def headE(fs):
        acc = {}
        for s in (0, 1, 2):
            d = V.jl(os.path.join(ROOT, "v3", "gpu", "full", "scores", f"stim_v3_{fs}_R4_s{s}.json"))
            for z in ("336", "448"):
                for n, v in d[z]["r_inside"].items():
                    acc.setdefault(n, []).append(v)
        return {n: float(np.mean(v)) for n, v in acc.items()}

    rows = {}
    for bb, Ls in (("dinov2", (2, 4, 6, 8, 10, 12)), ("clip", (4, 8, 12, 16, 20, 24)), ("siglip", (4, 8, 12, 16, 20, 24))):
        for L in Ls:
            sc = sweep(bb, L)
            e = V.per_config(GRID, lambda c: sc[c + "_illusory"] - sc[c + "_control180"])
            rows[f"{bb}_L{L}"] = (bb, L, V.wild(e)["ci"], V.pct_ci(e))
    agree = {}
    for nul in ("prespecified", "near_exterior"):
        for iv, k in (("wcu", 2), ("pct", 3)):
            ok = [u for u, x in rows.items() if (x[1] in res["verdicts"][x[0]][nul]) == side(x[k])]
            agree[f"{nul}_{iv}"] = dict(n_agree=len(ok), disagree=sorted(set(rows) - set(ok)))
    swap = {}
    for fs, (net, L) in (("dinov2_L12", ("dinov2", 12)), ("dinov2_L6", ("dinov2", 6)), ("siglip_L24", ("siglip", 24))):
        sc = headE(fs)
        e = V.per_config(GRID, lambda c: sc[c + "_illusory"] - sc[c + "_control180"])
        w = V.wild(e)
        swap[fs] = dict(E=w["mean"], ci=w["ci"], token_prespecified=L in res["verdicts"][net]["prespecified"],
                        agrees_prespecified=(L in res["verdicts"][net]["prespecified"]) == side(w["ci"]))
    res["sweep_agreement"] = agree
    res["swap_rows"] = swap
    v = res["verdicts"]
    res["rule_7_6"] = dict(
        clip_prespecified_7_to_24=v["clip"]["prespecified"] == list(range(7, 25)),
        clip_near_exterior_none=v["clip"]["near_exterior"] == [],
        read_blocks_passing_both={net: sorted(set(v[net]["both"]) & set(Ls)) for net, Ls in
                                  (("dinov2", (2, 4, 6, 8, 10, 12)), ("clip", (4, 8, 12, 16, 20, 24)),
                                   ("siglip", (4, 8, 12, 16, 20, 24)))},
        dinov2_block7_both=7 in v["dinov2"]["both"])
    res["seconds"] = time.time() - t0
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1)[:8000])


if __name__ == "__main__":
    main()
