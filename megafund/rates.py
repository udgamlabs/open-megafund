"""Cancer drug success rates rebuilt from the public dataset udgamlabs/open-cancer-drug-paths, and the megafund table.

    python -m megafund rates                 # the four rates, with drug-level bootstrap intervals
    python -m megafund table [--n 10000]     # funds A and B on the 2012, 2019 and open rates

The dataset is downloaded once from Hugging Face into ./data/ (about 5 MB; CC BY 4.0). Each path in it already carries
the outcome of every transition (success, failure or censored); a rate is successes / (successes + failures).
Intervals resample drugs, not paths, so one drug tested in many cancers is not counted as independent evidence.
"""
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np

from .params import load
from .sim import NAMES, Fund

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
REPO = "https://huggingface.co/datasets/udgamlabs/open-cancer-drug-paths/resolve/main/"
FILES = {"paths": "data/paths.parquet", "scores": "tables/path_check_scores.json"}
TRANSITIONS = {"p12": ("outcome_p1_to_p2", "phase 1 to 2"), "p23": ("outcome_p2_to_p3", "phase 2 to 3"),
               "p3a": ("outcome_p3_to_approval", "phase 3 to approval"),
               "p1a": ("outcome_p1_to_approval", "phase 1 to approval")}
# Wong, Siah and Lo (2019), Biostatistics 20(2):273-286, Table 2, oncology, path-by-path (corrected version).
WONG2019 = {"p12": 0.576, "p23": 0.327, "p3a": 0.355, "p1a": 0.034}
RHO_OPEN = 0.26  # one-factor correlation from drug mechanism groups: tables/correlation.csv in the dataset


def fetch(name):
    """Download one dataset file into ./data/ (once) and return its path."""
    DATA.mkdir(exist_ok=True)
    out = DATA / Path(FILES[name]).name
    if not out.exists():
        with urllib.request.urlopen(REPO + FILES[name]) as r:
            out.write_bytes(r.read())
    return out


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_paths():
    import pyarrow.parquet as pq
    cols = ["drug_key"] + [c for c, _ in TRANSITIONS.values()]
    return pq.read_table(fetch("paths"), columns=cols).to_pylist()


def rates(paths, n_boot=1000, seed=0):
    drugs = sorted({p["drug_key"] for p in paths})
    idx = {d: i for i, d in enumerate(drugs)}
    out = {}
    for t, (col, label) in TRANSITIONS.items():
        s, f, c = np.zeros(len(drugs)), np.zeros(len(drugs)), 0
        for p in paths:
            o = p[col]
            if o == "success":
                s[idx[p["drug_key"]]] += 1
            elif o == "failure":
                f[idx[p["drug_key"]]] += 1
            elif o == "censored":
                c += 1
        S, F = s.sum(), f.sum()
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            b = rng.integers(0, len(drugs), len(drugs))
            den = s[b].sum() + f[b].sum()
            if den:
                boots.append(s[b].sum() / den)
        out[t] = {"transition": label, "rate": round(float(S / (S + F)), 4), "success": int(S), "failure": int(F),
                  "censored": c, "ci95": [round(float(np.percentile(boots, 2.5)), 4),
                                          round(float(np.percentile(boots, 97.5)), 4)],
                  "wong2019": WONG2019[t]}
    return out


def approval_precision():
    """Share of credited approvals the fresh label check found right (post-hoc sensitivity, not a validated rate)."""
    s = json.loads(fetch("scores").read_text())["fresh_check_t3b"]["approved"]["by_stratum"]["approved"]
    return s["correct"] / s["n"], s


# --- the 2012 funds on other rates -------------------------------------------------------------------------------------
def rescale(T, target):
    """Rows P1, P2, P3 rescaled so that the eventual chance of moving up equals the target.

    Each row's chance of staying put is kept, so durations are unchanged; the 2012 NDA approval step is kept, and the
    phase 3 row is set so that phase 3 -> NDA -> approval equals the target phase 3 to approval rate.
    """
    T = np.array(T, dtype=float).copy()
    nda_s = T[NAMES["NDA"] - 1, NAMES["APP"] - 1] / (1 - T[NAMES["NDA"] - 1, NAMES["NDA"] - 1])
    want = {"P1": target["p12"], "P2": target["p23"], "P3": min(1.0, target["p3a"] / nda_s)}
    for ph, s in want.items():
        i = NAMES[ph] - 1
        stay = T[i, i]
        up = T[i].copy()
        up[: i + 1] = 0
        move = 1 - stay
        if up.sum() > 0:
            T[i] = 0
            T[i, i] = stay
            T[i, 0] = move * (1 - s)
            T[i] += up / up.sum() * move * s
    return T


def table(nsims=10000, workers=4, rho=RHO_OPEN):
    r = rates(load_paths())
    open_rates = {t: r[t]["rate"] for t in ("p12", "p23", "p3a")}
    q, _ = approval_precision()
    corrected = dict(open_rates, p3a=round(r["p3a"]["success"] * q / (r["p3a"]["success"] + r["p3a"]["failure"]), 4))
    sets = {"2012 inputs": None, "2019 paid data (Wong, Siah, Lo)": WONG2019, "open data": open_rates,
            "open data, approvals corrected (post-hoc)": corrected}
    out = {"rates": r, "rate_sets": sets, "transition_rho": [0.0, rho], "nsims": nsims, "funds": {}}
    for fund in ("A", "B"):
        p = load(HERE / "inputs" / "fsl2012" / f"init_params_fn_NBTx_{fund}.m")
        base = np.array(p["trans_prob"])
        out["funds"][fund] = {}
        for name, target in sets.items():
            pp = dict(p, trans_prob=(base if target is None else rescale(base, target)).tolist())
            for tr in (0.0, rho):
                res = Fund(pp, trans_rho=tr).run(nsims, 0, workers=workers)
                row = {k: res[k] for k in ("roe_ann_mean", "roe_ann_se", "p_default_A1", "p_default_A2")}
                out["funds"][fund][f"{name} | correlation {tr}"] = row
                print(f"fund {fund} | {name:42s} | correlation {tr:<4} | equity return {100 * row['roe_ann_mean']:5.1f}%"
                      f" a year | junior bond default {100 * row['p_default_A2']:.1f}%", flush=True)
    return out
