"""Command line.

    python -m megafund t1 [--n 10000]                     reproduce Fernandez, Stein and Lo (2012), funds A and B
    python -m megafund sim A|B [--n N] [--rho R] [--trans-rho R]
    python -m megafund rates                              success rates from the public dataset
    python -m megafund table [--n 10000]                  both funds on the 2012, 2019 and open rates
"""
import argparse
import datetime
import json
from pathlib import Path

from .params import load
from .sim import Fund

HERE = Path(__file__).resolve().parent.parent
INPUTS = {"A": HERE / "inputs/fsl2012/init_params_fn_NBTx_A.m", "B": HERE / "inputs/fsl2012/init_params_fn_NBTx_B.m"}
EXPECTED = {"A": 0.091, "B": 0.106}  # RBOMatlab Testing/run_tests.m
Z = 1.645


def t1(n, seed, workers):
    """Pass if each fund's mean annualised equity return is within 1.645 standard errors of RBOMatlab's value."""
    out = {"date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "runs": {}}
    ok = True
    for w in ("A", "B"):
        r = Fund(load(INPUTS[w])).run(n, seed, workers)
        gap = abs(r["roe_ann_mean"] - EXPECTED[w]) / r["roe_ann_se"]
        r.update(expected=EXPECTED[w], gap_in_se=round(gap, 3), passed=gap <= Z)
        out["runs"][w] = r
        ok &= r["passed"]
        print(f"fund {w}: {100 * r['roe_ann_mean']:.1f}% a year (paper's code: {100 * EXPECTED[w]:.1f}%), "
              f"{gap:.2f} SE apart -> {'pass' if r['passed'] else 'FAIL'}", flush=True)
    out["passed"] = ok
    return out


def main():
    ap = argparse.ArgumentParser(prog="python -m megafund")
    ap.add_argument("cmd", choices=["t1", "sim", "rates", "table"])
    ap.add_argument("which", nargs="?", default="A", choices=["A", "B"])
    ap.add_argument("--n", type=int, default=10000, help="simulated funds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rho", type=float, default=None, help="value correlation (2012 model); default from the inputs")
    ap.add_argument("--trans-rho", type=float, default=0.0, help="correlation of phase transitions (extension)")
    ap.add_argument("--out", type=Path, default=None, help="also write the result as JSON")
    a = ap.parse_args()
    if a.cmd == "sim":
        res = Fund(load(INPUTS[a.which]), trans_rho=a.trans_rho, value_rho=a.rho).run(a.n, a.seed, a.workers)
        print(json.dumps(res, indent=1))
    elif a.cmd == "t1":
        res = t1(a.n, a.seed, a.workers)
        print("T1", "PASS" if res["passed"] else "FAIL")
    else:
        from . import rates as R
        if a.cmd == "rates":
            res = R.rates(R.load_paths())
            for v in res.values():
                print(f"{v['transition']:20s} {100 * v['rate']:5.1f}%  (95% CI {100 * v['ci95'][0]:.1f}-"
                      f"{100 * v['ci95'][1]:.1f}; {v['success']} successes, {v['failure']} failures; "
                      f"2019 paid data {100 * v['wong2019']:.1f}%)")
        else:
            res = R.table(a.n, a.workers)
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
