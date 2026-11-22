"""Read the RBOMatlab parameter files (inputs/fsl2012/*.m) into a plain dict.

Only the fields the Markov-mode simulator uses are read. The .m files are kept unchanged (see inputs/fsl2012/SOURCE.md);
this parser is ours. It reads `name = value;` lines, where value is a number, a matrix `[...]`, a cell `{'P2'}` or
`ones(7,1).*1.5`, after removing MATLAB comments and line continuations.
"""
import re
from pathlib import Path

STATES = ["DSC", "PRE", "P1", "P2", "P3", "NDA", "APP"]
PRICING_COLS = ["vmu", "vsigma", "vmx", "UpFront", "Milestone", "mu", "sigma", "max", "FutureCostEst"]


def _clean(text):
    out = []
    for line in text.splitlines():
        line = line.split("%", 1)[0]
        out.append(line.replace("...", " "))
    return "\n".join(out)


def _matrix(body):
    rows = [r for r in body.split(";") if r.strip()]
    return [[float(x) for x in re.split(r"[,\s]+", r.strip()) if x] for r in rows]


def _value(raw):
    raw = raw.strip()
    if raw.startswith("["):
        m = _matrix(raw.strip("[]"))
        return m[0] if len(m) == 1 else m
    if raw.startswith("{"):
        return re.findall(r"'([^']*)'", raw)
    m = re.fullmatch(r"ones\((\d+),1\)\.\*([\d.]+)", raw.replace(" ", ""))
    if m:
        return [float(m.group(2))] * int(m.group(1))
    if raw in ("true", "false"):
        return raw == "true"
    try:
        return float(eval(raw, {"__builtins__": {}}, {}))  # plain arithmetic such as 0.05/2
    except Exception:  # noqa: BLE001
        return raw


def _vector(raw):
    """[0.05/2, 0.08/2, 0] -> floats, evaluating each entry."""
    return [float(eval(x, {"__builtins__": {}}, {})) for x in re.split(r",\s*|\s+", raw.strip().strip("[]")) if x]


def load(path):
    text = _clean(Path(path).read_text())
    # statements end with ';' outside brackets
    stmts, depth, cur = [], 0, []
    for ch in text:
        if ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        if ch == ";" and depth == 0:
            stmts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    vals = {}
    for s in stmts:
        if "=" not in s:
            continue
        name, raw = s.split("=", 1)
        name = name.strip().split()[-1] if name.strip() else ""
        if not re.fullmatch(r"[A-Za-z_][\w.]*", name):
            continue
        if name in ("bonds.coupon", "bonds.interest_coverage", "bonds.capital_structure"):
            vals[name] = _vector(raw)
        else:
            vals[name] = _value(raw)
    p = {
        "rho": vals["assets.rho"],
        "trans_prob": vals["assets.trans_prob"],
        "pricing": vals["assets.pricing_params"],
        "sell_in_phase": vals["assets.sell_in_phase"][0],
        "sale_time": vals["assets.sale_time"],
        "ratio_unchanged": vals["assets.ratio_unchanged"],
        "equity_stake": vals["assets.equity_stake"],
        "capital_structure": vals["bonds.capital_structure"],
        "ic_pers": int(vals["bonds.IC_pers"]),
        "interest_coverage": vals["bonds.interest_coverage"],
        "amort_timing": vals["bonds.amort_timing"],
        "coupon": vals["bonds.coupon"],
        "servicing_rate": vals["bonds.servicing_rate"],
        "svc_accrual": vals["bonds.svc_accrual_int_rate"],
        "cash_accrual": vals["bonds.cash_accrual_int_rate"],
        "initial_cash": vals["simu.initial_cash"],
        "initial_compounds": [int(x) for x in vals["simu.initial_compounds"]],
        "mode": vals["assets.mode"].strip("'"),
        "ce_amt": vals.get("ce.ce_amt", 0.0),
    }
    assert p["mode"] == "Markov", p["mode"]
    assert p["ce_amt"] == 0, "guarantees are not ported"
    assert len(p["trans_prob"]) == 7 and all(len(r) == 7 for r in p["trans_prob"])
    assert len(p["pricing"]) == 7 and all(len(r) == 9 for r in p["pricing"])
    for r in p["trans_prob"]:
        assert abs(sum(r) - 1) < 1e-9, r
    return p
