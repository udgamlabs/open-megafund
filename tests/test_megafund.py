"""Run: python -m pytest -q tests/   (the full T1 check is `python -m megafund t1`, about 10,000 funds each)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from megafund.__main__ import EXPECTED, INPUTS  # noqa: E402
from megafund.params import load  # noqa: E402
from megafund.rates import rescale  # noqa: E402
from megafund.sim import NAMES, Fund  # noqa: E402


def test_inputs_parse():
    for w in ("A", "B"):
        p = load(INPUTS[w])
        T = np.array(p["trans_prob"])
        assert T.shape[0] == T.shape[1] - 1 or T.shape[0] == T.shape[1]
        assert np.allclose(T.sum(axis=1), 1.0, atol=1e-6)


def test_rescale_hits_targets():
    T = np.array(load(INPUTS["B"])["trans_prob"])
    target = {"p12": 0.6, "p23": 0.35, "p3a": 0.4}
    R = rescale(T, target)
    assert np.allclose(R.sum(axis=1), 1.0)
    for ph, t in (("P1", "p12"), ("P2", "p23")):
        i = NAMES[ph] - 1
        up = R[i, i + 1:].sum()
        assert abs(up / (up + R[i, 0]) - target[t]) < 1e-9
        assert R[i, i] == T[i, i]  # durations kept


def test_small_t1_close_to_paper():
    """A quick version of T1: 1,000 funds, a looser 3-standard-error bar."""
    for w in ("A", "B"):
        r = Fund(load(INPUTS[w])).run(1000, 0, workers=1)
        assert abs(r["roe_ann_mean"] - EXPECTED[w]) <= 3 * r["roe_ann_se"]


if __name__ == "__main__":
    test_inputs_parse(); test_rescale_hits_targets(); test_small_t1_close_to_paper(); print("ok")
