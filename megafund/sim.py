"""Megafund simulator: a Python port of RBOMatlab's `simulate_CFs_fn` (Markov mode, one compound type, no guarantee).

Licence: GPL-3.0-or-later. This file follows the structure of RBOMatlab (Fagnan, Gromatzky, Fernandez, Stein, Lo;
github.com/dfagnan/RBOMatlab, commit e43bb575), whose files carry GPL-3.0 headers. See inputs/fsl2012/SOURCE.md.

Model, per simulated fund (half-year periods):
- Buy compounds at t=1 while cash allows: price = this phase's trial cost (capped log-normal) + upfront fee, and reserve
  the future-cost estimate. Keep two periods of coupons aside.
- Each period every funded compound moves along a Markov chain of phases (DSC, PRE, P1, P2, P3, NDA, APP).
- A compound is sold when it reaches the sale phase (or approval). The sale price is a capped log-normal draw,
  correlated across compounds through one common factor per fund (rho, on log value), times the fund's equity stake.
  Cash arrives ceil(sale_time) periods later.
- A compound that advanced must be funded again (next trial cost + milestone) before it moves on.
- Each period: default test, then servicing fee, coupons, scheduled amortisation; interest-coverage tests for each
  tranche over the next IC_pers periods; compounds are sold early to cure a shortfall; spare cash funds compounds.
- After a default everything is liquidated and cash goes down the waterfall; the default is permanent.
- At the end the rest is liquidated; equity gets the remaining cash. ROE = cash / equity - 1 (floored at -1).

Extension (not in the 2012 model): `trans_rho` > 0 makes the phase transitions themselves correlated, through a
one-factor Gaussian copula with a fresh common factor each period (Lo and Siah 2021 use a single-factor Gaussian
copula on transitions; their exact timing of the factor is in their online supplement, not read).
"""
import math

import numpy as np
_erf = np.vectorize(math.erf)


def ndtr(x):
    return 0.5 * (1.0 + _erf(np.asarray(x) / math.sqrt(2.0)))

SLD, DSC, PRE, P1, P2, P3, NDA, APP = 0, 1, 2, 3, 4, 5, 6, 7
NAMES = {"DSC": DSC, "PRE": PRE, "P1": P1, "P2": P2, "P3": P3, "NDA": NDA, "APP": APP}
VMU, VSIG, VMX, UPF, MIL, CMU, CSIG, CMX, FCE = range(9)


class Fund:
    def __init__(self, p, trans_rho=0.0, value_rho=None):
        self.p = p
        self.T = np.array(p["trans_prob"])
        self.pp = np.array(p["pricing"])
        self.rho = p["rho"] if value_rho is None else value_rho
        self.trans_rho = trans_rho
        self.sell = NAMES[p["sell_in_phase"]]
        amort = np.array(p["amort_timing"], dtype=int)
        self.npers = int(amort.max()) + 1
        self.sale_lag = math.ceil(max(p["sale_time"]))
        self.horizon = self.npers + self.sale_lag
        self.nominal = p["initial_cash"] * np.array(p["capital_structure"])
        self.coupon = np.array(p["coupon"])
        self.nb = len(self.nominal) - 1
        # amortisation schedule, periods 1..npers (column 0 unused)
        self.amort = np.zeros((self.nb, self.npers + 1))
        left = self.nominal[: self.nb].copy()
        for t in range(1, self.npers + 1):
            for b in range(self.nb):
                a0, a1 = amort[b]
                if a0 <= t <= a1:
                    due = self.nominal[b] / (a1 - a0 + 1)
                    self.amort[b, t] = due if due < left[b] else left[b]
                left[b] -= self.amort[b, t]
        counts = p["initial_compounds"]
        self.start = np.repeat(np.arange(1, 8), counts)
        self.svc = p["servicing_rate"]
        self.eq = p["equity_stake"]

    # --- draws ---------------------------------------------------------------------------------------------------
    def trial_cost(self, phase, rng):
        c = np.zeros(len(phase))
        e = (phase >= PRE) & (phase <= P3)
        r = phase[e] - 1
        c[e] = np.minimum(np.exp(rng.standard_normal(r.size) * self.pp[r, CSIG] + self.pp[r, CMU]), self.pp[r, CMX])
        return c

    def price(self, phase, z, rng):
        r = phase - 1
        mu, sig, mx = self.pp[r, VMU], self.pp[r, VSIG], self.pp[r, VMX]
        e = rng.standard_normal(r.size)
        x = e if self.rho == 0 else math.sqrt(self.rho) * z + math.sqrt(1 - self.rho) * e
        return np.minimum(np.exp(x * sig + mu), mx) * self.eq

    def step_phase(self, phase, rng):
        if self.trans_rho == 0:
            u = rng.random(phase.size)
        else:
            zt = rng.standard_normal()
            u = ndtr(math.sqrt(self.trans_rho) * zt + math.sqrt(1 - self.trans_rho) * rng.standard_normal(phase.size))
        cd = np.cumsum(self.T[phase - 1], axis=1)
        cd[:, -1] = 1.0
        return (u[:, None] >= cd).sum(axis=1) + 1

    # --- one fund ------------------------------------------------------------------------------------------------
    def run_one(self, rng):
        p, H, NP, ic_pers = self.p, self.horizon, self.npers, self.p["ic_pers"]
        n = self.start.size
        phase = self.start.copy()
        invested = np.zeros(n, bool)
        acquired = np.zeros(n, bool)
        tip = np.zeros(n, int)
        cash = np.zeros(H + 2)
        cash[1] = p["initial_cash"]
        bv = self.nominal.copy()
        unpaid = 0.0
        in_default = False
        dflt = {"svc": False, "A1": False, "A2": False}

        cost = self.trial_cost(phase, rng)
        price_tmp = cost + self.pp[phase - 1, UPF]
        ps = self.sell - 1  # MATLAB psindx (1-based row of the phase before the sale phase)
        fce = (ps >= 1) * (self.pp[phase - 1, FCE] - self.pp[ps - 1, FCE])
        left = cash[1] - ic_pers * float((self.nominal[: self.nb] * self.coupon[: self.nb]).sum())
        for k in range(n):
            if left >= price_tmp[k] + fce[k]:
                left -= price_tmp[k] + fce[k]
                cash[1] -= price_tmp[k]
                invested[k] = acquired[k] = True
            else:
                phase[k] = DSC
        cash[2] = cash[1]
        z = rng.standard_normal()

        def sale(idx, t, adjust_uninvested):
            pr = self.price(phase[idx], z, rng)
            if adjust_uninvested:
                ni = ~invested[idx]
                pr[ni] = np.maximum(pr[ni] - cost[idx][ni], 0)
            return pr

        def book(idx, pr, t):
            cash[min(t + self.sale_lag, H)] += pr.sum()
            phase[idx] = SLD
            tip[idx] = 0

        def liquidate(t):
            idx = np.flatnonzero((phase != DSC) & (phase != SLD))
            if idx.size:
                pr = sale(idx, t, True)
                pr[tip[idx] == t - 1] *= p["ratio_unchanged"]  # never moved since the start (RBOMatlab rule)
                book(idx, pr, t)

        def svc_due():
            return self.svc * bv.sum() + unpaid * (1 + p["svc_accrual"])

        for t in range(2, NP + 1):
            cur = cash[t]
            old = phase.copy()
            elig = invested & (phase != DSC) & (phase != SLD) & (phase != APP)
            if elig.any():
                phase[elig] = self.step_phase(phase[elig], rng)
            sell = elig & ((phase == self.sell) | (phase == APP))
            if sell.any():
                idx = np.flatnonzero(sell)
                book(idx, sale(idx, t, True), t)
            dsc = elig & (phase == DSC)
            tip[dsc] = 0
            trans = elig & ~sell & ~dsc & (phase != old)
            if trans.any():
                nda = phase == NDA
                invested[nda & trans] = True
                m = ~nda & trans
                invested[m] = False
                tc = self.trial_cost(phase, rng)
                cost[m] = tc[m] + self.pp[phase[m] - 1, MIL]
                tip[trans] = 0
            stat = elig & ~sell & ~dsc & ~trans
            tip[stat] += 1

            if not in_default:  # default test on this period's opening cash
                due = svc_due()
                a1 = self.amort[0, t] + bv[0] * self.coupon[0]
                a2 = self.amort[1, t] + bv[1] * self.coupon[1]
                if cur < due:
                    dflt["svc"] = True
                    dflt["A1"] |= bv[0] > 0
                    dflt["A2"] |= bv[1] > 0
                elif cur < due + a1:
                    dflt["A1"] = True
                    dflt["A2"] |= bv[1] > 0
                elif cur < due + a1 + a2 and bv[1] > 0:
                    dflt["A2"] = True
                in_default = dflt["svc"] or dflt["A1"] or dflt["A2"]

            if not in_default:
                due = svc_due()
                paid = min(due, cur)
                unpaid = round((due - paid) * 1e5) / 1e5
                cur -= paid
                for b in range(self.nb):
                    i_paid = min(self.coupon[b] * bv[b], cur)
                    cur -= i_paid
                for b in range(self.nb):
                    pr_paid = min(self.amort[b, t], cur)
                    cur -= pr_paid
                    bv[b] -= pr_paid
                    if bv[b] < 1e-4:
                        bv[b] = 0.0
                ic_short = np.zeros(self.nb)
                for b in range(self.nb):
                    if bv[b] > 0:
                        k_cash = cur + cash[t + 1 : t + ic_pers + 1].sum()
                        due_b, k_left = self._ic(b, t, k_cash, bv, unpaid)
                        ic_short[b] = p["interest_coverage"][b] * due_b - k_left
                eq_due = unpaid + self.sale_lag * bv[-1] * self.svc
                short = max(ic_short.max(), eq_due - cur)
                if short > 0:
                    idx = np.flatnonzero((phase != DSC) & (phase != SLD))
                    if idx.size:
                        pr = sale(idx, t, True)
                        order = np.argsort(-pr, kind="stable")
                        cum = np.cumsum(pr[order])
                        hit = np.flatnonzero(cum >= short)
                        nsold = hit[0] + 1 if hit.size else order.size
                        chosen = order[:nsold]
                        book(idx[chosen], pr[chosen], t)
                elif t < NP:
                    nxt = list(range(t + 1, t + ic_pers + 1)) if t < NP - ic_pers else [t + 1]
                    need = self.amort[:, nxt].sum() + len(nxt) * (
                        float((self.coupon[: self.nb] * bv[: self.nb]).sum()) + self.svc * bv.sum()
                    )
                    avail = max(cur - need, 0.0)
                    if avail > 0:
                        spent = 0.0
                        for k in np.flatnonzero(
                            (phase != DSC) & (phase != APP) & (phase != SLD) & acquired & ~invested
                        ):
                            if cost[k] <= avail:
                                invested[k] = True
                                avail -= cost[k]
                                spent += cost[k]
                        cur -= spent
            else:
                liquidate(t)
                for b in range(self.nb):
                    if cur > 0:
                        i_paid = min(cur, self.coupon[b] * bv[b])
                        cur -= i_paid
                        pr_paid = min(cur, bv[b])
                        bv[b] -= pr_paid
                        if bv[b] < 1e-4:
                            bv[b] = 0.0
                        cur -= pr_paid
                if not dflt["A1"] and dflt["A2"] and bv[0] > 0:
                    dflt["A1"] = True
            cash[t] = cur
            if t != NP:
                cash[t + 1] += cur * (1 + p["cash_accrual"])

        liquidate(NP)
        for t in range(NP, H):
            cur = cash[t]
            if not in_default:
                due = svc_due()
                paid = min(due, cur)
                unpaid = round((due - paid) * 1e5) / 1e5
                cur -= paid
            cash[t] = cur
            cash[t + 1] = cash[t] + cash[t + 1]
        roe = max((cash[H] - self.nominal[-1]) / self.nominal[-1], -1.0)
        return roe, dflt["A1"], dflt["A2"], dflt["svc"]

    def _ic(self, b, t, k_cash, bv, unpaid):
        """RBOMatlab ic_test_result: cash due on tranche b over the next IC_pers periods, and the cash left for it after
        the senior tranches' scheduled payments."""
        NP, ic_pers = self.npers, self.p["ic_pers"]
        if t == NP:
            return 0.0, k_cash
        n_ahead = min(ic_pers, NP - t)
        k = k_cash
        for bb in range(b):
            btemp = bv[bb]
            for m in range(1, n_ahead + 1):
                k -= self.amort[bb, t + m] + self.coupon[bb] * btemp + self.svc * btemp
                btemp -= self.amort[bb, t + m]
        sp = si = 0.0
        ss = unpaid
        bvb = bv[b]
        for m in range(1, n_ahead + 1):
            sp += self.amort[b, t + m]
            si += self.coupon[b] * bvb
            ss += self.svc * bvb
            bvb -= self.amort[b, t + m]
        return sp + si + ss, k

    def _chunk(self, args):
        n, child = args
        rng = np.random.default_rng(child)
        return [self.run_one(rng) for _ in range(n)]

    def run(self, nsims, seed, workers=1, chunks=20):
        """`chunks` fixed seed streams (SeedSequence(seed).spawn), so the result does not depend on `workers`."""
        kids = np.random.SeedSequence(seed).spawn(chunks)
        sizes = [nsims // chunks + (1 if i < nsims % chunks else 0) for i in range(chunks)]
        jobs = [(s, k) for s, k in zip(sizes, kids) if s]
        if workers > 1:
            import multiprocessing as mp

            with mp.get_context("spawn").Pool(workers) as pool:
                parts = pool.map(self._chunk, jobs)
        else:
            parts = [self._chunk(j) for j in jobs]
        out = np.array([r for part in parts for r in part], dtype=float)
        roe = out[:, 0]
        years = self.horizon / 2
        ann = (1 + roe) ** (1 / years) - 1
        return {
            "nsims": nsims,
            "seed": seed,
            "years": years,
            "roe_ann_mean": float(ann.mean()),
            "roe_ann_se": float(ann.std(ddof=1) / math.sqrt(nsims)),
            "roe_ann_sd": float(ann.std(ddof=1)),
            "roe_cum_mean": float(roe.mean()),
            "p_equity_loss": float((roe < 0).mean()),
            "p_default_A1": float(out[:, 1].mean()),
            "p_default_A2": float(out[:, 2].mean()),
            "p_default_svc": float(out[:, 3].mean()),
        }
